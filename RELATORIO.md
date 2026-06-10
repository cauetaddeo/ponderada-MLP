# Relatorio do projeto MLP do Zero

Este relatorio explica o que foi implementado, por que cada arquivo existe e como as principais funcoes trabalham. A ideia e servir como guia de estudo para voce conseguir defender o codigo e entender onde cada requisito do notebook foi atendido.

## Problema de importacao corrigido

O erro era:

```text
ModuleNotFoundError: No module named 'mlp'
```

Ele acontecia quando o teste era executado assim:

```bash
python tests/test_gradients.py
```

Nesse modo, o Python coloca a pasta `tests/` no inicio do `sys.path`. Como o pacote `mlp/` fica na raiz do projeto, e nao dentro de `tests/`, o import `from mlp import MLP` falhava.

A correcao foi adicionar, no inicio de `tests/test_gradients.py`, a raiz do projeto ao `sys.path`:

```python
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
```

Tambem adicionei um bloco:

```python
if __name__ == "__main__":
    test_gradient_check_small_network()
    print("gradient check ok")
```

Assim, o mesmo arquivo funciona dos dois jeitos:

```bash
python tests/test_gradients.py
pytest tests/test_gradients.py
```

Se aparecer `No module named 'numpy'` ou `No module named 'pytest'`, o problema ja nao e importacao do projeto: significa que as dependencias ainda nao foram instaladas no ambiente virtual.

## Como rodar depois de instalar dependencias

Ative o ambiente virtual:

```bash
.venv\Scripts\activate
```

Instale as dependencias:

```bash
pip install -r requirements.txt
```

Rode o gradient check diretamente:

```bash
python tests/test_gradients.py
```

Ou rode com pytest:

```bash
pytest tests/test_gradients.py
```

Treine a rede:

```bash
python -m scripts.train_mnist --run-name baseline_relu --layers 784-256-128-10 --epochs 15 --learning-rate 0.05 --momentum 0.9
```

Compare duas configuracoes:

```bash
python -m scripts.run_experiments
```

## Estrutura geral

```text
.
|-- mlp/
|   |-- __init__.py
|   |-- activations.py
|   |-- losses.py
|   |-- network.py
|   `-- optimizers.py
|-- scripts/
|   |-- train_mnist.py
|   `-- run_experiments.py
|-- tests/
|   `-- test_gradients.py
|-- notebooks/
|   `-- experimentos.ipynb
|-- README.md
|-- RELATORIO.md
`-- requirements.txt
```

## Arquivo `mlp/__init__.py`

Esse arquivo transforma a pasta `mlp/` em um pacote Python e expõe a classe principal:

```python
from .network import MLP
```

Com isso, outros arquivos podem importar a rede de forma simples:

```python
from mlp import MLP
```

## Arquivo `mlp/activations.py`

Este arquivo guarda as funcoes de ativacao das camadas ocultas.

### `relu(z)`

Recebe uma matriz `z` e troca valores negativos por zero:

```python
return np.maximum(0.0, z)
```

A ReLU foi usada porque e simples, rapida e reduz o problema de saturacao que aparece em funcoes como sigmoid/tanh.

### `relu_derivative(z)`

Calcula a derivada da ReLU:

- Se `z > 0`, derivada e `1`.
- Se `z <= 0`, derivada e `0`.

Essa derivada e usada no backpropagation para saber quanto do gradiente passa para a camada anterior.

### `tanh(z)` e `tanh_derivative(z)`

Foram incluidas para permitir comparacao com outra ativacao. A `tanh` comprime os valores entre `-1` e `1`, mas pode saturar quando os valores ficam muito grandes ou muito pequenos.

### `get_activation(name)`

Recebe o nome de uma ativacao, como `"relu"` ou `"tanh"`, e devolve duas funcoes:

```python
(funcao_de_ativacao, derivada_da_ativacao)
```

Isso torna a ativacao configuravel, como o notebook pedia.

## Arquivo `mlp/losses.py`

Este arquivo guarda funcoes relacionadas a saida da rede, loss e metricas.

### `softmax(logits)`

Transforma os logits da camada final em probabilidades. O codigo subtrai o maior valor de cada linha antes do `exp`:

```python
shifted = logits - np.max(logits, axis=1, keepdims=True)
```

Isso evita overflow numerico quando algum logit e muito alto.

### `cross_entropy(probs, y_true)`

Calcula a penalizacao da previsao. Se a rede atribui probabilidade alta para a classe correta, a loss diminui. Se atribui probabilidade baixa, a loss aumenta.

O `np.clip` evita `log(0)`, que causaria infinito:

```python
clipped = np.clip(probs, 1e-12, 1.0)
```

### `accuracy(probs, y_true)`

Pega a classe de maior probabilidade com `argmax` e compara com o rotulo real.

### `one_hot(y, n_classes)`

Converte rotulos como:

```text
[3, 0, 9]
```

em uma matriz one-hot. Isso facilita calcular o gradiente da softmax com cross-entropy:

```python
dZ = (probs - y_one_hot) / batch_size
```

## Arquivo `mlp/optimizers.py`

Este arquivo implementa o otimizador.

### Classe `SGD`

SGD significa Stochastic Gradient Descent. Ele atualiza pesos e vieses na direcao oposta ao gradiente:

```python
params[key] -= learning_rate * grad
```

Tambem inclui momentum opcional. O momentum acumula parte da atualizacao anterior para suavizar o caminho do treinamento:

```python
velocity = momentum * velocity - learning_rate * grad
params[key] += velocity
```

## Arquivo `mlp/network.py`

Este e o arquivo principal do projeto.

### Classe `MLP`

Representa a rede neural inteira.

### `__init__(layer_sizes, activation="relu", seed=42)`

Recebe a arquitetura como lista:

```python
[784, 256, 128, 10]
```

Isso significa:

- 784 entradas, uma por pixel do MNIST.
- 256 neuronios na primeira camada oculta.
- 128 neuronios na segunda camada oculta.
- 10 saidas, uma para cada digito.

### `_init_params()`

Inicializa pesos e biases. Para ReLU, usei inicializacao He:

```python
scale = np.sqrt(2.0 / fan_in)
```

Essa escolha ajuda a manter a escala das ativacoes estavel ao longo das camadas. Os biases comecam em zero.

### `forward(x)`

Executa a passagem para frente.

Para cada camada oculta:

```python
Z = A_anterior @ W + b
A = ativacao(Z)
```

Na camada final:

```python
logits = A_anterior @ W + b
probs = softmax(logits)
```

A funcao tambem guarda um `cache` com `A` e `Z` de cada camada. Esse cache e essencial para o backpropagation.

### `compute_loss(x, y)`

Faz o forward e calcula a cross-entropy. E usada tambem no gradient check numerico.

### `backward(cache, y)`

Implementa o backpropagation manualmente.

Na saida, como estamos usando softmax + cross-entropy, o gradiente fica:

```python
d_z = (probs - y_one_hot) / n_samples
```

Depois, para cada camada:

```python
dW = A_anterior.T @ dZ
db = soma(dZ)
dA_anterior = dZ @ W.T
dZ_anterior = dA_anterior * derivada_ativacao(Z_anterior)
```

Essa e a parte mais importante do projeto, porque mostra que o treinamento nao esta usando PyTorch, TensorFlow ou autograd.

### `fit(...)`

Treina a rede por varias epocas. A cada epoca:

1. Embaralha os indices do treino.
2. Divide os dados em mini-batches.
3. Faz forward no batch.
4. Calcula os gradientes com `backward`.
5. Atualiza pesos com SGD.
6. Calcula loss/acuracia de treino e validacao.

### `predict_proba(x)`

Retorna as probabilidades da softmax.

### `predict(x)`

Retorna a classe mais provavel.

### `evaluate(x, y)`

Calcula loss e acuracia em um conjunto, como teste ou validacao.

## Arquivo `scripts/train_mnist.py`

Este script executa um treino completo no MNIST.

### `load_mnist(validation_size=10000)`

Carrega o MNIST. Primeiro tenta usar `keras.datasets.mnist`. Se Keras/TensorFlow nao estiver instalado, usa o fallback `load_mnist_from_idx()`, que baixa os arquivos IDX oficiais do MNIST para `data/mnist/`.

Depois disso, normaliza os pixels para o intervalo `[0, 1]`, transforma imagens 28x28 em vetores de 784 valores e separa uma parte do treino para validacao.

### `load_mnist_from_idx()`

Baixa e le os quatro arquivos oficiais do MNIST:

- imagens de treino;
- labels de treino;
- imagens de teste;
- labels de teste.

Esse fallback foi adicionado porque seu ambiente virtual estava em Python 3.14 e nao tinha `keras/tensorflow` disponivel. Assim o notebook consegue rodar sem depender de TensorFlow.

### `parse_layers(raw)`

Converte uma string como:

```text
784-256-128-10
```

em:

```python
[784, 256, 128, 10]
```

### `plot_history(history, output_path)`

Gera um grafico com:

- Loss por epoca.
- Acuracia por epoca.

O arquivo e salvo em `results/`.

### `confusion_matrix(y_true, y_pred, n_classes=10)`

Cria a matriz de confusao. A linha representa o rotulo real e a coluna representa a predicao da rede.

### `plot_confusion_matrix(matrix, output_path)`

Gera uma imagem da matriz de confusao.

### `main()`

Organiza o fluxo do script:

1. Le argumentos da linha de comando.
2. Carrega MNIST.
3. Cria o MLP.
4. Treina.
5. Avalia no teste.
6. Salva metricas, curvas e matriz de confusao.

## Arquivo `scripts/run_experiments.py`

Este script roda pelo menos duas configuracoes, atendendo ao requisito de comparacao do notebook.

As configuracoes atuais sao:

- `baseline_relu`: `784-256-128-10`.
- `deeper_relu`: `784-256-128-64-10`.

Para cada experimento, ele chama `scripts.train_mnist` com argumentos diferentes. Depois le o JSON de metricas e cria `results/experiments.csv`.

## Arquivo `tests/test_gradients.py`

Este teste verifica se o backpropagation esta correto.

Ele cria uma rede pequena:

```python
MLP([4, 6, 5, 3])
```

Depois compara:

- Gradiente analitico: calculado pelo metodo `backward`.
- Gradiente numerico: aproximado por diferenca finita.

A formula numerica e:

```python
(loss_plus - loss_minus) / (2 * epsilon)
```

Se os dois valores ficam proximos, o gradiente provavelmente esta correto.

## Arquivo `notebooks/experimentos.ipynb`

O notebook serve para registrar e visualizar os experimentos. Ele nao contem a implementacao principal da rede, porque o codigo importante fica no pacote `mlp/`.

Use o notebook para:

- Rodar `scripts/run_experiments.py`.
- Abrir `results/experiments.csv`.
- Visualizar curvas e matriz de confusao.
- Escrever a analise final dos resultados.

## Arquivo `README.md`

O README explica como instalar, rodar, treinar, comparar experimentos e interpretar os resultados. A secao "Decisoes e dificuldades" esta preenchida em primeira pessoa com as principais escolhas e problemas encontrados.

## O que eu fiz no projeto

1. Criei a estrutura exigida pelo notebook.
2. Implementei MLP do zero com NumPy.
3. Implementei ReLU/tanh e derivadas.
4. Implementei softmax e cross-entropy.
5. Implementei backpropagation manual.
6. Implementei SGD com learning rate configuravel e momentum opcional.
7. Criei script para treinar no MNIST.
8. Criei script para comparar duas arquiteturas.
9. Criei gradient check numerico.
10. Corrigi o problema de importacao do teste.
11. Criei README e notebook de experimentos.
12. Criei commits descritivos para atender ao requisito de historico.
13. Adicionei metricas de avaliacao: accuracy, precision, recall, F1, balanced accuracy e relatorio por digito.
14. Atualizei o notebook para imprimir tabelas de metricas de cada experimento.

## Verificacao dos requisitos do notebook

| Requisito | Status | Onde esta |
| --- | --- | --- |
| Forward pass com numero arbitrario de camadas | Cumprido | `mlp/network.py`, metodo `forward` |
| Ao menos 2 camadas ocultas | Cumprido | Configuracoes `784-256-128-10` e `784-256-128-64-10` |
| Ativacao configuravel | Cumprido | `mlp/activations.py` e argumento `--activation` |
| Softmax + cross-entropy | Cumprido | `mlp/losses.py` |
| Backpropagation manual | Cumprido | `mlp/network.py`, metodo `backward` |
| Mini-batches com SGD | Cumprido | `mlp/network.py` e `mlp/optimizers.py` |
| Learning rate configuravel | Cumprido | argumento `--learning-rate` |
| Acuracia >= 92% | Cumprido | `baseline_relu` = 0.9812 e `deeper_relu` = 0.9823 |
| Plot de loss e acuracia | Cumprido | `results/*_curves.png` |
| Comparacao de 2 configuracoes | Cumprido | `scripts/run_experiments.py` |
| README com secoes obrigatorias | Cumprido | `README.md` |
| Decisoes e dificuldades em primeira pessoa | Cumprido | `README.md` |
| Historico de commits minimo 6 | Cumprido | `git log` |
| Gradient check numerico | Cumprido | `tests/test_gradients.py` |
| Matriz de confusao comentavel | Cumprido | `results/*_confusion.png` |

## Diagnostico de overfitting

Ao rodar os experimentos completos, a acuracia de treino chegou a `1.0000`. Isso nao significa que a rede acertou 100% no teste. Os resultados salvos em `results/` mostram:

| Experimento | Train accuracy | Validation accuracy | Test accuracy | Gap treino-validacao |
| --- | --- | --- | --- | --- |
| baseline_relu | 1.0000 | 0.9810 | 0.9812 | 0.0190 |
| deeper_relu | 1.0000 | 0.9818 | 0.9823 | 0.0182 |

Minha interpretacao: existe overfitting leve, porque a rede memorizou o conjunto de treino ao final das 15 epocas, mas a diferenca para validacao/teste fica perto de 2 pontos percentuais. Nao parece haver vazamento do conjunto de teste, porque a matriz de confusao ainda tem erros: `baseline_relu` errou 188 dos 10.000 exemplos de teste e `deeper_relu` errou 177.

Para reduzir esse overfitting, as opcoes mais simples seriam:

- treinar por menos epocas, por exemplo parar perto da melhor validacao;
- diminuir um pouco a arquitetura;
- adicionar regularizacao L2;
- adicionar early stopping;
- usar data augmentation, se fosse permitido/necessario.

## Como explicar a decisao tecnica principal

A decisao mais importante foi deixar a rede com arquitetura configuravel. Em vez de escrever uma rede fixa com exatamente duas camadas ocultas, `network.py` percorre uma lista de tamanhos. Isso permite testar arquiteturas diferentes sem reescrever o backpropagation.

Exemplo:

```python
MLP([784, 256, 128, 10])
MLP([784, 256, 128, 64, 10])
```

O mesmo codigo treina as duas.

## Possiveis erros e causas

### `No module named 'mlp'`

Era o erro de caminho de importacao. Foi corrigido em `tests/test_gradients.py`.

### `No module named 'numpy'`

Significa que as dependencias nao foram instaladas no ambiente virtual.

Resolva com:

```bash
pip install -r requirements.txt
```

### `No module named 'pytest'`

Mesmo caso: falta instalar dependencias. Enquanto isso, voce pode rodar:

```bash
python tests/test_gradients.py
```

### `No module named 'matplotlib'`

O script de treino usa matplotlib para salvar as curvas. Instale as dependencias.

### Erros ao carregar MNIST

O codigo tenta Keras primeiro. Se Keras nao existir, baixa o MNIST diretamente dos arquivos IDX oficiais. Se houver erro nessa parte, provavelmente e problema de conexao para baixar o dataset ou de permissao para escrever em `data/mnist/`.
