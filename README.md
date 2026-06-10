# MLP do Zero no MNIST

Este projeto implementa um Multi-Layer Perceptron usando apenas NumPy para os calculos da rede neural. O carregamento do MNIST tenta usar Keras quando disponivel; se Keras/TensorFlow nao estiver instalado, o script baixa os arquivos IDX oficiais do MNIST diretamente.

## Como rodar

Crie e ative um ambiente virtual:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Treine a configuracao principal:

```bash
python -m scripts.train_mnist --run-name baseline_relu --layers 784-256-128-10 --epochs 15 --learning-rate 0.05 --momentum 0.9
```

Rode a comparacao entre duas arquiteturas:

```bash
python -m scripts.run_experiments
```

Rode o teste de gradiente:

```bash
python tests/test_gradients.py
python -m pytest tests/test_gradients.py
```

Se `pytest` ainda nao estiver instalado, o comando direto `python tests/test_gradients.py` ja valida o backpropagation.

Os arquivos gerados ficam em `results/`: metricas em JSON, curvas de loss/acuracia, matriz de confusao e tabela de experimentos.

## Arquitetura escolhida

A configuracao principal e:

- Entrada: 784 valores, um para cada pixel normalizado da imagem 28x28.
- Camada oculta 1: 256 neuronios com ReLU.
- Camada oculta 2: 128 neuronios com ReLU.
- Saida: 10 neuronios com softmax, um por digito.
- Loss: cross-entropy.
- Otimizador: SGD com learning rate configuravel e momentum opcional.

Escolhi duas camadas ocultas porque o enunciado pede ao menos duas e porque essa arquitetura ainda e pequena o suficiente para treinar rapido em NumPy. A ReLU simplifica os gradientes, reduz saturacao em comparacao com `tanh` e combina bem com inicializacao He. A saida usa softmax + cross-entropy porque esse par transforma os logits em probabilidades e produz um gradiente simples: `probabilidades - rotulos_one_hot`.

## Resultados

Estes foram os resultados obtidos no conjunto de teste do MNIST:

| Experimento | Arquitetura | Ativacao | Learning rate | Momentum | Accuracy | Recall macro | F1 macro |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_relu | 784-256-128-10 | ReLU | 0.05 | 0.9 | 0.9812 | 0.9810 | 0.9811 |
| deeper_relu | 784-256-128-64-10 | ReLU | 0.03 | 0.9 | 0.9823 | 0.9821 | 0.9821 |

Tambem acompanhei o possivel overfitting comparando treino, validacao e teste:

| Experimento | Train accuracy | Validation accuracy | Test accuracy | Gap treino-validacao |
| --- | --- | --- | --- | --- |
| baseline_relu | 1.0000 | 0.9810 | 0.9812 | 0.0190 |
| deeper_relu | 1.0000 | 0.9818 | 0.9823 | 0.0182 |

Arquivos esperados:

- `results/baseline_relu_curves.png`: curva de loss e acuracia.
- `results/baseline_relu_confusion.png`: matriz de confusao.
- `results/baseline_relu_metrics.json`: metricas finais.
- `results/baseline_relu_classification_report.csv`: precision, recall e F1 por digito.
- `results/experiments.csv`: tabela comparativa.

A meta do trabalho era atingir acuracia de teste maior ou igual a 92%. As duas configuracoes passaram essa meta com folga.

### Curvas de treinamento

As curvas abaixo mostram a evolucao da loss e da acuracia ao longo das epocas. A loss cai de forma consistente e a acuracia de treino sobe ate `1.0000`, enquanto a validacao estabiliza perto de `0.98`. Essa diferenca indica overfitting leve: a rede memorizou o treino, mas ainda manteve boa generalizacao no conjunto de validacao e teste.

**baseline_relu**

![Curvas de loss e acuracia da baseline](results/baseline_relu_curves.png)

**deeper_relu**

![Curvas de loss e acuracia da rede mais profunda](results/deeper_relu_curves.png)

### Matrizes de confusao

As matrizes de confusao mostram quantas imagens de cada digito foram classificadas corretamente ou confundidas com outros digitos. A diagonal principal concentra quase todos os exemplos, o que confirma o bom desempenho. Ainda assim, existem erros fora da diagonal: a `baseline_relu` errou 188 imagens de teste e a `deeper_relu` errou 177.

**baseline_relu**

![Matriz de confusao da baseline](results/baseline_relu_confusion.png)

**deeper_relu**

![Matriz de confusao da rede mais profunda](results/deeper_relu_confusion.png)

## Checklist dos requisitos

- [x] Forward pass para numero arbitrario de camadas.
- [x] Ao menos 2 camadas ocultas nas configuracoes principais.
- [x] Ativacao configuravel (`relu` e `tanh`).
- [x] Softmax + cross-entropy na saida.
- [x] Backpropagation manual.
- [x] Mini-batches com SGD e learning rate configuravel.
- [x] Plot de loss e acuracia ao longo do treinamento.
- [x] Comparacao de 2 configuracoes.
- [x] Matriz de confusao.
- [x] Precision, recall, F1, balanced accuracy e metricas por digito.
- [x] Teste de gradiente numerico.
- [x] Historico com mais de 6 commits descritivos.
- [x] Acuracia final >= 92% confirmada nos resultados salvos.

## Decisoes e dificuldades

1. A decisao tecnica mais dificil foi escolher uma arquitetura que passasse de 92% sem transformar o projeto em uma caixa-preta dificil de explicar. Eu escolhi `784-256-128-10` como baseline porque ela tem duas camadas ocultas, aprende bem o MNIST e ainda deixa o backpropagation facil de acompanhar. Depois comparei com `784-256-128-64-10` para testar se uma camada extra traria ganho real.

2. O que mais me chamou atencao foi que, com 15 epocas, a acuracia de treino chegou a `1.0000`, mas validacao e teste ficaram perto de `0.982`. No inicio isso parecia erro de metrica, mas a matriz de confusao mostrou que ainda havia erros no teste. Aprendi a olhar treino, validacao e teste juntos antes de concluir que o modelo esta perfeito.

3. Tambem tive dificuldade com o ambiente de dados. O codigo tenta carregar o MNIST por `keras.datasets.mnist`, mas no meu ambiente local o TensorFlow/Keras nao estava disponivel. Por isso deixei um fallback que baixa os arquivos IDX oficiais do MNIST, mantendo o treinamento da rede todo em NumPy.

4. Se eu fosse refazer do zero, eu implementaria early stopping e talvez regularizacao L2 desde o inicio. Isso ajudaria a evitar que a rede memorizasse totalmente o treino nas ultimas epocas, mesmo mantendo a acuracia de teste acima da meta.

## Implementacao

O forward pass aceita um numero arbitrario de camadas. Para cada camada oculta, o codigo calcula:

```python
Z = A_anterior @ W + b
A = relu(Z)
```

Na camada final, calcula os logits e aplica softmax. No backward, o gradiente inicial da saida e:

```python
dZ = (probs - y_one_hot) / batch_size
```

Em seguida, para cada camada, os gradientes sao:

```python
dW = A_anterior.T @ dZ
db = soma(dZ)
dZ_anterior = (dZ @ W.T) * derivada_relu(Z_anterior)
```

Essa divisao por `batch_size` deixa a escala do gradiente estavel quando o tamanho do mini-batch muda.

## Estrutura

```text
.
|-- README.md
|-- mlp/
|   |-- __init__.py
|   |-- activations.py
|   |-- losses.py
|   |-- network.py
|   `-- optimizers.py
|-- notebooks/
|   `-- experimentos.ipynb
|-- results/
|-- scripts/
|   |-- run_experiments.py
|   `-- train_mnist.py
|-- tests/
|   `-- test_gradients.py
`-- requirements.txt
```
