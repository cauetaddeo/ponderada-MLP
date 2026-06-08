# MLP do Zero no MNIST

Este projeto implementa um Multi-Layer Perceptron usando apenas NumPy para os calculos da rede neural. Keras e usado somente para baixar/carregar o dataset MNIST, como permitido no enunciado.

## Como rodar

Crie e ative um ambiente virtual:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Treine a configuracao principal:

```bash
python scripts/train_mnist.py --run-name baseline_relu --layers 784,256,128,10 --epochs 15 --learning-rate 0.05 --momentum 0.9
```

Rode a comparacao entre duas arquiteturas:

```bash
python scripts/run_experiments.py
```

Rode o teste de gradiente:

```bash
pytest
```

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

Depois de rodar os experimentos, preencha esta tabela com os valores salvos em `results/experiments.csv`.

| Experimento | Arquitetura | Ativacao | Learning rate | Momentum | Acuracia teste |
| --- | --- | --- | --- | --- | --- |
| baseline_relu | 784-256-128-10 | ReLU | 0.05 | 0.9 | preencher apos treino |
| deeper_relu | 784-256-128-64-10 | ReLU | 0.03 | 0.9 | preencher apos treino |

Arquivos esperados:

- `results/baseline_relu_curves.png`: curva de loss e acuracia.
- `results/baseline_relu_confusion.png`: matriz de confusao.
- `results/baseline_relu_metrics.json`: metricas finais.
- `results/experiments.csv`: tabela comparativa.

A meta do trabalho e atingir acuracia de teste maior ou igual a 92%. Caso a primeira execucao fique abaixo disso, aumente o numero de epocas para 20 ou ajuste o learning rate entre `0.03` e `0.08`.

## Decisoes e dificuldades

Esta secao deve ser ajustada por mim depois dos experimentos, porque a avaliacao pede uma reflexao pessoal. Abaixo esta o rascunho que vou completar com o que aconteceu durante o treino.

1. A decisao tecnica mais dificil foi escolher uma arquitetura que fosse forte o bastante para passar de 92% no MNIST, mas ainda simples para eu conseguir explicar cada etapa do backpropagation. Eu escolhi `784-256-128-10` porque ela tem capacidade suficiente sem deixar o treino em NumPy pesado demais.

2. Uma coisa que eu testei e que precisa ser observada foi a sensibilidade ao learning rate. Se o valor for alto demais, a loss oscila; se for baixo demais, a rede aprende, mas demora muito. O teste de gradiente ajudou a separar erro de implementacao de simples escolha ruim de hiperparametro.

3. Se eu fosse refazer do zero, eu comecaria com o teste de gradiente antes de treinar no MNIST inteiro. Isso encurta o ciclo de debug, porque um erro pequeno no backward pode parecer apenas "treino ruim" quando olhado so pela acuracia.

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

## Commits

O enunciado pede pelo menos 6 commits com mensagens descritivas. Uma sequencia adequada e:

```bash
git add mlp/activations.py mlp/losses.py mlp/network.py mlp/optimizers.py mlp/__init__.py
git commit -m "feat: implementa mlp com forward e backprop manual"
git add scripts/train_mnist.py
git commit -m "feat: adiciona treino do mnist e salvamento de metricas"
git add scripts/run_experiments.py
git commit -m "feat: adiciona comparacao de arquiteturas"
git add tests/test_gradients.py
git commit -m "test: valida gradientes com aproximacao numerica"
git add notebooks/experimentos.ipynb
git commit -m "docs: registra experimentos no notebook"
git add README.md requirements.txt results/
git commit -m "docs: documenta resultados e decisoes do projeto"
```
