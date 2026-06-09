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
python scripts/run_experiments.py
```

Rode o teste de gradiente:

```bash
python tests/test_gradients.py
pytest
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

Depois de rodar os experimentos, preencha esta tabela com os valores salvos em `results/experiments.csv`.

| Experimento | Arquitetura | Ativacao | Learning rate | Momentum | Accuracy | Recall macro | F1 macro |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_relu | 784-256-128-10 | ReLU | 0.05 | 0.9 | preencher apos treino | preencher apos treino | preencher apos treino |
| deeper_relu | 784-256-128-64-10 | ReLU | 0.03 | 0.9 | preencher apos treino | preencher apos treino | preencher apos treino |

Arquivos esperados:

- `results/baseline_relu_curves.png`: curva de loss e acuracia.
- `results/baseline_relu_confusion.png`: matriz de confusao.
- `results/baseline_relu_metrics.json`: metricas finais.
- `results/baseline_relu_classification_report.csv`: precision, recall e F1 por digito.
- `results/experiments.csv`: tabela comparativa.

A meta do trabalho e atingir acuracia de teste maior ou igual a 92%. Caso a primeira execucao fique abaixo disso, aumente o numero de epocas para 20 ou ajuste o learning rate entre `0.03` e `0.08`.

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
- [ ] Acuracia final >= 92% deve ser confirmada depois de rodar `python scripts/run_experiments.py` ou a primeira celula do notebook.

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

