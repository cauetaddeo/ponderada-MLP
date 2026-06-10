import numpy as np

from .activations import get_activation
from .losses import accuracy, cross_entropy, one_hot, softmax
from .optimizers import SGD


class MLP:
    def __init__(self, layer_sizes, activation="relu", seed=42):
        if len(layer_sizes) < 3:
            raise ValueError("Use pelo menos entrada, uma camada oculta e saida.")

        # A lista define entrada, camadas ocultas e saida.
        self.layer_sizes = list(layer_sizes)
        self.activation_name = activation
        self.activation, self.activation_derivative = get_activation(activation)
        self.rng = np.random.default_rng(seed)
        self.params = {}
        self._init_params()

    def _init_params(self):
        # Cria pesos W e biases b para cada transicao entre camadas.
        for layer_idx, (fan_in, fan_out) in enumerate(
            zip(self.layer_sizes[:-1], self.layer_sizes[1:]), start=1
        ):
            # Inicializacao He para ReLU ajuda a manter a escala das ativacoes.
            scale = np.sqrt(2.0 / fan_in) if self.activation_name == "relu" else np.sqrt(1.0 / fan_in)
            self.params[f"W{layer_idx}"] = self.rng.normal(0.0, scale, size=(fan_in, fan_out))
            self.params[f"b{layer_idx}"] = np.zeros((1, fan_out), dtype=np.float64)

    @property
    def n_layers(self):
        return len(self.layer_sizes) - 1

    def forward(self, x):
        # Cache guarda valores intermediarios usados depois no backward.
        cache = {"A0": x}
        activation_input = x

        # Camadas ocultas usam a ativacao configurada.
        for layer_idx in range(1, self.n_layers):
            z = activation_input @ self.params[f"W{layer_idx}"] + self.params[f"b{layer_idx}"]
            activation_input = self.activation(z)
            cache[f"Z{layer_idx}"] = z
            cache[f"A{layer_idx}"] = activation_input

        # A camada final usa softmax para gerar probabilidades.
        output_idx = self.n_layers
        logits = activation_input @ self.params[f"W{output_idx}"] + self.params[f"b{output_idx}"]
        probs = softmax(logits)
        cache[f"Z{output_idx}"] = logits
        cache[f"A{output_idx}"] = probs
        return probs, cache

    def compute_loss(self, x, y):
        probs, _ = self.forward(x)
        return cross_entropy(probs, y)

    def backward(self, cache, y):
        n_samples = y.shape[0]
        grads = {}
        y_encoded = one_hot(y, self.layer_sizes[-1])

        # Softmax + cross-entropy simplifica o gradiente inicial.
        d_z = (cache[f"A{self.n_layers}"] - y_encoded) / n_samples
        for layer_idx in range(self.n_layers, 0, -1):
            a_prev = cache[f"A{layer_idx - 1}"]
            # Gradientes dos pesos e biases da camada atual.
            grads[f"W{layer_idx}"] = a_prev.T @ d_z
            grads[f"b{layer_idx}"] = np.sum(d_z, axis=0, keepdims=True)

            if layer_idx > 1:
                # Propaga o gradiente para a camada anterior.
                d_a_prev = d_z @ self.params[f"W{layer_idx}"].T
                d_z = d_a_prev * self.activation_derivative(cache[f"Z{layer_idx - 1}"])

        return grads

    def fit(
        self,
        x_train,
        y_train,
        x_val=None,
        y_val=None,
        epochs=10,
        batch_size=128,
        learning_rate=0.01,
        momentum=0.0,
        shuffle=True,
        verbose=True,
    ):
        optimizer = SGD(learning_rate=learning_rate, momentum=momentum)
        # Historico usado para graficos e analise de overfitting.
        history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
        n_samples = x_train.shape[0]

        for epoch in range(1, epochs + 1):
            # Embaralhar evita que a ordem dos dados influencie os mini-batches.
            indices = np.arange(n_samples)
            if shuffle:
                self.rng.shuffle(indices)

            # Cada mini-batch faz forward, backward e atualizacao de pesos.
            for start in range(0, n_samples, batch_size):
                batch_idx = indices[start : start + batch_size]
                x_batch = x_train[batch_idx]
                y_batch = y_train[batch_idx]
                _, cache = self.forward(x_batch)
                grads = self.backward(cache, y_batch)
                optimizer.step(self.params, grads)

            # Ao final da epoca, mede o desempenho no conjunto de treino.
            train_probs, _ = self.forward(x_train)
            train_loss = cross_entropy(train_probs, y_train)
            train_acc = accuracy(train_probs, y_train)
            history["loss"].append(train_loss)
            history["accuracy"].append(train_acc)

            message = f"epoch={epoch:02d} loss={train_loss:.4f} acc={train_acc:.4f}"
            if x_val is not None and y_val is not None:
                # Validacao ajuda a observar generalizacao e overfitting.
                val_probs, _ = self.forward(x_val)
                val_loss = cross_entropy(val_probs, y_val)
                val_acc = accuracy(val_probs, y_val)
                history["val_loss"].append(val_loss)
                history["val_accuracy"].append(val_acc)
                message += f" val_loss={val_loss:.4f} val_acc={val_acc:.4f}"

            if verbose:
                print(message)

        return history

    def predict_proba(self, x):
        probs, _ = self.forward(x)
        return probs

    def predict(self, x):
        return np.argmax(self.predict_proba(x), axis=1)

    def evaluate(self, x, y):
        probs = self.predict_proba(x)
        return {"loss": cross_entropy(probs, y), "accuracy": accuracy(probs, y)}
