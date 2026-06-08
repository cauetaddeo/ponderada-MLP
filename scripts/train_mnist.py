import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlp import MLP


def load_mnist(validation_size=10000):
    try:
        from keras.datasets import mnist
    except ImportError as exc:
        raise ImportError(
            "Instale as dependencias com `pip install -r requirements.txt`. "
            "O pacote keras e usado apenas para carregar o MNIST."
        ) from exc

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], -1).astype(np.float64) / 255.0
    x_test = x_test.reshape(x_test.shape[0], -1).astype(np.float64) / 255.0
    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)

    if validation_size:
        x_val = x_train[-validation_size:]
        y_val = y_train[-validation_size:]
        x_train = x_train[:-validation_size]
        y_train = y_train[:-validation_size]
    else:
        x_val, y_val = None, None

    return x_train, y_train, x_val, y_val, x_test, y_test


def parse_layers(raw):
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def plot_history(history, output_path):
    epochs = np.arange(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, history["loss"], label="treino")
    if history.get("val_loss"):
        axes[0].plot(epochs, history["val_loss"], label="validacao")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoca")
    axes[0].legend()

    axes[1].plot(epochs, history["accuracy"], label="treino")
    if history.get("val_accuracy"):
        axes[1].plot(epochs, history["val_accuracy"], label="validacao")
    axes[1].set_title("Acuracia")
    axes[1].set_xlabel("Epoca")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def confusion_matrix(y_true, y_pred, n_classes=10):
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for real, pred in zip(y_true, y_pred):
        matrix[real, pred] += 1
    return matrix


def plot_confusion_matrix(matrix, output_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    ax.set_title("Matriz de confusao")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Treina um MLP em NumPy no MNIST.")
    parser.add_argument("--layers", default="784,256,128,10", help="Dimensoes separadas por virgula.")
    parser.add_argument("--activation", default="relu", choices=["relu", "tanh"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default="baseline")
    args = parser.parse_args()

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    x_train, y_train, x_val, y_val, x_test, y_test = load_mnist()
    model = MLP(parse_layers(args.layers), activation=args.activation, seed=args.seed)
    history = model.fit(
        x_train,
        y_train,
        x_val=x_val,
        y_val=y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
    )
    test_metrics = model.evaluate(x_test, y_test)
    y_pred = model.predict(x_test)
    conf = confusion_matrix(y_test, y_pred)

    payload = {
        "config": vars(args),
        "history": history,
        "test": test_metrics,
    }
    metrics_path = results_dir / f"{args.run_name}_metrics.json"
    plot_path = results_dir / f"{args.run_name}_curves.png"
    confusion_path = results_dir / f"{args.run_name}_confusion.png"

    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot_history(history, plot_path)
    plot_confusion_matrix(conf, confusion_path)

    print(f"Teste: loss={test_metrics['loss']:.4f} accuracy={test_metrics['accuracy']:.4f}")
    print(f"Metricas salvas em {metrics_path}")
    print(f"Curvas salvas em {plot_path}")
    print(f"Matriz de confusao salva em {confusion_path}")


if __name__ == "__main__":
    main()
