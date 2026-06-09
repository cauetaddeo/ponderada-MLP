import argparse
import csv
import gzip
import json
import sys
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mlp import MLP


MNIST_URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}


def _read_idx_images(path):
    with gzip.open(path, "rb") as file:
        data = np.frombuffer(file.read(), dtype=np.uint8)
    magic, n_images, rows, cols = data[:16].view(">i4")
    if magic != 2051:
        raise ValueError(f"Arquivo de imagens MNIST invalido: {path}")
    return data[16:].reshape(n_images, rows, cols)


def _read_idx_labels(path):
    with gzip.open(path, "rb") as file:
        data = np.frombuffer(file.read(), dtype=np.uint8)
    magic, n_labels = data[:8].view(">i4")
    if magic != 2049:
        raise ValueError(f"Arquivo de labels MNIST invalido: {path}")
    return data[8:].reshape(n_labels)


def _download_mnist_file(url, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return
    print(f"Baixando {url}...")
    urllib.request.urlretrieve(url, output_path)


def load_mnist_from_idx():
    data_dir = PROJECT_ROOT / "data" / "mnist"
    paths = {name: data_dir / url.rsplit("/", 1)[-1] for name, url in MNIST_URLS.items()}
    for name, url in MNIST_URLS.items():
        _download_mnist_file(url, paths[name])

    x_train = _read_idx_images(paths["train_images"])
    y_train = _read_idx_labels(paths["train_labels"])
    x_test = _read_idx_images(paths["test_images"])
    y_test = _read_idx_labels(paths["test_labels"])
    return (x_train, y_train), (x_test, y_test)


def load_mnist(validation_size=10000):
    try:
        from keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    except (ImportError, ModuleNotFoundError):
        print("Keras/TensorFlow nao encontrado. Usando fallback IDX oficial do MNIST.")
        (x_train, y_train), (x_test, y_test) = load_mnist_from_idx()

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
    normalized = raw.replace("-", ",")
    return [int(item.strip()) for item in normalized.split(",") if item.strip()]


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


def classification_metrics(matrix):
    true_positive = np.diag(matrix).astype(np.float64)
    predicted = matrix.sum(axis=0).astype(np.float64)
    support = matrix.sum(axis=1).astype(np.float64)
    total = matrix.sum()

    precision = np.divide(true_positive, predicted, out=np.zeros_like(true_positive), where=predicted != 0)
    recall = np.divide(true_positive, support, out=np.zeros_like(true_positive), where=support != 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )

    per_class = []
    for label in range(matrix.shape[0]):
        per_class.append(
            {
                "class": int(label),
                "precision": float(precision[label]),
                "recall": float(recall[label]),
                "f1": float(f1[label]),
                "support": int(support[label]),
            }
        )

    weights = support / total
    return {
        "accuracy": float(true_positive.sum() / total),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "weighted_precision": float(np.sum(precision * weights)),
        "weighted_recall": float(np.sum(recall * weights)),
        "weighted_f1": float(np.sum(f1 * weights)),
        "balanced_accuracy": float(np.mean(recall)),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def write_classification_report(metrics, output_path):
    fieldnames = ["class", "precision", "recall", "f1", "support"]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics["per_class"])


def training_diagnostics(history, test_metrics):
    final_train_accuracy = history["accuracy"][-1]
    final_train_loss = history["loss"][-1]
    diagnostics = {
        "final_train_accuracy": final_train_accuracy,
        "final_train_loss": final_train_loss,
        "test_accuracy": test_metrics["accuracy"],
        "test_loss": test_metrics["loss"],
    }

    if history.get("val_accuracy"):
        val_accuracies = history["val_accuracy"]
        best_epoch = int(np.argmax(val_accuracies)) + 1
        final_val_accuracy = val_accuracies[-1]
        diagnostics.update(
            {
                "final_val_accuracy": final_val_accuracy,
                "final_val_loss": history["val_loss"][-1],
                "best_val_epoch": best_epoch,
                "best_val_accuracy": val_accuracies[best_epoch - 1],
                "best_val_loss": history["val_loss"][best_epoch - 1],
                "train_val_accuracy_gap": final_train_accuracy - final_val_accuracy,
                "train_test_accuracy_gap": final_train_accuracy - test_metrics["accuracy"],
            }
        )

    return diagnostics


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
    parser.add_argument("--layers", default="784-256-128-10", help="Dimensoes separadas por hifen ou virgula.")
    parser.add_argument("--activation", default="relu", choices=["relu", "tanh"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default="baseline")
    args = parser.parse_args()

    results_dir = PROJECT_ROOT / "results"
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
    class_metrics = classification_metrics(conf)
    diagnostics = training_diagnostics(history, test_metrics)
    test_metrics.update(
        {
            "precision_macro": class_metrics["macro_precision"],
            "recall_macro": class_metrics["macro_recall"],
            "f1_macro": class_metrics["macro_f1"],
            "precision_weighted": class_metrics["weighted_precision"],
            "recall_weighted": class_metrics["weighted_recall"],
            "f1_weighted": class_metrics["weighted_f1"],
            "balanced_accuracy": class_metrics["balanced_accuracy"],
        }
    )

    payload = {
        "config": vars(args),
        "history": history,
        "test": test_metrics,
        "classification": class_metrics,
        "diagnostics": diagnostics,
    }
    metrics_path = results_dir / f"{args.run_name}_metrics.json"
    report_path = results_dir / f"{args.run_name}_classification_report.csv"
    plot_path = results_dir / f"{args.run_name}_curves.png"
    confusion_path = results_dir / f"{args.run_name}_confusion.png"

    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_classification_report(class_metrics, report_path)
    plot_history(history, plot_path)
    plot_confusion_matrix(conf, confusion_path)

    print(
        "Teste: "
        f"loss={test_metrics['loss']:.4f} "
        f"accuracy={test_metrics['accuracy']:.4f} "
        f"recall_macro={test_metrics['recall_macro']:.4f} "
        f"f1_macro={test_metrics['f1_macro']:.4f}"
    )
    if "train_val_accuracy_gap" in diagnostics:
        print(
            "Diagnostico: "
            f"train_acc={diagnostics['final_train_accuracy']:.4f} "
            f"val_acc={diagnostics['final_val_accuracy']:.4f} "
            f"gap_train_val={diagnostics['train_val_accuracy_gap']:.4f} "
            f"best_val_epoch={diagnostics['best_val_epoch']}"
        )
    print(f"Metricas salvas em {metrics_path}")
    print(f"Relatorio por classe salvo em {report_path}")
    print(f"Curvas salvas em {plot_path}")
    print(f"Matriz de confusao salva em {confusion_path}")


if __name__ == "__main__":
    main()
