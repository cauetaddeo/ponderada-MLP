import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# Configuracoes comparadas na entrega.
EXPERIMENTS = [
    {
        "run_name": "baseline_relu",
        "layers": "784-256-128-10",
        "activation": "relu",
        "learning_rate": "0.05",
        "momentum": "0.9",
        "epochs": "15",
    },
    {
        "run_name": "deeper_relu",
        "layers": "784-256-128-64-10",
        "activation": "relu",
        "learning_rate": "0.03",
        "momentum": "0.9",
        "epochs": "15",
    },
]


def main():
    # --epochs e --batch-size ajudam a testar rapidamente sem editar a lista.
    parser = argparse.ArgumentParser(description="Roda as configuracoes comparativas do MLP no MNIST.")
    parser.add_argument("--epochs", type=int, default=None, help="Sobrescreve o numero de epocas dos experimentos.")
    parser.add_argument("--batch-size", type=int, default=None, help="Sobrescreve o tamanho do mini-batch.")
    args = parser.parse_args()

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    rows = []
    for base_exp in EXPERIMENTS:
        # Copia para permitir sobrescrever parametros sem alterar o original.
        exp = base_exp.copy()
        if args.epochs is not None:
            exp["epochs"] = str(args.epochs)

        command = [
            # Usa -m para resolver imports a partir da raiz do projeto.
            sys.executable,
            "-m",
            "scripts.train_mnist",
            "--run-name",
            exp["run_name"],
            "--layers",
            exp["layers"],
            "--activation",
            exp["activation"],
            "--learning-rate",
            exp["learning_rate"],
            "--momentum",
            exp["momentum"],
            "--epochs",
            exp["epochs"],
        ]
        if args.batch_size is not None:
            command.extend(["--batch-size", str(args.batch_size)])

        # Cada experimento gera seu proprio JSON de metricas.
        subprocess.run(command, check=True, cwd=PROJECT_ROOT)
        metrics_path = results_dir / f"{exp['run_name']}_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        diagnostics = metrics.get("diagnostics", {})
        rows.append(
            {
                **exp,
                "final_train_accuracy": f"{diagnostics.get('final_train_accuracy', 0.0):.4f}",
                "final_val_accuracy": f"{diagnostics.get('final_val_accuracy', 0.0):.4f}",
                "best_val_accuracy": f"{diagnostics.get('best_val_accuracy', 0.0):.4f}",
                "best_val_epoch": diagnostics.get("best_val_epoch", ""),
                "train_val_accuracy_gap": f"{diagnostics.get('train_val_accuracy_gap', 0.0):.4f}",
                "train_test_accuracy_gap": f"{diagnostics.get('train_test_accuracy_gap', 0.0):.4f}",
                "test_loss": f"{metrics['test']['loss']:.4f}",
                "test_accuracy": f"{metrics['test']['accuracy']:.4f}",
                "precision_macro": f"{metrics['test']['precision_macro']:.4f}",
                "recall_macro": f"{metrics['test']['recall_macro']:.4f}",
                "f1_macro": f"{metrics['test']['f1_macro']:.4f}",
                "precision_weighted": f"{metrics['test']['precision_weighted']:.4f}",
                "recall_weighted": f"{metrics['test']['recall_weighted']:.4f}",
                "f1_weighted": f"{metrics['test']['f1_weighted']:.4f}",
                "balanced_accuracy": f"{metrics['test']['balanced_accuracy']:.4f}",
            }
        )

    table_path = results_dir / "experiments.csv"
    # Consolida as configuracoes e resultados em uma tabela comparativa.
    with table_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Tabela de configuracoes salva em {table_path}")


if __name__ == "__main__":
    main()
