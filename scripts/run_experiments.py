import csv
import json
import subprocess
import sys
from pathlib import Path


EXPERIMENTS = [
    {
        "run_name": "baseline_relu",
        "layers": "784,256,128,10",
        "activation": "relu",
        "learning_rate": "0.05",
        "momentum": "0.9",
        "epochs": "15",
    },
    {
        "run_name": "deeper_relu",
        "layers": "784,256,128,64,10",
        "activation": "relu",
        "learning_rate": "0.03",
        "momentum": "0.9",
        "epochs": "15",
    },
]


def main():
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    rows = []
    for exp in EXPERIMENTS:
        command = [
            sys.executable,
            "scripts/train_mnist.py",
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
        subprocess.run(command, check=True)
        metrics_path = results_dir / f"{exp['run_name']}_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(
            {
                **exp,
                "test_loss": f"{metrics['test']['loss']:.4f}",
                "test_accuracy": f"{metrics['test']['accuracy']:.4f}",
            }
        )

    table_path = results_dir / "experiments.csv"
    with table_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Tabela de configuracoes salva em {table_path}")


if __name__ == "__main__":
    main()
