from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brain_tumor_hpsc.visualization import (
    save_accuracy_loss_chart,
    save_performance_summary,
    save_training_time_chart,
)


METRICS_DIR = ROOT / "reports" / "metrics"
IMAGE_DIR = ROOT / "images"


def main() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(METRICS_DIR / "hardware_results.csv")
    cpu_history = pd.read_csv(METRICS_DIR / "training_history_cpu.csv")
    gpu_history = pd.read_csv(METRICS_DIR / "training_history_gpu.csv")

    save_training_time_chart(results, IMAGE_DIR / "training_time_comparison.png")
    save_accuracy_loss_chart(cpu_history, gpu_history, IMAGE_DIR / "accuracy_loss_comparison.png")
    save_performance_summary(results, IMAGE_DIR / "performance_summary.png")
    print(f"Saved visualizations to {IMAGE_DIR}")


if __name__ == "__main__":
    main()
