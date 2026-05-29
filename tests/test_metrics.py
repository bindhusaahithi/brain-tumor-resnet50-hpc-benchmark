from __future__ import annotations

import pandas as pd

from brain_tumor_hpsc.metrics import build_classification_metrics, compute_benchmark_summary


def test_compute_benchmark_summary() -> None:
    results = pd.DataFrame(
        [
            {
                "environment": "CPU - Anvil HPC",
                "val_accuracy": 0.6646,
                "training_time_minutes": 35.16,
            },
            {
                "environment": "GPU - Colab T4",
                "val_accuracy": 0.8063,
                "training_time_minutes": 16.32,
            },
        ]
    )

    summary = compute_benchmark_summary(results)

    assert summary["time_reduction_percent"] == 53.58
    assert summary["validation_accuracy_gain_points"] == 14.17


def test_build_classification_metrics() -> None:
    metrics = build_classification_metrics(
        y_true=[0, 1, 1, 0],
        y_pred=[0, 1, 0, 0],
        class_names=["negative", "positive"],
    )

    assert metrics["confusion_matrix"] == [[2, 0], [1, 1]]
    assert "classification_report" in metrics
