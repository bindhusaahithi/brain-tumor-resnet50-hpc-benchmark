from __future__ import annotations

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def compute_benchmark_summary(results: pd.DataFrame) -> dict[str, float]:
    """Compute headline benchmark deltas from CPU/GPU result rows."""
    cpu = results.loc[results["environment"].str.contains("CPU", case=False)].iloc[0]
    gpu = results.loc[results["environment"].str.contains("GPU", case=False)].iloc[0]

    time_reduction = (cpu["training_time_minutes"] - gpu["training_time_minutes"]) / cpu[
        "training_time_minutes"
    ]
    val_accuracy_gain = gpu["val_accuracy"] - cpu["val_accuracy"]

    return {
        "time_reduction_percent": round(float(time_reduction * 100), 2),
        "validation_accuracy_gain_points": round(float(val_accuracy_gain * 100), 2),
    }


def build_classification_metrics(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
) -> dict[str, object]:
    """Create standard classification artifacts for model evaluation."""
    return {
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
