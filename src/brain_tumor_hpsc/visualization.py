from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from brain_tumor_hpsc.metrics import compute_benchmark_summary


def save_training_time_chart(results: pd.DataFrame, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#6B7280", "#0F766E"]
    ax.bar(results["environment"], results["training_time_minutes"], color=colors)
    ax.set_title("Training Time: CPU vs GPU")
    ax.set_ylabel("Minutes")
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.25)
    for index, value in enumerate(results["training_time_minutes"]):
        ax.text(index, value + 0.5, f"{value:.2f} min", ha="center")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_accuracy_loss_chart(
    cpu_history: pd.DataFrame,
    gpu_history: pd.DataFrame,
    output_path: str | Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(
        cpu_history["epoch"],
        cpu_history["val_accuracy"],
        label="CPU validation",
        color="#6B7280",
    )
    axes[0].plot(
        gpu_history["epoch"],
        gpu_history["val_accuracy"],
        label="GPU validation",
        color="#0F766E",
    )
    axes[0].set_title("Validation Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(
        cpu_history["epoch"],
        cpu_history["val_loss"],
        label="CPU validation",
        color="#6B7280",
    )
    axes[1].plot(
        gpu_history["epoch"],
        gpu_history["val_loss"],
        label="GPU validation",
        color="#0F766E",
    )
    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_performance_summary(results: pd.DataFrame, output_path: str | Path) -> None:
    summary = compute_benchmark_summary(results)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axis("off")
    lines = [
        "Brain Tumor Detection Benchmark",
        f"GPU training time reduction: {summary['time_reduction_percent']:.2f}%",
        "Validation accuracy gain: "
        f"{summary['validation_accuracy_gain_points']:.2f} percentage points",
        "Architecture: ResNet50 transfer learning",
        "Task: 4-class MRI classification",
    ]
    for index, line in enumerate(lines):
        size = 18 if index == 0 else 13
        weight = "bold" if index == 0 else "normal"
        ax.text(0.05, 0.85 - index * 0.16, line, fontsize=size, weight=weight)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
