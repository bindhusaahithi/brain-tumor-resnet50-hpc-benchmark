from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import tensorflow as tf

from brain_tumor_hpsc.config import load_config
from brain_tumor_hpsc.data import assert_image_dataset, build_image_dataset
from brain_tumor_hpsc.metrics import build_classification_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained brain tumor classifier.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--output-dir", default="reports/evaluation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    class_names = config["data"]["class_names"]
    image_size = tuple(config["data"]["image_size"])
    batch_size = int(config["data"]["batch_size"])

    assert_image_dataset(args.data_dir, class_names)
    test_ds = build_image_dataset(
        args.data_dir,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
    )
    model = tf.keras.models.load_model(args.model)
    loss, accuracy = model.evaluate(test_ds, verbose=0)
    probabilities = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(probabilities, axis=1).tolist()
    y_true = np.concatenate([labels.numpy() for _, labels in test_ds], axis=0).tolist()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "loss": float(loss),
        "accuracy": float(accuracy),
        **build_classification_metrics(y_true, y_pred, class_names),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(metrics)


if __name__ == "__main__":
    main()
