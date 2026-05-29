from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brain_tumor_hpsc.config import load_config
from brain_tumor_hpsc.data import assert_image_dataset, build_image_dataset
from brain_tumor_hpsc.model import build_resnet50_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the ResNet50 brain tumor classifier.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    class_names = config["data"]["class_names"]
    image_size = tuple(config["data"]["image_size"])
    batch_size = int(config["data"]["batch_size"])

    assert_image_dataset(args.data_dir, class_names)
    train_ds = build_image_dataset(
        args.data_dir,
        image_size=image_size,
        batch_size=batch_size,
        validation_split=float(config["data"]["validation_split"]),
        subset="training",
        seed=int(config["project"]["seed"]),
    )
    val_ds = build_image_dataset(
        args.data_dir,
        image_size=image_size,
        batch_size=batch_size,
        validation_split=float(config["data"]["validation_split"]),
        subset="validation",
        seed=int(config["project"]["seed"]),
    )

    model = build_resnet50_classifier(
        image_size=image_size,
        num_classes=len(class_names),
        learning_rate=float(config["training"]["learning_rate"]),
        freeze_backbone=bool(config["training"]["freeze_backbone"]),
    )
    model.fit(train_ds, validation_data=val_ds, epochs=int(config["training"]["epochs"]))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / "model.keras")


if __name__ == "__main__":
    main()
