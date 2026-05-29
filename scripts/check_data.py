from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brain_tumor_hpsc.config import load_config
from brain_tumor_hpsc.data import summarize_image_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and summarize the local MRI dataset.")
    parser.add_argument("--data-dir", default="data/raw/Training")
    parser.add_argument("--config", default="config/config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    summary = summarize_image_dataset(args.data_dir, config["data"]["class_names"])

    print("Dataset summary")
    for class_name, count in summary.items():
        print(f"{class_name}: {count}")


if __name__ == "__main__":
    main()
