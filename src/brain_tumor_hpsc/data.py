from __future__ import annotations

from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def assert_image_dataset(path: str | Path, class_names: list[str]) -> None:
    """Validate that the dataset folder has one subfolder per class."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_path}")

    missing = [name for name in class_names if not (dataset_path / name).exists()]
    if missing:
        raise ValueError(f"Missing class folders under {dataset_path}: {missing}")


def summarize_image_dataset(path: str | Path, class_names: list[str]) -> dict[str, int]:
    """Count image files per expected class folder."""
    assert_image_dataset(path, class_names)
    dataset_path = Path(path)
    return {
        class_name: sum(
            1
            for file_path in (dataset_path / class_name).rglob("*")
            if file_path.suffix.lower() in IMAGE_EXTENSIONS
        )
        for class_name in class_names
    }


def build_image_dataset(
    path: str | Path,
    image_size: tuple[int, int],
    batch_size: int,
    validation_split: float | None = None,
    subset: str | None = None,
    seed: int = 42,
    shuffle: bool = True,
):
    """Create a TensorFlow image dataset from a directory tree."""
    import tensorflow as tf

    kwargs = {
        "directory": str(path),
        "image_size": image_size,
        "batch_size": batch_size,
        "seed": seed,
        "shuffle": shuffle,
    }
    if validation_split is not None and subset is not None:
        kwargs["validation_split"] = validation_split
        kwargs["subset"] = subset

    dataset = tf.keras.utils.image_dataset_from_directory(**kwargs)
    return dataset.prefetch(tf.data.AUTOTUNE)
