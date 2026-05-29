from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_image_for_prediction(path: str | Path, image_size: tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize(image_size)
    array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(array, axis=0)


def predict_image(
    model,
    image_path: str | Path,
    class_names: list[str],
    image_size: tuple[int, int],
) -> dict[str, float | str]:
    probabilities = model.predict(load_image_for_prediction(image_path, image_size), verbose=0)[0]
    best_index = int(np.argmax(probabilities))
    return {
        "class_name": class_names[best_index],
        "confidence": float(probabilities[best_index]),
        "probabilities": {
            name: float(probabilities[index]) for index, name in enumerate(class_names)
        },
    }
