from __future__ import annotations

import tempfile
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brain_tumor_hpsc.config import load_config
from brain_tumor_hpsc.predict import predict_image

CONFIG = load_config(ROOT / "config" / "config.yaml")
CLASS_NAMES = CONFIG["data"]["class_names"]
IMAGE_SIZE = tuple(CONFIG["data"]["image_size"])


st.set_page_config(page_title="Brain Tumor MRI Classifier", layout="wide")
st.title("Brain Tumor MRI Classifier")

left, right = st.columns([0.9, 1.1])

with left:
    model_file = st.file_uploader("Upload trained Keras model", type=["keras", "h5"])
    image_file = st.file_uploader("Upload MRI image", type=["jpg", "jpeg", "png"])
    st.caption("Portfolio demo only. Not for clinical diagnosis.")

with right:
    if image_file:
        st.image(image_file, caption="Uploaded MRI", use_container_width=True)

    if model_file and image_file:
        try:
            import tensorflow as tf

            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = Path(tmpdir) / model_file.name
                image_path = Path(tmpdir) / image_file.name
                model_path.write_bytes(model_file.getbuffer())
                image_path.write_bytes(image_file.getbuffer())

                model = tf.keras.models.load_model(model_path)
                prediction = predict_image(model, image_path, CLASS_NAMES, IMAGE_SIZE)

            st.metric("Predicted class", prediction["class_name"])
            st.metric("Confidence", f"{prediction['confidence']:.2%}")
            probabilities = pd.DataFrame(
                prediction["probabilities"].items(),
                columns=["Class", "Probability"],
            )
            st.bar_chart(probabilities.set_index("Class"))
        except ModuleNotFoundError:
            st.warning(
                "The hosted demo is running in lightweight mode. "
                "Install requirements-ml.txt locally to enable Keras model inference."
            )
        except Exception as exc:
            st.error(f"Unable to run inference: {exc}")
    else:
        st.info("Upload a trained model and an MRI image to run inference.")
