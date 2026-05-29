# Runbook

## Local Development

1. Create a virtual environment.
2. Install dependencies from `requirements.txt`.
3. Place the dataset under `data/raw/`.
4. Generate visualizations with `python scripts/make_visualisations.py`.
5. Launch the demo with `streamlit run app/streamlit_app.py`.

## Training

Use:

```bash
python scripts/train.py --config config/config.yaml --data-dir data/raw/Training --output-dir artifacts
```

Training writes a Keras model to `artifacts/model.keras`.

## Evaluation

Use:

```bash
python scripts/evaluate.py --model artifacts/model.keras --data-dir data/raw/Testing --output-dir reports/evaluation
```

Evaluation writes `metrics.json`.

## Troubleshooting

- If TensorFlow is missing, install dependencies or run on Colab.
- If the demo cannot find a model, train one first or upload a `.keras` file in the app.
- If class labels look wrong, verify folder names match the class names in `config/config.yaml`.

## Publish Checklist

- Do not upload private datasets or patient information.
- Keep trained models out of git unless the dataset license allows sharing.
- Run visualizations before publishing so `images/` is populated.
- Include the resume bullet from `docs/resume_bullets.md`.
