# Brain Tumor Detection with ResNet50: CPU vs GPU Benchmark

Industry-style applied science project comparing CPU and GPU training performance for a ResNet50 brain MRI classifier.

The model classifies MRI images into four categories:

- Glioma tumor
- Meningioma tumor
- Pituitary tumor
- No tumor

The project is designed for an Amazon Applied Science internship portfolio: it shows problem framing, reproducible ML workflow, clean engineering structure, visual reporting, and a live demo surface.

## Highlights

- Transfer learning with ImageNet-pretrained ResNet50
- CPU vs GPU benchmark using the same model configuration
- Streamlit demo for local prediction and result explanation
- Reproducible training/evaluation scripts
- Visualization pipeline for benchmark charts
- Clean data and model artifact structure
- Model card, project report, and runbook documentation
- CI-ready quality checks with linting and tests
- Responsible AI notes for medical ML limitations

## Results Summary

| Environment | Train Accuracy | Validation Accuracy | Train Loss | Validation Loss | Training Time |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU - Anvil HPC | 84.20% | 66.46% | 0.3966 | 0.8273 | 35.16 min |
| GPU - Colab T4 | 84.00% | 80.63% | 0.4006 | 0.4969 | 16.32 min |

The GPU run reduced training time by about 53.6% and improved validation accuracy by 14.17 percentage points.

## Project Structure

```text
HPSC/
  app/                         Streamlit live demo
  config/                      Reproducible experiment settings
  data/                        Local dataset folders, ignored by git
  docs/                        Project report, runbook, model card
  images/                      Generated visualizations and screenshots
  notebooks/                   Exploration notebooks
  reports/metrics/             Benchmark and training history CSV files
  scripts/                     Command-line training/evaluation utilities
  src/brain_tumor_hpsc/        Production-style Python package
  tests/                       Lightweight quality checks
  .github/workflows/           Continuous integration checks
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

For local development checks, install the development dependencies:

```bash
pip install -r requirements-dev.txt
```

For GPU training, install the TensorFlow build compatible with your CUDA environment, or run the training notebook/script on Google Colab.

## Dataset Layout

Place the MRI dataset locally using this format:

```text
data/raw/
  Training/
    glioma/
    meningioma/
    notumor/
    pituitary/
  Testing/
    glioma/
    meningioma/
    notumor/
    pituitary/
```

The `data/` folder contains placeholders only. Medical images and trained model files are intentionally excluded from git.

## Train

```bash
python scripts/train.py --config config/config.yaml --data-dir data/raw/Training --output-dir artifacts
```

## Evaluate

```bash
python scripts/evaluate.py --model artifacts/model.keras --data-dir data/raw/Testing --output-dir reports/evaluation
```

Evaluation writes metrics and a confusion matrix to `reports/evaluation/`.

## Generate Visualizations

```bash
python scripts/make_visualisations.py
```

This creates:

- `images/training_time_comparison.png`
- `images/accuracy_loss_comparison.png`
- `images/performance_summary.png`

## Live Demo

```bash
streamlit run app/streamlit_app.py
```

Upload an MRI image, select a trained model file, and view the predicted tumor class with confidence scores.

## Resume Bullet

Built an end-to-end ResNet50 brain MRI classifier with CPU vs GPU benchmarking, reproducible TensorFlow training/evaluation pipelines, Streamlit inference demo, and responsible AI documentation; GPU acceleration reduced training time by 53.6% and improved validation accuracy from 66.46% to 80.63%.

## Quality Checks

```bash
ruff check .
pytest
```

## Industry Readiness Checklist

- Reproducible configuration is stored in `config/config.yaml`.
- Raw data and trained models are excluded from git.
- Evaluation produces machine-readable artifacts.
- Project includes a model card, runbook, and responsible-use warnings.
- Source code is separated from scripts and app code.
- CI configuration is included for linting and unit tests.

## Responsible Use

This project is for academic and portfolio demonstration. It is not a clinical diagnostic tool. Any real-world medical deployment would require clinical validation, bias analysis, calibration, monitoring, privacy controls, and regulatory review.
