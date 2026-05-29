# Model Card

## Model

ResNet50 transfer learning classifier for brain MRI tumor classification.

## Intended Use

Academic demonstration, applied science portfolio, and CPU/GPU performance benchmarking.

## Not Intended For

Clinical diagnosis, emergency triage, or automated medical decision-making.

## Classes

- Glioma
- Meningioma
- No tumor
- Pituitary

## Metrics

| Environment | Validation Accuracy | Validation Loss |
| --- | ---: | ---: |
| CPU - Anvil HPC | 66.46% | 0.8273 |
| GPU - Colab T4 | 80.63% | 0.4969 |

## Limitations

- Dataset provenance and demographic representativeness must be reviewed before any serious use.
- Validation accuracy alone is insufficient for medical deployment.
- Requires confusion matrix, sensitivity, specificity, calibration, and external validation.
- Predictions may be sensitive to MRI acquisition differences, preprocessing, and image quality.

## Responsible Use Checklist

- Validate on independent hospitals or scanners.
- Track per-class recall, especially for tumor-positive classes.
- Add human-in-the-loop review.
- Monitor drift after deployment.
- Keep data governance, privacy, and consent requirements explicit.
