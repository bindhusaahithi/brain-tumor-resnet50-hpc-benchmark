# Security and Data Handling

This project should not store private medical data, credentials, or trained model artifacts in git.

## Data Rules

- Keep MRI images under `data/raw/` locally only.
- Do not commit patient identifiers or metadata.
- Use de-identified public datasets for portfolio demos.
- Confirm dataset license terms before publishing.

## Model Artifact Rules

- Keep trained models under `artifacts/`.
- Do not publish a model as clinically useful without validation.
- Document the dataset and run ID used to create any shared model artifact.
