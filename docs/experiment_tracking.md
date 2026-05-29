# Experiment Tracking

Use this file to record controlled runs. Each row should represent one training run.

| Run ID | Date | Environment | Dataset Version | Model | Epochs | Batch Size | Val Accuracy | Val Loss | Training Time | Notes |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| cpu-anvil-resnet50-v1 | 2026-05-28 | Anvil HPC, 16 CPU cores | local-v1 | ResNet50 frozen backbone | 50 | 32 | 0.6646 | 0.8273 | 35.16 min | Baseline CPU run |
| gpu-colab-t4-resnet50-v1 | 2026-05-28 | Google Colab, NVIDIA T4 | local-v1 | ResNet50 frozen backbone | 50 | 32 | 0.8063 | 0.4969 | 16.32 min | GPU benchmark |

## Minimum Metadata for New Runs

- Dataset source and version
- Train/validation/test split method
- Hardware environment
- Random seed
- Model architecture
- Hyperparameters
- Final metrics
- Error analysis notes
