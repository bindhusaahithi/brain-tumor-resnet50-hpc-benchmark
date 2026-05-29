# Project Report

## Problem

Brain tumor diagnosis often depends on MRI interpretation by trained medical experts. This project explores whether a ResNet50-based transfer learning model can classify MRI scans into four tumor categories while also quantifying the practical benefit of GPU acceleration.

## Objective

Build a reproducible brain MRI classifier and compare CPU versus GPU training performance using accuracy, loss, and training time.

## Method

- Resize MRI images to 224 x 224 pixels.
- Use ImageNet-pretrained ResNet50 as a frozen feature extractor.
- Add a task-specific classification head with global average pooling, a dense ReLU layer, and a four-class softmax layer.
- Train with Adam and sparse categorical crossentropy for 50 epochs.
- Compare CPU training on Anvil HPC against GPU training on Google Colab T4.

## Key Findings

- GPU training completed in 16.32 minutes compared with 35.16 minutes on CPU.
- GPU training improved validation accuracy from 66.46% to 80.63%.
- GPU validation loss was lower, suggesting better generalization in the recorded run.

## Applied Science Value

The project demonstrates practical ML experimentation: controlled benchmark setup, reproducible model configuration, clear metrics, and a live inference interface. For an Amazon Applied Science role, the strongest signal is not just the model choice, but the end-to-end thinking around measurement, scalability, responsible use, and production readiness.
