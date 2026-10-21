# Model Checkpoints & Architecture Artifacts

This directory stores trained deep learning and baseline machine learning checkpoints, training histories, and architecture summaries for the Remaining Useful Life (RUL) prediction framework.

## Directory Structure
- `checkpoints/`: PyTorch model weights (`.pt`) containing state dictionaries, optimizer states, epoch counters, and validation scores. Checkpoint files are excluded from Git version control via `.gitignore`.
- `baselines/`: Serialized classical machine learning baseline models (Random Forest, Ridge Regression).
- `summaries/`: Parameter counts, serialized model size, and inference latency benchmarks.

## Supported Architectures
1. **Hybrid 1D-CNN + LSTM**: Proposed architecture combining local multi-scale temporal convolutions with recurrent sequence modeling.
2. **LSTM Baseline**: Multi-layer LSTM network for pure recurrent temporal modeling.
3. **1D-CNN Baseline**: Pure temporal convolutional network with residual skip connections.
4. **MLP Baseline**: Multi-layer perceptron benchmark flattening sensor windows.
5. **Classical ML Baselines**: Ridge Regression and Random Forest Regressors.
