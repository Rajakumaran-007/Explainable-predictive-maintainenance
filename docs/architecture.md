# Deep Learning Architecture Specifications

## 1. Hybrid 1D-CNN + LSTM Network Architecture

```text
Input Window: S ∈ R^(W × D)  [Batch, Channels=D, Length=W]
        │
        ▼
[Conv1D Layer 1]       -> Filters: 32, Kernel: 5, Stride: 1, Padding: Same
        │
[BatchNorm1D + ReLU]   -> Normalization & Non-linear Activation
        │
[Conv1D Layer 2]       -> Filters: 64, Kernel: 3, Stride: 1, Padding: Same
        │
[BatchNorm1D + ReLU]   -> Multi-scale Feature Representation
        │
        ▼  Transpose to (Batch, Sequence=W, Features=64)
[LSTM Layer 1]         -> Hidden Dim: 64, Return Sequences: True
        │
[Dropout (p=0.2)]      -> Regularization & Epistemic Uncertainty Sampling
        │
[LSTM Layer 2]         -> Hidden Dim: 64, Return Sequences: False (Last Hidden State)
        │
[Dense Layer 1]        -> 64 Units, ReLU
        │
[Dense Layer 2]        -> 32 Units, ReLU
        │
[Linear Output Layer]  -> 1 Unit: Scalar RUL Prediction (Cycles)
```

## 2. Baseline Architecture Specifications

### 2.1 LSTM Baseline
- Input: `(Batch, W, D)`
- Layer 1: LSTM (hidden_dim=64, return_sequences=True)
- Dropout: `p = 0.2`
- Layer 2: LSTM (hidden_dim=64, return_sequences=False)
- Dense: Linear(64, 32) -> ReLU -> Linear(32, 1)

### 2.2 1D-CNN Baseline
- Input: `(Batch, D, W)`
- Conv1D(32, kernel=5) -> BatchNorm -> ReLU -> MaxPool1D(2)
- Conv1D(64, kernel=3) -> BatchNorm -> ReLU -> MaxPool1D(2)
- Conv1D(128, kernel=3) -> BatchNorm -> ReLU -> AdaptiveAvgPool1D(1)
- Dense: Linear(128, 32) -> ReLU -> Linear(32, 1)

### 2.3 Classical Baselines
- Feature Vector: Flattened window summary statistics (Mean, Std, Min, Max, Last value per sensor)
- Models: Ridge Regression ($\alpha = 1.0$), Random Forest Regressor ($N_{trees}=100$).
