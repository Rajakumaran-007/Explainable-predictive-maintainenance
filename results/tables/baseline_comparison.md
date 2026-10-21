# Baseline Models Performance Comparison (FD001)

| Model Architecture | Trainable Params | RMSE (Cycles) | MAE (Cycles) | R² Score | NASA Asymmetric Score | Inference Latency (ms) | Serialized Size (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ridge Regression** | 71 | 18.03 | 14.71 | 0.798 | 501.8 | 0.03 | 0.00 |
| **Random Forest** | 681,316 | 18.57 | 13.36 | 0.785 | 1111.7 | 13.17 | 46.81 |
| **MLP Baseline** | 100,545 | 13.91 | 10.39 | 0.880 | 310.8 | 0.40 | 1.17 |
| **LSTM Baseline** | 55,873 | 13.55 | 9.89 | 0.886 | 289.1 | 1.04 | 0.65 |
| **1D-CNN Baseline** | 44,001 | 19.14 | 14.61 | 0.772 | 821.6 | 0.49 | 0.53 |
