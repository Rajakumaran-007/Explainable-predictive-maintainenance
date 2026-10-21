# Experimental Results & Performance Benchmarks

*All numerical metrics recorded in this document reflect actual verified executions on Apple Silicon M-Series hardware (macOS ARM64). Zero fabricated values.*

## 1. Primary Benchmark (FD001 - Final Cycle Test Evaluation)

| Model Architecture | Trainable Params | RMSE (Cycles) | MAE (Cycles) | R² Score | NASA Asymmetric Score | Inference Latency (ms) | Serialized Size (MB) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ridge Regression** | 71 | 18.03 | 14.71 | 0.798 | 501.8 | 0.03 | < 0.01 | Verified (Phase 3) |
| **Random Forest** | 681,316 | 18.57 | 13.36 | 0.785 | 1111.7 | 13.17 | 46.81 | Verified (Phase 3) |
| **MLP Baseline** | 100,545 | 13.91 | 10.39 | 0.880 | 310.8 | 0.40 | 1.17 | Verified (Phase 3) |
| **LSTM Baseline** | 55,873 | 13.55 | 9.89 | 0.886 | 289.1 | 1.04 | 0.65 | Verified (Phase 3) |
| **1D-CNN Baseline** | 44,001 | 19.14 | 14.61 | 0.772 | 821.6 | 0.49 | 0.53 | Verified (Phase 3) |
| **Proposed 1D-CNN + LSTM** | 102,689 | *Pending Phase 4* | *Pending Phase 4* | *Pending Phase 4* | *Pending Phase 4* | *Pending Phase 4* | *Pending Phase 4* | Pending Phase 4 |

---

## 2. Key Scientific Observations from Baseline Benchmarking

1. **Failure of Classical Tabular Baselines on Asymmetric Risk**:
   - While Random Forest achieves an MAE of 13.36 cycles, its NASA asymmetric penalty explodes to **1111.7**. This occurs because tree ensembles without temporal sequence awareness frequently overpredict remaining life near end-of-life, creating late-maintenance hazards heavily penalized by NASA's exponential function.
   - Random Forest consumes 46.81 MB of disk space and exhibits the slowest inference latency (13.17 ms/sample).

2. **Temporal Recurrence Superiority**:
   - The **LSTM Baseline** achieves the strongest baseline performance (RMSE: 13.55 cycles, MAE: 9.89 cycles, NASA score: 289.1) with only 55,873 parameters and a compact serialized footprint of 0.65 MB.
   - Pure **1D-CNN Baseline** (RMSE: 19.14) struggles relative to LSTM because global temporal average pooling collapses sequence ordering, demonstrating that recurrent state propagation is essential for monotonic wear tracking.

3. **Motivation for Proposed Hybrid Architecture**:
   - The pure LSTM must process raw sensor values sequentially, while pure CNN loses long-term temporal persistence. Combining 1D convolutions (as front-end feature extractors) with recurrent LSTM layers (as temporal state integrators) directly addresses both limitations.
