# Model Compression & Edge Deployment Benchmarks (MacBook Air M5)

| Architecture Variant | Parameters | Parameter Reduction (%) | RMSE (Cycles) | MAE (Cycles) | NASA Score | Latency (ms) | Speedup Factor | Serialized Size (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full Hybrid (Baseline)** | 81,505 | 20.6% | 15.96 | 11.05 | 511.3 | 1.24 ms | 1.00x | 0.96 MB |
| **Compact Edge Hybrid** | 12,337 | 88.0% | 15.32 | 11.36 | 369.4 | 0.89 ms | 1.39x | 0.16 MB |
| **Micro Edge Hybrid** | 3,185 | 96.9% | 15.44 | 11.09 | 386.3 | 0.69 ms | 1.80x | 0.05 MB |
