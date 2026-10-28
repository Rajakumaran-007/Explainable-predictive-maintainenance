# Cross-Condition Generalization & Domain Shift Study

| Evaluation Regime | Flight Conditions | Test Engines | RMSE (Cycles) | MAE (Cycles) | R² Score | NASA Asymmetric Score | Mean Bias (Cycles) | Evaluation Protocol |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **FD001 In-Domain (Sea Level Benchmark)** | 1 | 100 | 14.39 | 10.40 | 0.870 | 380.5 | -1.20 | In-Domain Train & Test |
| **Zero-Shot Transfer (FD001 Model -> FD002)** | 6 | 259 | 53.19 | 42.40 | -0.534 | 710290.5 | +30.98 | Zero-Shot Cross-Subset Transfer |
| **FD002 Condition-Aware (Multi-Regime Model)** | 6 | 259 | 18.13 | 13.81 | 0.822 | 1323.5 | -6.75 | Condition-Aware In-Domain |
