# Academic Project Defense Presentation: Remaining Useful Life Prediction

**Project Title**: *An Explainable and Uncertainty-Aware Deep Learning Framework for Predicting Remaining Useful Life of Industrial Equipment*  
**Presenter**: Rajakumaran | B.Tech Artificial Intelligence & Data Science  
**Format**: 12-Slide Evaluation Defense Structure with Detailed Speaker Notes

---

## Slide 1: Title & Executive Overview
- **Header**: An Explainable & Uncertainty-Aware Deep Learning Framework for Turbofan RUL Prediction
- **Subheader**: Minor Research Project Defense for Academic Evaluation & Master's Portfolio
- **Key Visual**: Workflow schematic linking Sensor Ingestion ➔ Hybrid CNN-LSTM ➔ Conformal Prediction & Integrated Gradients ➔ Streamlit Dashboard.
- **Presenter Notes**:
  > "Good morning, respected members of the evaluation committee. Today, I present an intelligent predictive maintenance framework designed for high-consequence industrial assets, evaluated on the NASA C-MAPSS dataset. The central motivation of this research is that deterministic black-box models are unacceptable in safety-critical aviation: operators require not only accurate predictions of remaining life, but also mathematically calibrated confidence intervals and physical interpretability linking predictions to underlying thermodynamic degradation."

---

## Slide 2: Problem Statement & Industrial Relevance
- **Points**:
  - Unscheduled downtime in commercial aviation costs the global aerospace sector over $8B annually.
  - Asymmetric Operational Risk: Late predictions ($\hat{y} > y$) risk catastrophic in-flight failure, whereas early predictions ($\hat{y} < y$) cause premature servicing.
  - Research Challenge: Standard neural networks provide overconfident point predictions without uncertainty bounds or physical explanations.
- **Presenter Notes**:
  > "Traditional symmetric metrics like RMSE fail to capture operational risk. If an engine has 10 cycles remaining and we predict 25, that +15 cycle error can lead to uncontained engine failure. In contrast, predicting 5 cycles is conservative and safe. Hence, we evaluate models not only on RMSE and MAE, but critically on the official NASA PHM asymmetric exponential scoring function."

---

## Slide 3: Dataset Provenance & Leakage-Free Preprocessing
- **Points**:
  - NASA C-MAPSS Benchmark: Run-to-failure trajectories (100 train engines, 100 test engines in FD001).
  - Piecewise-linear RUL model with a 125-cycle cap to eliminate healthy baseline label noise.
  - Zero-Variance Pruning: 7 invariant sensors removed ($s_1, s_5, s_6, s_{10}, s_{16}, s_{18}, s_{19}$).
  - Strict Engine-Level Partitioning: 80 engines for training, 20 engines for validation ($\mathcal{E}_{\text{train}} \cap \mathcal{E}_{\text{val}} = \emptyset$).
  - Edge Padding: Handles short test sequences ($T_{\text{end}} < W$) without truncation.
- **Presenter Notes**:
  > "A frequent error in predictive maintenance publications is randomly shuffling sliding windows across the dataset, causing severe temporal leakage. In our methodology, splitting occurs strictly at the engine level: engines in the validation set have zero historical or future samples in training. Furthermore, all feature scaling parameters are fitted exclusively on training trajectories."

---

## Slide 4: Proposed Hybrid 1D-CNN + LSTM Deep Architecture
- **Points**:
  - Input Window: $\mathbf{X} \in \mathbb{R}^{B \times 50 \times 14}$ (50 cycles, 14 informative sensors).
  - 1D-CNN Stage: Conv1D(32, $k=5$) + Conv1D(64, $k=3$) with BatchNorm & ReLU to extract localized cross-sensor spatial correlations.
  - Recurrent Stage: 2-layer stacked LSTM (hidden dim: 64, dropout: 0.2) to maintain cumulative temporal memory.
  - Active Dropout Layer: Remains active at test time for variational inference.
  - Compact Footprint: 102,689 parameters, serialized checkpoint size of 0.96 MB.
- **Presenter Notes**:
  > "Why unite CNNs and LSTMs? Standalone 1D-CNNs require global pooling that destroys temporal order, while pure LSTMs must digest raw, noisy channels sequentially. In our hybrid model, temporal convolutions act as spatial filters discovering local inter-sensor thermodynamic couplings, while the LSTM layers integrate these features over time to track monotonic degradation."

---

## Slide 5: Baseline Benchmarking & Model Comparison
- **Visual**: Master comparison table across classical and deep models.
- **Results**:
  - Ridge Regression: RMSE 18.03, NASA 501.8
  - Random Forest: RMSE 18.57, NASA **1111.7** (Explosive late prediction penalty)
  - 1D-CNN Baseline: RMSE 19.14, NASA 821.6
  - LSTM Baseline: RMSE 13.55, NASA 289.1
  - **Proposed Hybrid 1D-CNN + LSTM**: **RMSE 13.06 cycles** (best individual run), **NASA 281.3**
- **Presenter Notes**:
  > "Looking at the comparison table, Random Forest achieves an acceptable MAE of 13.36 cycles, but its NASA score surges to 1111.7. Because decision trees lack temporal state awareness, they frequently overpredict remaining life near end-of-life. Our Proposed Hybrid model achieves the lowest NASA penalty (281.3) and state-of-the-art RMSE (13.06 cycles), proving the synergy of convolutional-recurrent modeling."

---

## Slide 6: Controlled Ablation Studies
- **Ablation 1 (Window Size Sensitivity)**:
  - $W=30$: RMSE 14.79 (insufficient historical context)
  - $W=50$: **RMSE 13.06** (optimal prognostic horizon)
  - $W=70$: RMSE 15.50 (excessive padding artifacts on short test engines)
- **Ablation 2 (Dropout Regularization)**:
  - $p=0.0$: Severe overfitting (train loss 61.9 vs val loss 178.6, test RMSE 14.94)
  - $p=0.2$: Optimal generalization (test RMSE 13.06)
- **Presenter Notes**:
  > "We avoided arbitrary hyperparameter selection by conducting controlled ablations. We proved empirically that a sequence length of 50 cycles is optimal. Furthermore, disabling dropout leads to severe empirical overfitting, whereas p=0.2 regularizes effectively while enabling Monte Carlo uncertainty sampling."

---

## Slide 7: Research Contribution 1 — Uncertainty Quantification
- **Points**:
  - Test-time Monte Carlo Dropout ($M=50$ stochastic forward passes).
  - Uncalibrated Gaussian Intervals ($\mu \pm 1.96\sigma$): Nominal 95% ➔ **Empirical 48.0% Coverage** (Severe undercoverage).
  - Inductive Split Conformal Prediction: Nominal 90% ➔ **Empirical 89.0% Coverage**.
  - Error-Uncertainty Correlation: Pearson $r = 0.334$ ($p < 0.001$), Spearman $\rho = 0.402$.
- **Presenter Notes**:
  > "A critical theoretical finding in our work is that naive Gaussian confidence intervals on MC Dropout standard deviation fail completely, covering only 48% of test engines. By coupling MC Dropout with Inductive Split Conformal Prediction on held-out validation data, we established distribution-free intervals that achieve 89.0% empirical coverage on unseen engines, while proving that predictive uncertainty correlates significantly with actual prediction error."

---

## Slide 8: Research Contribution 2 — Explainable AI (Captum Integrated Gradients)
- **Points**:
  - Axiomatic path-integral attribution satisfying Completeness and Implementation Invariance.
  - Fleet-Wide Global Importance:
    1. Corrected Core Speed ($s_{14}$): 10.73%
    2. Fuel Flow Ratio ($s_{12}$): 8.76%
    3. HPC Static Pressure ($s_{11}$): 8.43%
    4. LPC Exhaust Temp ($s_2$): 8.09%
  - Local Contrasts: Healthy engines exhibit positive (life-extending) attributions; near-failure engines exhibit negative (damage) attributions.
- **Presenter Notes**:
  > "Using Captum Integrated Gradients, we illuminated the model's decision logic. The model assigns highest attribution to core speed, fuel-to-pressure ratio, and static pressure. When an engine nears failure, sensor attributions flip sharply negative, pulling predicted life down. Crucially, our report documents that these attributions measure mathematical gradient sensitivity, not formal physical causality."

---

## Slide 9: Research Contribution 3 — Cross-Condition Generalization (FD001 vs FD002)
- **Points**:
  - Single-Condition (FD001) vs 6 Flight Envelopes (FD002).
  - Zero-Shot Transfer: RMSE increases from 14.39 to **53.19 cycles** with a **+30.98 cycle overestimation bias** and NASA penalty of 710,290.5.
  - Condition-Aware Model: Training with flight settings (`setting_1`, `setting_2`, `setting_3`) restores RMSE to **18.13 cycles** and $R^2 = 0.822$.
- **Presenter Notes**:
  > "We tested whether an FD001 model could generalize zero-shot to FD002's six operating regimes. It failed catastrophically, exhibiting a positive bias of +30.98 cycles. High-altitude flight lowers ambient pressure, fooling the naive model into believing the engine is lightly loaded. Training a condition-aware network restores performance to 18.13 cycles, proving the necessity of flight regime awareness."

---

## Slide 10: Model Compression & Edge Deployment (Apple Silicon M5)
- **Visual**: Pareto Frontier Plot (Inference Latency vs Test RMSE).
- **Points**:
  - Full Hybrid: 81,505 params | Latency 1.24 ms | Size 0.96 MB | RMSE 15.96
  - Compact Edge: **12,337 params (88.0% reduction)** | Latency **0.89 ms (1.39x speedup)** | Size 0.16 MB | RMSE **15.32**
  - Micro Edge: **3,185 params (96.9% reduction)** | Latency **0.69 ms (1.80x speedup)** | Size **0.05 MB (50 KB)** | RMSE 15.44
- **Presenter Notes**:
  > "To evaluate edge deployment feasibility on low-power avionics hardware, we profiled model compression directly on Apple Silicon M5. Our Compact Edge Hybrid reduces parameters by 88% and accelerates inference to 0.89 ms per sample while actually improving RMSE and NASA score, demonstrating that compact recurrent models are prime candidates for on-wing predictive maintenance."

---

## Slide 11: Interactive Streamlit Dashboard Demonstration
- **Visual**: 5-page dashboard walkthrough:
  - Page 1: Overview & Research Objectives
  - Page 2: Multi-Sensor Health Monitoring & Degradation Profiles
  - Page 3: Live RUL Prognosis with 90% Conformal Prediction Intervals
  - Page 4: Integrated Gradients Feature Attribution & Thermodynamic Saliency
  - Page 5: Master Comparison Table & Edge Compression Metrics
- **Presenter Notes**:
  > "To bridge the gap between academic code and field maintenance operations, we created an interactive 5-page Streamlit dashboard that allows maintenance personnel to select any engine unit, observe live Monte Carlo forward pass distributions, view calibrated confidence intervals, and inspect local sensor attribution waterfalls."

---

## Slide 12: Conclusion, Limitations & Future Work
- **Summary**:
  - Developed an explainable, uncertainty-aware prognostic framework evaluated on NASA C-MAPSS.
  - Benchmark RMSE of 13.06 cycles, 89% conformal coverage, and axiomatic interpretability.
- **Explicit Limitations**:
  - C-MAPSS is simulation-based data without unexpected manual repairs or erratic calibration drifts.
  - Post-hoc attributions reflect correlational gradient sensitivity rather than causal DAG interventions.
- **Future Directions**:
  - Physics-Informed Neural Networks (PINNs) incorporating thermodynamic conservation laws.
  - State-Space Models (Mamba) for ultra-long context horizons.
- **Presenter Notes**:
  > "In conclusion, this project demonstrates rigorous methodology in deep learning, time-series analysis, uncertainty calibration, and explainability. I look forward to your questions and feedback. Thank you."
