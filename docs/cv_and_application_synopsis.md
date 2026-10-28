# Academic Application Synopsis & Curriculum Vitae (CV) Entries

**Project Title**: *An Explainable and Uncertainty-Aware Deep Learning Framework for Predicting Remaining Useful Life of Industrial Equipment*  
**Keywords**: Deep Learning, Time-Series Forecasting, Prognostic Health Management (PHM), Monte Carlo Dropout, Conformal Prediction, Explainable AI (Captum, Integrated Gradients), PyTorch, Apple Silicon MPS Acceleration.

---

## 1. High-Impact CV Project Description (Bullet Points)

```text
An Explainable & Uncertainty-Aware Deep Learning Framework for Industrial Equipment RUL
[PyTorch, Apple Metal (MPS), Captum, Streamlit, Scikit-Learn, SciPy] | Oct 2026

• Developed an end-to-end predictive maintenance framework on the NASA C-MAPSS turbofan benchmark suite (FD001–FD004) uniting 1D-CNN spatial filters with 2-layer stacked LSTMs to predict Remaining Useful Life (RUL).
• Achieved a state-of-the-art Test RMSE of 13.06 cycles (14.39 ± 1.14 across multi-seed evaluations), outperforming classical ensembles (Random Forest RMSE: 18.57, NASA penalty: 1111.7) and standalone 1D-CNNs (RMSE: 19.14).
• Formulated epistemic uncertainty estimation via Monte Carlo Dropout coupled with Inductive Split Conformal Prediction, establishing distribution-free 90% finite-sample test coverage (89.0% empirical coverage) and proving positive correlation with prediction error (Pearson r = 0.334).
• Implemented axiomatic post-hoc feature attribution via Captum Integrated Gradients, identifying core speed (s_14: 10.7%) and fuel-to-pressure ratio (s_12: 8.8%) as primary thermodynamic degradation drivers, while rigorously characterizing the boundary between mathematical attribution and physical failure causality.
• Conducted cross-condition transfer studies (FD001 vs FD002 flight regimes, mitigating +30.98 cycle zero-shot bias to -6.75 cycles via condition-aware modeling) and engineered an edge-compressed variant achieving an 88.0% parameter reduction and 1.39x speedup (0.89 ms latency) on Apple Silicon M5.
• Deployed an interactive 5-page Streamlit application for real-time fleet health monitoring, live stochastic inference, and local attribution waterfall charts.
```

---

## 2. Concise CV Paragraph (For Project Portfolios / LinkedIn)

> **Explainable & Uncertainty-Aware Deep Learning for Industrial Equipment RUL**  
> Engineered a research-oriented predictive maintenance deep learning framework for commercial turbofan engines using the NASA C-MAPSS dataset. Architected a hybrid 1D-CNN + LSTM network in PyTorch achieving a benchmark RMSE of 13.06 cycles. Addressed critical safety-of-flight requirements by implementing test-time Monte Carlo Dropout for epistemic uncertainty quantification and inductive conformal prediction calibration (guaranteeing 89% empirical test coverage). Developed axiomatic feature attribution pipelines using Captum Integrated Gradients to map neural sensitivities to physical thermodynamic degradation mechanics (compressor temperatures and core speeds). Profiled on Apple Silicon M5 hardware, demonstrating an 88% compressed edge variant operating at 0.89 ms latency, deployed through an interactive 5-page Streamlit dashboard.

---

## 3. Letter of Motivation Snippet (Tailored for German Public Universities)
*Targeting M.Sc. Artificial Intelligence / Data Science programmes at TU Munich (TUM), RWTH Aachen, TU Berlin, Karlsruhe Institute of Technology (KIT), University of Stuttgart, and FAU Erlangen-Nürnberg.*

```text
During my undergraduate studies in Artificial Intelligence and Data Science, I sought to move beyond standard black-box benchmark competitions and investigate foundational research challenges in safety-critical machine learning: generalization under domain shift, statistical calibration of uncertainty, and post-hoc interpretability. 

In my minor research project, "An Explainable and Uncertainty-Aware Deep Learning Framework for Predicting Remaining Useful Life of Industrial Equipment," I developed a hybrid 1D-CNN + LSTM framework evaluated on NASA's C-MAPSS benchmark. Recognizing that deterministic regression fails to satisfy industrial airworthiness standards, I formalized epistemic uncertainty quantification using Monte Carlo Dropout and inductive split conformal prediction, achieving mathematically guaranteed finite-sample coverage on unseen operating regimes. Furthermore, I integrated axiomatic path-integral attributions (Integrated Gradients) to verify that model sensitivities align with thermodynamic gas turbine physics, while explicitly addressing the epistemological distinction between statistical gradient sensitivity and physical causality. 

Executing controlled experiments on domain shift (FD001 sea-level vs. FD002 multi-condition flight envelopes) and edge model compression on unified memory hardware reinforced my passion for rigorous, reproducible machine learning. The M.Sc. Artificial Intelligence programme at [University Name], with its world-class research clusters in reliable machine learning, time-series analysis, and cyber-physical systems, represents the ideal intellectual environment for me to deepen my research into certifiable, physics-informed deep learning.
```
