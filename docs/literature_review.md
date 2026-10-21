# Literature Review: Deep Learning, Uncertainty & Explainability in Prognostics

## 1. Traditional Prognostics & Machine Learning in PHM
Early Remaining Useful Life (RUL) estimation relied on physics-of-failure (PoF) models (Paris' law, crack propagation models) which require exact physical geometry and material properties rarely available in operating commercial fleets. 

With the introduction of the NASA C-MAPSS dataset (Saxena et al., 2008), data-driven prognostic modeling gained widespread traction:
- **Heimes (2008)** implemented early Multi-Layer Perceptrons (MLPs) using recurrent connections on C-MAPSS.
- **Mosallam et al. (2016)** demonstrated non-parametric Bayesian approaches using k-nearest neighbors and Gaussian Process Regression for degradation modeling.
- **Sutrisno et al. (2012)** benchmarked classical Support Vector Regression (SVR) and Random Forests.

## 2. Deep Learning for Time-Series RUL Estimation
- **Babu et al. (2016)** pioneered deep 2D Convolutional Neural Networks on C-MAPSS by treating multivariate sliding windows as 2D spatial patches, demonstrating superiority over SVR and MLPs.
- **Zheng et al. (2017)** developed Long Short-Term Memory (LSTM) networks capable of learning long-term dependencies across operational cycles.
- **Li et al. (2018)** and **Al-Dulaimi et al. (2019)** proposed hybrid architectures combining CNN feature extractors with recurrent layers to jointly leverage local temporal convolution and global recurrent state transitions.

## 3. Uncertainty Quantification in Safety-Critical AI
Deterministic point predictions are inherently insufficient in aviation predictive maintenance:
- **Gal & Ghahramani (2016)** proved that dropout applied at test time in deep networks acts as an approximate variational inference in Bayesian neural networks (Monte Carlo Dropout).
- **Angelopoulos & Bates (2021)** introduced modern conformal prediction frameworks providing distribution-free finite-sample coverage guarantees.

## 4. Explainable AI (XAI) for Industrial Sensor Systems
- **Lundberg & Lee (2017)** unified cooperative game theory (Shapley values) into SHAP for local and global feature attribution.
- **Sundararajan et al. (2017)** formulated Integrated Gradients, satisfying completeness and implementation invariance for deep neural networks.
