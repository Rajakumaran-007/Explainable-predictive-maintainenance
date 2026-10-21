# Project Limitations & Scientific Scope

## 1. Simulated vs Real-World Physical Degradation
The C-MAPSS dataset is generated using high-fidelity thermo-dynamical simulation code. While it introduces realistic operational noise and damage propagation curves, it does not fully encompass unpredictable real-world operational factors such as unexpected maintenance interventions, erratic sensor calibration drifts, or physical contamination.

## 2. Epistemic vs Aleatoric Uncertainty Scope
Monte Carlo Dropout approximates epistemic uncertainty (model parameter uncertainty due to finite training observations). It does not explicitly isolate heteroscedastic aleatoric uncertainty (intrinsic sensor noise). Conformal prediction intervals provide rigorous marginal coverage guarantees, but conditional coverage across specific operating sub-populations requires further investigation.

## 3. Post-Hoc Interpretability vs Physical Causality
Feature attributions produced by SHAP or Integrated Gradients quantify how sensitive the neural network's output is to specific sensor variations. Attribution is fundamentally correlational and does not constitute mathematical proof of physical causality without domain-specific causal DAGs (Directed Acyclic Graphs).
