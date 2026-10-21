# Project Overview: Explainable & Uncertainty-Aware RUL Prediction

## 1. Executive Summary & Academic Scope
This research-oriented project develops an intelligent, end-to-end predictive maintenance framework designed for high-consequence industrial machinery, specifically turbofan jet engines from the NASA C-MAPSS benchmark suite. 

Engineered specifically as a minor research project for submission to top German public universities (such as TU Munich, RWTH Aachen, TU Berlin, University of Stuttgart, Karlsruhe Institute of Technology, and FAU Erlangen-Nürnberg for M.Sc. Artificial Intelligence / Data Science admissions), this work emphasizes:
1. **Methodological Rigor**: Complete mathematical formalization of remaining useful life estimation, error bounds, and evaluation protocols.
2. **Hybrid Deep Learning**: Synergistic integration of 1D Convolutional Neural Networks (for extracting local thermodynamic sensor patterns) and Long Short-Term Memory networks (for capturing long-term non-linear degradation dynamics).
3. **Epistemic Uncertainty Quantification**: Monte Carlo Dropout inference coupled with empirical calibration and conformal prediction intervals.
4. **Explainable AI (XAI)**: Global and local post-hoc feature attribution using SHAP (SHapley Additive exPlanations) and Captum Integrated Gradients, distinguishing mathematical correlation from physical degradation indicators.
5. **Cross-Condition Robustness**: Systematic investigation of domain shift and performance stability across complex multi-regime operational envelopes (FD001 vs FD002/FD004).
6. **Edge & Hardware Efficiency**: Measured inference latency, memory footprint, and MPS hardware acceleration benchmarks on Apple Silicon (ARM64 unified memory architecture).

---

## 2. Research Questions (RQs)

- **RQ1 (Architectural Synergy):** Does a hybrid 1D-CNN + LSTM architecture outperform standalone recurrent (LSTM) and purely convolutional (1D-CNN) baselines in predicting Remaining Useful Life on complex multivariate sensor time-series?
- **RQ2 (Uncertainty Reliability):** Can Monte Carlo Dropout variance accurately signal difficult, degraded, or out-of-distribution engine operating states, and how do empirical prediction intervals compare with conformalized coverage guarantees?
- **RQ3 (Interpretability & Physical Consistency):** Do gradient and perturbation-based attribution methods (SHAP / Integrated Gradients) identify sensor degradation signatures that align with physical thermodynamic wear phenomena (e.g., HPC outlet temperature $T_{30}$, LPT outlet temperature $T_{50}$, bleed enthalpy)?
- **RQ4 (Domain Shift & Generalization):** How do models trained under single operating conditions (FD001) behave when exposed to multiple flight envelope transitions (FD002/FD004), and how does condition-aware normalization mitigate performance degradation?

---

## 3. Mathematical Foundations

### 3.1 Remaining Useful Life (RUL) Formulation
Let an industrial asset $i$ generate a sequence of multivariate observations over discrete operational cycles $t \in \{1, 2, \dots, T_i\}$:
$$\mathbf{X}_i = [\mathbf{x}_{i,1}, \mathbf{x}_{i,2}, \dots, \mathbf{x}_{i,T_i}]^\top \in \mathbb{R}^{T_i \times D}$$
where $D = D_{settings} + D_{sensors}$ denotes the number of operational settings and sensor measurements.

The exact Remaining Useful Life at time $t$ for an engine failing at cycle $T_i$ is:
$$\text{RUL}^*_{i,t} = T_i - t$$

Under the piecewise-linear degradation model (Saxena et al., 2008), the regression target $y_{i,t}$ is clipped by an upper threshold $RUL_{max}$:
$$y_{i,t} = \min(T_i - t, RUL_{max})$$

### 3.2 Sliding Window Sequence Generation
Given a window length $W$ and stride $S$, the temporal input sample ending at cycle $t$ is:
$$\mathbf{S}_{i,t} = \mathbf{X}_{i, t - W + 1 : t} \in \mathbb{R}^{W \times D}$$
The learning objective is to optimize parameterized model weights $\mathbf{\theta}^*$ minimizing the empirical risk over training engines $\mathcal{E}_{train}$:
$$\mathbf{\theta}^* = \arg\min_{\mathbf{\theta}} \frac{1}{|\mathcal{D}|} \sum_{(\mathbf{S}, y) \in \mathcal{D}} \mathcal{L}(f_\mathbf{\theta}(\mathbf{S}), y) + \lambda \|\mathbf{\theta}\|_2^2$$

### 3.3 Asymmetric NASA PHM Scoring
Conventional symmetric loss functions ($L_1, L_2$) fail to distinguish the operational risk of overestimating vs underestimating equipment lifespan:
$$d_j = \hat{y}_j - y_j$$
$$S = \sum_{j=1}^{N_{test}} \begin{cases} \exp\left(-\frac{d_j}{13}\right) - 1, & d_j < 0 \quad (\text{Early Alert}) \\ \exp\left(\frac{d_j}{10}\right) - 1, & d_j \ge 0 \quad (\text{Late Alert / Failure Risk}) \end{cases}$$
Because $10 < 13$, positive error (delayed maintenance) receives an exponential penalty growth rate $e^{0.10 d_j}$ that dwarfs the early prediction penalty $e^{0.0769 |d_j|}$.

---

## 4. Hardware Optimization & macOS MPS Acceleration
Experiments are engineered to run natively on Apple Silicon architecture using PyTorch's Metal Performance Shaders (`mps`) backend:
- Unified memory architecture eliminates PCIe host-to-device bus transfer bottlenecks.
- Batch sizes and sequence lengths are tuned to optimize GPU core utilization without exhausting unified memory cache lines.
- Multi-worker dataloaders default to `num_workers = 0` on macOS to bypass Darwin IPC shared memory serialization overhead.
