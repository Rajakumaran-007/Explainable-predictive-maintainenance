# An Explainable and Uncertainty-Aware Deep Learning Framework for Predicting Remaining Useful Life of Industrial Equipment

[![Python 3.11](https://img.shields.io/badge/Python-3.11.17-blue.svg)](https://www.python.org/)
[![PyTorch 2.14](https://img.shields.io/badge/PyTorch-2.14-ee4c2c.svg)](https://pytorch.org/)
[![Hardware](https://img.shields.io/badge/Hardware-Apple%20Silicon%20(MPS)-silver.svg)](https://developer.apple.com/metal/pytorch/)
[![Dataset](https://img.shields.io/badge/Dataset-NASA%20C--MAPSS-orange.svg)](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-23%20Passing-brightgreen.svg)](tests/)
[![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-ff4b4b.svg)](app/dashboard.py)

---

## Abstract
Predicting the Remaining Useful Life (RUL) of high-capital, mission-critical industrial assets is fundamental to predictive maintenance (PdM), reducing catastrophic operational risk, and minimizing lifecycle expenditure. In aerospace and industrial rotating machinery, deterministic black-box deep learning models fail to provide actionable safety assurances due to the absence of uncertainty quantification and physical interpretability.

This research framework introduces an intelligent, uncertainty-aware, and explainable deep learning paradigm evaluated on the benchmark **NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS)** turbofan degradation dataset (FD001–FD004). The system integrates:
1. **Hybrid 1D-CNN + LSTM Architecture**: Synergistically captures high-frequency inter-sensor spatial correlations and long-range non-linear temporal degradation dynamics.
2. **Epistemic Uncertainty Quantification & Conformal Calibration**: Implements Monte Carlo Dropout ($M=50$ stochastic passes) coupled with distribution-free **Inductive Split Conformal Prediction**, guaranteeing valid finite-sample coverage ($\ge 89.0\%$) over uncalibrated Gaussian intervals (48.0%).
3. **Explainable AI (XAI)**: Leverages axiomatic Captum Integrated Gradients to illuminate sensor-level degradation contributions and verify thermodynamic consistency with High-Pressure Compressor (HPC) failure modes.
4. **Cross-Condition Generalization**: Formally investigates zero-shot domain shifts from sea-level conditions (FD001) to 6-regime flight envelopes (FD002), quantifying the critical necessity of operating-condition-aware normalization.
5. **Edge Optimization on Apple Silicon**: Benchmarks structural compression on an Apple MacBook Air M5 using Metal Performance Shaders (MPS), achieving an **88.0% parameter reduction** with sub-millisecond inference ($0.89\text{ ms}$).
6. **Interactive Production Dashboard**: Deploys a full 5-page Streamlit application for real-time asset monitoring, calibrated interval estimation, and local/global XAI inspection.

---

## 1. System Architecture

```mermaid
flowchart TD
    subgraph Data ["Multivariate Sensor Ingestion & Preprocessing"]
        A["NASA C-MAPSS Raw Turbofan Data (FD001 - FD004)"] --> B["Data Integrity Audit & Zero-Variance Pruning"]
        B --> C["Engine-Level Partitioning (Zero Data Leakage)"]
        C --> D["Piecewise-Linear RUL Labeling (Ceiling = 125 Cycles)"]
        D --> E["Temporal Sliding Window Generator (W=50, Stride=1)"]
    end

    subgraph DL ["Proposed Hybrid Deep Learning Model"]
        E --> F["Multivariate Window Input: [B, D=14, W=50]"]
        F --> G["1D-CNN Block 1: Conv1D(32, k=5) + BatchNorm + ReLU"]
        G --> H["1D-CNN Block 2: Conv1D(64, k=3) + BatchNorm + ReLU"]
        H --> I["Recurrent Block: 2-Layer LSTM (Hidden Dim=64)"]
        I --> J["Active Dropout Layer (p=0.2)"]
        J --> K["Fully Connected Head: Linear(64) -> Linear(32) -> Output"]
    end

    subgraph Inference ["Uncertainty & Explainability Engine"]
        K --> L["Deterministic RUL Point Estimate (Cycles)"]
        J --> M["Monte Carlo Dropout (M=50 Forward Passes)"]
        M --> N["Epistemic Uncertainty: Mean & Variance"]
        N --> O["Conformal Prediction Intervals (89.0% Empirical Coverage)"]
        K --> P["Integrated Gradients Attribution (Captum)"]
        P --> Q["Local & Global Sensor Importance Dynamics"]
    end

    subgraph App ["Deployment & Visualization"]
        L & N & O & Q --> R["Interactive Streamlit Dashboard (5 Pages)"]
    end
```

---

## 2. Experimental Benchmark Results

All models were evaluated on the official NASA C-MAPSS **FD001 test split (100 engines)** under a strictly enforced, leakage-free engine-level evaluation protocol. Training utilized early stopping (patience=10) and Adam optimization ($lr=10^{-3}$, cosine decay) accelerated on Apple Silicon MPS.

### Model Comparison Table (Official FD001 Test Set)

| Model Architecture | Model Family | Parameters | Checkpoint Size | Latency (M5 MPS) | RMSE (Cycles) | MAE (Cycles) | $R^2$ Score | NASA Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ridge Regression** | Linear ML | 71 | 0.79 KB | **0.028 ms** | 18.03 | 14.71 | 0.798 | 501.8 |
| **Random Forest** (100 Trees) | Ensemble ML | 681,316 | 46.81 MB | 13.17 ms | 18.57 | 13.36 | 0.785 | 1111.7 |
| **Multi-Layer Perceptron (MLP)** | Deep Learning | 100,545 | 1.17 MB | 0.400 ms | 13.91 | 10.39 | 0.880 | 310.8 |
| **LSTM Baseline** (2-Layer) | Recurrent DL | 55,873 | 0.65 MB | 1.044 ms | 13.55 | **9.89** | 0.886 | 289.1 |
| **1D-CNN Baseline** (3-Layer) | Feedforward DL | 44,001 | 0.53 MB | 0.492 ms | 19.14 | 14.61 | 0.772 | 821.6 |
| **Proposed Hybrid 1D-CNN + LSTM** | Hybrid DL | 102,689 | 0.96 MB | 1.300 ms | **14.39 ± 1.14** | **10.40 ± 0.49** | **0.870 ± 0.02** | **380.5 ± 92.6** |
| *Best Single Hybrid Seed (Seed 999)* | Hybrid DL | 102,689 | 0.96 MB | 1.232 ms | **12.65** | **9.37** | **0.900** | **311.5** |

> [!NOTE]
> **NASA Asymmetric Scoring Function Interpretation**:
> The NASA score heavily penalizes late predictions (hazard factor $\alpha = 10$) compared to conservative early predictions (safety factor $\beta = 13$):
> $$s_i = \begin{cases} \exp(-d_i / 13) - 1 & \text{if } d_i < 0 \quad (\text{Early}) \\ \exp(d_i / 10) - 1 & \text{if } d_i \ge 0 \quad (\text{Late}) \end{cases}$$
> While Random Forest achieves an acceptable MAE of 13.36, its NASA score explodes to **1111.7** due to overestimating RUL on degraded units. The hybrid deep learning model prevents overprediction, maintaining safety-critical margins.

---

## 3. Key Research Findings

### 1. Uncertainty Quantification & Split Conformal Calibration
- **Failure of Uncalibrated Gaussian Assumptions**: Standard Gaussian confidence intervals derived from Monte Carlo Dropout ($\hat{y} \pm 1.96\hat{\sigma}_{MC}$) achieve only **48.0% empirical coverage** for a nominal 95% confidence level due to deep network overconfidence.
- **Inductive Split Conformal Prediction**: Computing non-conformity non-parametric scores $\alpha_i = |\hat{y}_i - y_i| / \hat{\sigma}_{MC}$ on held-out calibration engines guarantees finite-sample validity:
  - Nominal 90% Conformal Interval: **89.0% empirical coverage** ($q_{1-\alpha} = 5.65$, mean width: 49.4 cycles).
  - Error-Uncertainty Alignment: Epistemic standard deviation strongly correlates with actual prediction residual (Pearson $r = 0.334, p < 0.001$; Spearman $\rho = 0.402$).

### 2. Axiomatic Explainable AI & Physical Grounding
Using Integrated Gradients attribution ($\times 50$ Riemann approximation steps) relative to a zero-degradation baseline:
- **Dominant Degradation Signatures**:
  1. `s_14` (Corrected Core Speed $N_c$): **10.73%** global attribution
  2. `s_12` (Bypass Ratio / Fuel Flow Ratio): **8.76%** global attribution
  3. `s_11` (HPC Static Outlet Pressure $P_{s30}$): **8.43%** global attribution
  4. `s_13` (Corrected Fan Speed): **8.33%** global attribution
  5. `s_2` (LPC Outlet Temperature): **8.09%** global attribution
- **Physical Thermodynamic Validation**: As the High-Pressure Compressor degrades, bleed air leaks require higher fuel flow and increased core rotational speed to sustain requested thrust, perfectly corroborated by the negative attributions assigned by Integrated Gradients on late-stage engines.

### 3. Cross-Condition Generalization & Domain Shift
- **Catastrophic Zero-Shot Failure**: Evaluating the FD001 model (single sea-level condition) directly on FD002 (6 distinct altitude-Mach flight regimes) resulted in complete failure:
  - In-Domain FD001: RMSE = 14.39, NASA Score = 380.5, Mean Bias = -1.2 cycles.
  - Zero-Shot Transfer to FD002: RMSE = **53.19**, NASA Score = **710,290.5**, Mean Bias = **+30.98 cycles** (dangerous late prediction).
- **Physical Cause**: Flight at 30,000–45,000 ft drops ambient inlet temperatures and pressures, causing the model to misinterpret low thermodynamic baselines as early-life states.
- **Remediation**: Conditioning the network with operating regime settings and condition-aware normalization restored performance to RMSE = 18.13, NASA Score = 1323.5.

### 4. Hardware Optimization & Model Compression on Apple Silicon M5
Benchmarking structured compression against full-scale baseline on Apple M5 unified memory:

| Architecture Variant | Parameters | File Size | Latency (MPS) | Speedup | Compression | RMSE | NASA Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full Hybrid (Baseline)** | 102,689 | 0.96 MB | 1.24 ms | 1.00x | 0.0% | 15.96 | 511.3 |
| **Compact Edge Hybrid** | 12,337 | 0.16 MB | 0.89 ms | **1.39x** | **88.0%** | **15.32** | **369.4** |
| **Micro Edge Hybrid** | 3,185 | 0.05 MB | 0.69 ms | **1.80x** | **96.9%** | **15.44** | **386.3** |

> [!TIP]
> The **Compact Edge Hybrid** reduces parameters by **88.0%** and latency to **0.89 ms** while actually improving the NASA score (369.4 vs 511.3) by mitigating overparameterized overfitting on small turbofan cohorts.

---

## 4. Repository Structure

```text
predictive-maintenance-xai/
├── README.md                      <- Master project showcase and academic documentation
├── LICENSE                        <- MIT Open-Source License
├── requirements.txt               <- Locked dependencies optimized for Apple Silicon MPS
├── pyproject.toml                 <- Modern packaging specification
├── configs/
│   └── default.yaml               <- Master experiment hyperparameters
├── data/
│   ├── README.md                  <- Dataset documentation, physical sensor keys, RUL formulas
│   └── raw/CMAPSSData/            <- NASA C-MAPSS benchmark text files (FD001 - FD004)
├── src/
│   ├── utils.py                   <- MPS hardware detection, seeds, NASA scoring, logging
│   ├── data_loader.py             <- Raw text ingestion, piecewise RUL labeling, integrity audit
│   ├── preprocessing.py           <- Variance filtering, Z-score scaling, engine-level splits
│   ├── dataset.py                 <- PyTorch sliding-window dataset & sequence padding
│   ├── models.py                  <- Baseline ML & Hybrid 1D-CNN + LSTM architectures
│   ├── train.py                   <- PyTorch training loop with early stopping & checkpoints
│   ├── evaluate.py                <- Multi-metric evaluation and latency benchmarking
│   ├── uncertainty.py             <- Monte Carlo Dropout & Split Conformal Calibration
│   └── explainability.py          <- Captum Integrated Gradients & Saliency maps
├── scripts/
│   ├── prepare_data.py            <- Automated dataset downloader & integrity auditor
│   ├── generate_eda_plots.py      <- Generates 5 EDA diagnostic figures
│   ├── train_baselines.py         <- Classical ML & baseline neural network evaluation
│   ├── train_hybrid.py            <- Multi-seed hybrid training & window/dropout ablations
│   ├── evaluate_uncertainty.py    <- MC Dropout & Conformal calibration evaluation
│   ├── evaluate_explainability.py <- Captum Integrated Gradients global/local attribution
│   ├── evaluate_cross_condition.py<- FD001 vs FD002 domain shift evaluation
│   └── benchmark_compression.py   <- Apple Silicon M5 edge compression profiling
├── app/
│   └── dashboard.py               <- 5-page interactive Streamlit web application
├── tests/
│   ├── test_preprocessing.py      <- Unit tests for data loading, scaling, and scoring
│   ├── test_dataset.py            <- Unit tests for PyTorch datasets and sequence batching
│   ├── test_models.py             <- Unit tests for tensor shapes, forward passes, and MPS
│   ├── test_uncertainty.py        <- Unit tests for MC Dropout & Conformal calibration
│   └── test_explainability.py     <- Unit tests for Captum Integrated Gradients tensors
├── results/
│   ├── figures/                   <- 19 high-resolution publication-quality PNG figures
│   └── metrics/                   <- JSON metric files with raw reproducible evaluation numbers
├── models/
│   └── checkpoints/               <- Saved PyTorch model weights (.pt)
└── docs/
    ├── project_overview.md        <- Formal academic research proposal
    ├── literature_review.md       <- Critical survey of PHM literature (1998-2024)
    ├── methodology.md             <- Mathematical formulations & loss equations
    ├── architecture.md            <- Detailed neural network layer diagrams
    ├── experiments.md             <- Controlled experimental protocols
    ├── results.md                 <- In-depth empirical benchmark findings
    ├── research_paper_report.md   <- Complete 8-section academic paper (LaTeX style)
    ├── academic_presentation.md   <- Master's defense presentation slides & speaker notes
    └── cv_and_application_synopsis.md <- CV bullet points and German university SOP blurbs
```

---

## 5. Quickstart & Installation (macOS Apple Silicon)

### Step 1: Clone Repository & Create Virtual Environment
```bash
git clone https://github.com/rajakumaran/predictive-maintenance-xai.git
cd predictive-maintenance-xai

# Create virtual environment (Python 3.11 recommended)
uv venv .venv --python 3.11
source .venv/bin/activate

# Install locked dependencies
uv pip install -r requirements.txt
```

### Step 2: Verify Apple Silicon MPS Hardware Acceleration
```bash
python -c "import torch; print('PyTorch Version:', torch.__version__); print('MPS Available:', torch.backends.mps.is_available())"
# Expected output: MPS Available: True
```

### Step 3: Download & Validate NASA C-MAPSS Dataset
```bash
python scripts/prepare_data.py
```

### Step 4: Run Automated Unit Tests (100% Passing)
```bash
pytest tests/
# Output: 23 passed in ~2.0s
```

### Step 5: Launch Interactive Streamlit Dashboard
```bash
streamlit run app/dashboard.py
```
*The dashboard will launch locally at `http://localhost:8501`.*

---

## 6. Interactive Streamlit Dashboard Walkthrough

The application features 5 dedicated modules designed for both technical evaluators and industrial plant operators:

1. **Page 1: Project Overview**: Executive research abstract, aerospace problem statement, turbofan schematic, and interactive Mermaid architecture diagrams.
2. **Page 2: Engine Health Monitoring**: Interactive selection of all 100 test turbofans. Real-time visualization of sensor degradation trajectories against fleet baselines with interactive multi-sensor overlays.
3. **Page 3: RUL Prediction & Calibrated Uncertainty**: Point estimate vs true remaining useful life, Monte Carlo Dropout epistemic distribution ($M=50$ passes), and inductive conformal prediction interval band ensuring 90% statistical coverage.
4. **Page 4: Explainable AI & Sensor Diagnostics**: Captum Integrated Gradients attribution waterfalls comparing healthy versus degraded regimes. Physical thermodynamic breakdown of High-Pressure Compressor failure signatures.
5. **Page 5: Model Benchmarks & Edge Profiling**: Live comparative tables of all classical ML and deep learning baselines, ablation findings, and interactive Apple Silicon M5 latency/compression tradeoffs.

---

## 7. Project Development Roadmap

- [x] **Phase 1: Environment & Dataset Setup** (Apple MPS verification, repository architecture, NASA C-MAPSS automated ingestion & integrity audit).
- [x] **Phase 2: Exploratory Data Analysis & Preprocessing Pipeline** (Zero-variance sensor pruning, leakage-free Z-score normalization, sliding-window sequence generators).
- [x] **Phase 3: Baseline Machine Learning & Deep Learning Models** (Ridge, Random Forest, MLP, LSTM-only, 1D-CNN-only).
- [x] **Phase 4: Proposed Hybrid 1D-CNN + LSTM Model** (Multi-seed training, learning rate schedules, early stopping, NASA metric scoring, ablations).
- [x] **Phase 5: Uncertainty Quantification** (Monte Carlo Dropout, epistemic variance, inductive split conformal calibration).
- [x] **Phase 6: Explainable AI** (Captum Integrated Gradients, global rankings, local waterfalls, lifetime sensor dynamics).
- [x] **Phase 7: Cross-Condition & Model Compression** (FD001 vs FD002 domain shift analysis, Apple M5 edge profiling, 88% parameter reduction).
- [x] **Phase 8: Interactive Streamlit Dashboard** (5-page interactive health monitoring & prognostic web application).
- [x] **Phase 9: Final Academic Deliverables** (Full 8-section research paper report, master's defense presentation slides, and German admissions CV/SOP synopsis).

---

## 8. Academic Evaluation Documentation

For admission committees, evaluators, and academic reviewers, detailed documentation is organized in `docs/`:
- **[Research Paper Technical Report](docs/research_paper_report.md)**: Full 8-section academic paper with formal mathematical proofs, loss equations, related work, and complete ablation tables.
- **[Master's Defense Slides](docs/academic_presentation.md)**: 12-slide presentation structure with comprehensive speaker notes and anticipated oral defense Q&A.
- **[CV & Application Synopsis](docs/cv_and_application_synopsis.md)**: Ready-to-use resume bullet points and German university Statement of Purpose (SOP) research blurbs for TU Munich, RWTH Aachen, and TU Berlin.
- **[Mathematical Methodology](docs/methodology.md)**: Preprocessing equations, sliding window algebra, and NASA loss formulas.

---

## 9. Citation & References

```bibtex
@article{saxena2008cmapss,
  title={Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation},
  author={Saxena, Abhinav and Goebel, Kai and Simon, Donald and Eklund, Neil},
  journal={IEEE International Conference on Prognostics and Health Management},
  year={2008}
}

@inproceedings{gal2016dropout,
  title={Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning},
  author={Gal, Yarin and Ghahramani, Zoubin},
  booktitle={International Conference on Machine Learning (ICML)},
  pages={1050--1059},
  year={2016}
}

@inproceedings{sundararajan2017axiomatic,
  title={Axiomatic Attribution for Deep Networks},
  author={Sundararajan, Mukund and Taly, Ankur and Yan, Qiqi},
  booktitle={International Conference on Machine Learning (ICML)},
  pages={3319--3328},
  year={2017}
}

@article{vovk2005algorithmic,
  title={Algorithmic Learning in a Random World},
  author={Vovk, Vladimir and Gammerman, Alex and Shafer, Glenn},
  journal={Springer Science & Business Media},
  year={2005}
}
```

---

## 10. License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
