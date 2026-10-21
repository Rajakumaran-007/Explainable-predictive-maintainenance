# Methodology: Time-Series Preprocessing, Modeling & Evaluation Protocols

## 1. Data Integrity and Preprocessing Protocol

### 1.1 Invariant Sensor Removal
In the NASA C-MAPSS FD001 dataset, certain sensors exhibit zero or near-zero variance across all operating cycles because operating conditions remain fixed at sea-level.
- Invariant Sensors: `s_1` (Fan inlet temp), `s_5` (Fan inlet pressure), `s_10` (Engine pressure ratio), `s_16` (Fuel-air ratio), `s_18` (Demanded fan speed), `s_19` (Demanded core speed), and `setting_3`.
- Retaining invariant sensors provides zero prognostic mutual information while causing zero-division singularity in standard normalization. Consequently, they are pruned during feature filtering.

### 1.2 Feature Normalization (Leakage-Free Z-Score / Min-Max)
All scaling transformations are fitted **strictly on the training engine trajectories** $\mathcal{E}_{train}$:
$$\mu_j = \frac{1}{N_{train}} \sum_{k \in \mathcal{E}_{train}} x_{k,j}, \quad \sigma_j = \sqrt{\frac{1}{N_{train}} \sum_{k \in \mathcal{E}_{train}} (x_{k,j} - \mu_j)^2}$$
$$x'_{k,j} = \frac{x_{k,j} - \mu_j}{\sigma_j + \epsilon}$$
These parameters are stored and applied unchanged to validation and testing datasets.

### 1.3 Engine-Level Splitting
To avoid temporal correlation leakage between adjacent time steps, the training set of 100 engines is split at the engine unit level:
- Training partition: 80 engines (e.g. IDs 1–80)
- Validation partition: 20 engines (e.g. IDs 81–100)
No window from engine unit $k$ can ever span across partitions.

### 1.4 Test Trajectory Sequence Padding
Test trajectories are terminated at arbitrary cycles prior to engine failure. For trajectories where the total operational cycles $T_{end} < W$ (the sequence window length), edge padding (repeating the initial cycle observation) or zero padding is applied so that an evaluation window of length $W$ is available for final cycle evaluation.

---

## 2. Model Architecture Design

### 2.1 1D-CNN + LSTM Hybrid Network
The proposed architecture processes an input multivariate window $\mathbf{S} \in \mathbb{R}^{W \times D}$:
1. **Temporal 1D Convolutional Blocks**:
   $$\mathbf{H}^{(1)} = \text{ReLU}\left(\text{BatchNorm}\left(\text{Conv1D}_{k=5, c=32}(\mathbf{S})\right)\right)$$
   $$\mathbf{H}^{(2)} = \text{ReLU}\left(\text{BatchNorm}\left(\text{Conv1D}_{k=3, c=64}(\mathbf{H}^{(1)})\right)\right)$$
   Convolutions extract local inter-sensor temporal cross-correlations without requiring manual feature engineering.
2. **Bidirectional / Unidirectional LSTM Layers**:
   $$\mathbf{h}_t, \mathbf{c}_t = \text{LSTM}(\mathbf{H}^{(2)}_t, (\mathbf{h}_{t-1}, \mathbf{c}_{t-1}))$$
   LSTMs maintain hidden state memory across the full temporal window, capturing gradual non-linear monotonic degradation trends.
3. **Dropout Regularization**: Active during training and inference (Monte Carlo Dropout).
4. **Fully Connected Regression Head**:
   $$\hat{y} = \mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \cdot \mathbf{h}_W + \mathbf{b}_1) + b_2$$

---

## 3. Uncertainty Quantification Formulation
For an input window $\mathbf{S}^*$, Monte Carlo Dropout executes $M$ stochastic forward passes with dropout masks active:
$$\hat{y}^{(m)} = f_{\hat{\mathbf{\theta}}^{(m)}}(\mathbf{S}^*), \quad m \in \{1, 2, \dots, M\}$$
- **Epistemic Mean Prediction**: $\mu_* = \frac{1}{M} \sum_{m=1}^M \hat{y}^{(m)}$
- **Epistemic Predictive Variance**: $\sigma^2_* = \frac{1}{M} \sum_{m=1}^M (\hat{y}^{(m)} - \mu_*)^2$
- **Epistemic Standard Deviation**: $\sigma_* = \sqrt{\sigma^2_*}$

### 3.1 Conformal Prediction Intervals
To provide finite-sample coverage guarantees without relying on Gaussian normality assumptions, conformal prediction uses held-out calibration non-conformity scores:
$$\alpha_i = |y_i - \hat{\mu}(x_i)|, \quad i \in \mathcal{D}_{calib}$$
Given error tolerance $\alpha = 0.10$, the $(1-\alpha)$-th empirical quantile $\hat{q}$ guarantees marginal coverage:
$$P\left(y_{new} \in [\hat{\mu}(x_{new}) - \hat{q}, \hat{\mu}(x_{new}) + \hat{q}]\right) \ge 1 - \alpha$$
