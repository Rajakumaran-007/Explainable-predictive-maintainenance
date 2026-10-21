# NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset

## 1. Overview & Provenance
The **C-MAPSS** (Commercial Modular Aero-Propulsion System Simulation) dataset is the preeminent benchmark in prognostic health management (PHM) and Remaining Useful Life (RUL) estimation for complex mechanical systems. Developed at NASA Glenn Research Center, the simulation accurately models the thermodynamic, aerodynamic, and control behaviors of a 90,000 lb thrust-class commercial turbofan jet engine operating across diverse real-world flight regimes.

- **Official Source**: [NASA Open Data Portal - C-MAPSS Jet Engine Simulated Data](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)
- **Primary Reference**: A. Saxena, K. Goebel, D. Simon, and N. Eklund, *"Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation"*, in Proceedings of the IEEE International Conference on Prognostics and Health Management (PHM), Denver, CO, 2008.

---

## 2. Dataset Subsets & Experimental Complexity

The dataset comprises four distinct subsets reflecting progressive real-world operational challenges:

| Dataset Subset | Operating Conditions | Fault Modes Simulated | Training Engines | Testing Engines | Degradation Mechanisms |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **FD001** | 1 (Sea Level) | 1 (HPC Degradation) | 100 | 100 | High Pressure Compressor degradation |
| **FD002** | 6 (Full Flight Envelope) | 1 (HPC Degradation) | 260 | 259 | HPC degradation under operating mode transitions |
| **FD003** | 1 (Sea Level) | 2 (HPC + Fan Degradation) | 100 | 100 | Combined HPC and Fan degradation |
| **FD004** | 6 (Full Flight Envelope) | 2 (HPC + Fan Degradation) | 249 | 248 | HPC and Fan degradation under 6 operating regimes |

---

## 3. Data Format & Column Structure

Each subset includes space-delimited text files without headers:
- `train_FD00*.txt`: Complete run-to-failure trajectories.
- `test_FD00*.txt`: Truncated operational trajectories ending prior to failure.
- `RUL_FD00*.txt`: Vector of ground-truth Remaining Useful Life values corresponding to the final observed cycle of each test engine.

### Matrix Layout (26 Columns):
| Index | Field Name | Description | Units / Scale |
| :---: | :--- | :--- | :--- |
| 1 | `engine_id` | Unique engine unit identifier | Integer (1, 2, ...) |
| 2 | `cycle` | Operational flight cycle count | Discrete time $t \in [1, T]$ |
| 3 | `setting_1` | Flight condition: Altitude | $0 - 42,000$ ft |
| 4 | `setting_2` | Flight condition: Mach number | $0.0 - 0.84$ Mach |
| 5 | `setting_3` | Flight condition: Throttle Resolver Angle (TRA) | $20 - 100^\circ$ |
| 6–26 | `s_1` – `s_21` | High-fidelity thermodynamic and pressure sensor readings | Continuous real values |

### Detailed Sensor Definitions:
- **`s_1`**: Total temperature at fan inlet ($T_{2}$)
- **`s_2`**: Total temperature at LPC outlet ($T_{24}$)
- **`s_3`**: Total temperature at HPC outlet ($T_{30}$)
- **`s_4`**: Total temperature at LPT outlet ($T_{50}$)
- **`s_5`**: Pressure at fan inlet ($P_{2}$)
- **`s_6`**: Total pressure in bypass duct ($P_{15}$)
- **`s_7`**: Total pressure at HPC outlet ($P_{30}$)
- **`s_8`**: Physical fan speed ($N_f$)
- **`s_9`**: Physical core speed ($N_c$)
- **`s_10`**: Engine pressure ratio ($EPR = P_{50} / P_2$)
- **`s_11`**: Static pressure at HPC outlet ($P_{s30}$)
- **`s_12`**: Ratio of fuel flow to static pressure ($W_{f} / P_{s30}$)
- **`s_13`**: Corrected fan speed ($N_{f,corr}$)
- **`s_14`**: Corrected core speed ($N_{c,corr}$)
- **`s_15`**: Bypass ratio ($BPR$)
- **`s_16`**: Burner fuel-air ratio ($FAR$)
- **`s_17`**: Bleed enthalpy ($htBleed$)
- **`s_18`**: Demanded fan speed ($N_{f,dmd}$)
- **`s_19`**: Demanded corrected fan speed ($PCN_{fR,dmd}$)
- **`s_20`**: HPT coolant bleed flow ($W_{31}$)
- **`s_21`**: LPT coolant bleed flow ($W_{32}$)

---

## 4. Ground Truth RUL Generation & Piecewise Linear Degradation Model

### A. Training Trajectories (Run-to-Failure)
In training data, every engine operates continuously until catastrophic failure ($T_{max}$).
Under a pure linear degradation model, the RUL at cycle $t$ is:
$$\text{RUL}_{linear}(t) = T_{max} - t$$

However, physical degradation in rotating machinery is negligible during the initial healthy operating regime. During this stage, sensor measurements reflect standard operational noise rather than component wear. Training a model to distinguish between $\text{RUL} = 250$ and $\text{RUL} = 300$ based solely on healthy sensor readings injects severe label noise and destabilizes gradient descent.

To resolve this, we adopt the standard **piecewise-linear degradation model** (Saxena et al., 2008; Babu et al., 2016):
$$\text{RUL}_{target}(t) = \min\left(T_{max} - t, RUL_{max}\right)$$
where $RUL_{max} = 125$ cycles. When $T_{max} - t > 125$, the target RUL is held constant at 125, representing the healthy initial phase. When $T_{max} - t \le 125$, the target decays linearly to 0 at failure.

### B. Testing Trajectories (Truncated Observations)
In test datasets, observation terminates at an arbitrary cycle $T_{end} < T_{failure}$.
The official `RUL_FD00*.txt` file supplies the exact remaining useful life from cycle $T_{end}$ onward ($R_{end}$).
For any historical cycle $t \le T_{end}$, the true RUL is:
$$\text{RUL}_{true}(t) = R_{end} + (T_{end} - t)$$
$$\text{RUL}_{target}(t) = \min\left(R_{end} + (T_{end} - t), RUL_{max}\right)$$

For official academic benchmarking, **only the prediction at cycle $T_{end}$** is evaluated against $R_{end}$.

---

## 5. Prevention of Data Leakage & Engine-Level Splitting

A frequent error in predictive maintenance research is shuffling sliding windows randomly across the entire dataset. In time-series prognostic modeling, this constitutes **catastrophic temporal data leakage**: windows from engine unit $k$ at cycle $t$ would be in the training set while windows from the same engine $k$ at cycle $t+1$ would appear in the validation set.

**Strict Protocol Implemented:**
1. Splitting is executed **strictly at the engine level** (e.g., engines 1–80 for training, engines 81–100 for validation).
2. All feature scaling parameters (mean, standard deviation, min, max) are calculated **exclusively on the training partition** and applied unchanged to validation and test partitions.
3. Test sets remain completely unseen until final evaluation.

---

## 6. Official NASA Asymmetric Evaluation Metric

Standard symmetric metrics like RMSE and MAE treat overestimating and underestimating RUL identically. In mission-critical aerospace applications:
- **Early Prediction ($\hat{y} < y$):** Engine is scheduled for maintenance earlier than needed. Results in minor operational cost but zero safety hazard.
- **Late Prediction ($\hat{y} > y$):** Engine is predicted to have remaining life when it is actually near failure. Causes catastrophic uncontained in-flight engine failure.

The **NASA PHM 2008 Asymmetric Scoring Function** penalizes late predictions exponentially more severely:
$$d_i = \hat{y}_i - y_i \quad (\text{error in cycles})$$
$$s_i = \begin{cases} \exp\left(-\frac{d_i}{13}\right) - 1, & d_i < 0 \quad (\text{Early Prediction / Safe}) \\ \exp\left(\frac{d_i}{10}\right) - 1, & d_i \ge 0 \quad (\text{Late Prediction / Hazardous}) \end{cases}$$
$$\text{NASA Score} = \sum_{i=1}^{N_{test}} s_i$$
