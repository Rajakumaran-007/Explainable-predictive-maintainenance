# Experimental Protocols & Benchmarking Matrix

## 1. Experimental Protocol
- **Random Seeds**: 42, 123, 999 (multi-seed evaluation for mean ± std reporting).
- **Subsets Evaluated**:
  - Primary Benchmark: FD001 (1 condition, 1 fault mode).
  - Cross-condition Benchmarks: FD002, FD003, FD004.
- **Window Lengths Evaluated**: $W \in \{30, 50, 70\}$ cycles.
- **Batch Sizes Evaluated**: $B \in \{16, 32, 64\}$.
- **Learning Rates**: $\eta \in \{10^{-3}, 10^{-4}\}$.
- **Evaluation Points**: Final observed cycle $T_{end}$ for each test engine against official NASA ground-truth vectors.

## 2. Planned Comparison Matrix
1. Ridge Regression Baseline
2. Random Forest Regressor Baseline
3. Simple MLP Baseline
4. LSTM-only Model
5. 1D-CNN-only Model
6. Proposed Hybrid 1D-CNN + LSTM Model
