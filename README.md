## 1. Sparse Topic Regression & Macroeconomic Forecasting (`1_sparse_topic_regression.ipynb`)

This notebook evaluates the predictive power of business news narratives on inflation.

### Overview
We map high-dimensional textual data (180 news topics derived from the WSJ) to inflation using sparse modeling techniques.

### Methodology
* **Variable Selection:** Fixed-cardinality **LARS (Least Angle Regression)** selecting exactly $k=5$ topics per target to ensure parsimony and interpretability.
* **Estimation:** Post-Selection OLS inference.
* **Evaluation:** Recursive **expanding window** forecasting (Out-of-sample $R^2$ vs. Historical mean).

### 📊 Key Analysis Sections
1.  **Macroeconomic Forecasting:** Predicting Real Activity (Industrial Production, Employment).
2.  **🚀 Extension - Inflation Dynamics:** Investigating the link between news narratives and US Inflation (CPI & PCE), testing the Phillips curve mechanism via text.
3.  **Asset Pricing:** Explaining aggregate Stock Market Returns and Volatility (VIX).
4.  **Sectoral Risk:** Modeling idiosyncratic volatility across 49 Industries (Ken French dataset).
5.  **Validation:** Benchmarking statistical topics against Economic Policy Uncertainty (EPU) indices.

### 📦 Data Sources
* **Text:** Topic concentrations $\theta_t$ (Bybee et al.).
* **Macro:** FRED (St. Louis Fed), FRED-MD.
* **Finance:** Kenneth French Data Library (Industry Portfolios).
* **Policy:** Baker, Bloom & Davis (EPU Indices).