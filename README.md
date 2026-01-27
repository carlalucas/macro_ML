## 1. Sparse Topic Regression & Macroeconomic Forecasting (`1_sparse_topic_regression.ipynb`)

This notebook evaluates the predictive power of business news narratives on inflation.

### Overview
We map high-dimensional textual data (180 news topics derived from the WSJ) to inflation using a sparse modeling technique.

### Methodology
* **Variable Selection:** Fixed-cardinality **LARS (Least Angle Regression)** selecting exactly $k=5$ topics per target to ensure parsimony and interpretability.
* **Estimation:** Post-Selection OLS inference.
* **Evaluation:** Recursive **expanding window** forecasting (Out-of-sample $R^2$ vs. Historical mean).

### Data sources
* **Text:** Topic concentrations $\theta_t$ (Bybee et al.).
* **Macro:** FRED (St. Louis Fed).