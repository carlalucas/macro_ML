# Replication: Bybee et al. (2021) - Section 4.1

This project replicates the specific methodology and results from **Section 4.1: Selection Via Lasso Regression** of the paper *"Business News and Business Cycles"* by Bybee, Kelly, Manela, and Xiu (2021).

The goal is to demonstrate that news attention (extracted from WSJ articles) corresponds closely with key macroeconomic indicators (Industrial Production, Employment, Market Returns, and Volatility).

## Key Features

* **Exact-5 Variable Selection (LARS):** Unlike standard Lasso which selects variables based on a penalty parameter $\lambda$, this script uses **Least Angle Regression (LARS)** to select **exactly 5 active topics** for each regression model. This strictly adheres to the paper's requirement for interpretability and comparability.
* **Expanding Window Forecast (OOS):** Implements a recursive out-of-sample (OOS) forecasting loop. At each time step $t$, the model re-selects the top 5 topics based *only* on past data (up to $t$) to predict $t+1$. This rigorous approach avoids look-ahead bias.
* **Post-Selection Inference:** Calculates coefficients and p-values using an OLS regression on the selected active set (Post-Lasso OLS), providing standard errors adjusted for model selection.
* **Dual-Track Analysis:**
    * **In-Sample Analysis:** Fits the model on the full dataset to demonstrate explanatory power and identify historical narrative drivers.
    * **Out-of-Sample Analysis:** Measures the true predictive power ($R^2_{OOS}$) simulating a real-time forecaster.
* **Visualization:** Generates dual-line plots comparing "In-Sample Fit" (blue), "Out-of-Sample Forecast" (red), and actual economic data (black).

## Outputs

For each target variable (Industrial Production Growth, Employment Growth, Market Returns, Market Volatility), the script outputs:

1.  **Regression Table:** A formatted table listing the top 5 news topics, their standardized coefficients ($\beta$), and p-values.
2.  **Performance Metrics:**
    * **In-Sample $R^2$:** How well news explains past variations.
    * **Out-of-Sample $R^2$:** How well news predicts future variations compared to a historical mean benchmark.
3.  **Time-Series Plot:** A visual comparison showing the divergence between the model's ex-post explanation (In-Sample) and ex-ante prediction (Out-of-Sample).

## Dependencies

Install the required packages using:
pip install pandas numpy matplotlib scikit-learn statsmodels pandas-datareader scipy