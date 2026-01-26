import pandas as pd
import pandas_datareader.data as web
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import lars_path, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# ==============================================================================
# 0. CONFIGURATION
# ==============================================================================
print("INITIALIZATION...")

MIN_TRAIN_SIZE = 120  # 10 years initialization for OOS
N_VARS = 5            # "Exactly five coefficients" (Paper Section 4.1)

def get_stars(p):
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.1: return "*"
    return ""

# ==============================================================================
# 1. DATA LOADING
# ==============================================================================
print(">>> 1. LOADING DATA...")

# A. THETA & PHI
try:
    theta = pd.read_csv('data/theta_monthly.csv', sep=None, engine='python')
    date_col = next((c for c in theta.columns if 'date' in str(c).lower()), theta.columns[0])
    theta[date_col] = pd.to_datetime(theta[date_col])
    theta.set_index(date_col, inplace=True)
    theta = theta.select_dtypes(include=[np.number])
    theta.columns = [str(i) for i in range(theta.shape[1])]
    
    phi = pd.read_csv('data/phi_scaled.csv', sep=None, engine='python', index_col=0)
    labels_map = {}
    for i in range(min(theta.shape[1], phi.shape[1])):
        words = phi.iloc[:, i].sort_values(ascending=False).head(2).index.tolist()
        labels_map[str(i)] = "-".join([str(w).lower() for w in words])
except:
    raise ValueError("No 'theta_monthly.csv' and 'phi_scaled.csv' in data/")

# B. MACRO FRED
tickers = {
    'INDPRO': 'IP',             # Industrial Production
    'PAYEMS': 'Emp',            # Employment
    'SPASTT01USM661N': 'MktRet',# S&P 500 Returns (OECD)
    'VIXCLS': 'MktVol'          # Volatility
}

try:
    fred_raw = web.DataReader(list(tickers.keys()), 'fred', start='1980-01-01')
    fred_m = fred_raw.resample('MS').mean()
    data = pd.DataFrame(index=fred_m.index)
    
    # Transformations
    data['IP'] = np.log(fred_m['INDPRO']).diff() * 100
    data['Emp'] = np.log(fred_m['PAYEMS']).diff() * 100
    data['MktRet'] = np.log(fred_m['SPASTT01USM661N']).diff() * 100
    data['MktVol'] = fred_m['VIXCLS'] # Already stationary
    
    data = data.dropna()
except:
    raise ConnectionError("FRED Error")

# Alignment
common_idx = theta.index.intersection(data.index)
theta = theta.loc[common_idx]
data = data.loc[common_idx]

# ==============================================================================
# LASSO 5 VARIABLES & OOS
# ==============================================================================

def select_exact_k_variables(X, y, k=5):
    """
    Uses LARS (Least Angle Regression) to find EXACTLY k variables.
    It's more precise than LassoCV for fixing an exact number of non-zero variables.
    """
    # lars_path returns the regularization path
    # method='lasso' ensures we respect the L1 constraint
    alphas, active, coefs = lars_path(X.values, y.values, method='lasso')
    
    # We look for the step where we have k active variables
    # coefs is of size (n_features, n_steps)
    n_active = np.count_nonzero(coefs, axis=0)
    
    # We take the index where we have exactly k variables (or the closest)
    # Generally LARS adds 1 var at a time, so we find k easily.
    step_idx = np.where(n_active == k)[0]
    
    if len(step_idx) > 0:
        chosen_step = step_idx[0]
    else:
        # Fallback if LARS skips a step (rare)
        chosen_step = np.argmin(np.abs(n_active - k))
    
    # We retrieve the indices of non-zero variables at this step
    active_indices = np.where(coefs[:, chosen_step] != 0)[0]
    
    # We return the column names
    return X.columns[active_indices].tolist()

def run_analysis_section_4_1(target_name, y_raw, X_raw):
    print(f"\nProcessing: {target_name}...")
    
    # 1. STANDARDIZATION (SD Units)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_std = pd.DataFrame(scaler_X.fit_transform(X_raw), index=X_raw.index, columns=X_raw.columns)
    y_std = pd.Series(scaler_y.fit_transform(y_raw.values.reshape(-1, 1)).flatten(), index=y_raw.index)
    
    # --------------------------------------------------------------------------
    # A. IN-SAMPLE ANALYSIS (TABLE)
    # --------------------------------------------------------------------------
    
    # Selection of 5 variables on the entire sample
    top_5_vars = select_exact_k_variables(X_std, y_std, k=N_VARS)
    
    # OLS regression on these 5 variables (Simplified Post-Selection Inference)
    # Note: Tibshirani et al. (2016) requires complex adjustments of degrees of freedom. Here we do the standard approximation "OLS on Active Set".
    X_ols = sm.add_constant(X_std[top_5_vars])
    model = sm.OLS(y_std, X_ols).fit()
    
    # Calculate In-Sample R2
    r2_in = model.rsquared
    
    # In-Sample Predictions (for the graph)
    preds_in_std = model.predict(X_ols)
    
    # --------------------------------------------------------------------------
    # B. OUT-OF-SAMPLE ANALYSIS (R2 OOS)
    # --------------------------------------------------------------------------
    trues_oos = []
    preds_oos = []
    hist_means = []
    
    # Expanding Window Loop
    # ATTENTION: We must redo the Lasso selection at EACH step to not cheat (no look-ahead bias).
    
    # Optimization: We only do the loop if necessary because it's very heavy
    # We do a 1-month step
    
    for i in range(MIN_TRAIN_SIZE, len(y_std)):
        # Train (Expanding)
        y_train = y_std.iloc[:i]
        X_train = X_std.iloc[:i]
        
        # Test (1 point)
        y_test_pt = y_std.iloc[i]
        X_test_row = X_std.iloc[[i]]
        
        # 1. Select 5 variables on the Train
        vars_t = select_exact_k_variables(X_train, y_train, k=N_VARS)
        
        # 2. Fit OLS on these 5 variables
        reg = LinearRegression()
        reg.fit(X_train[vars_t], y_train)
        
        # 3. Predict
        pred = reg.predict(X_test_row[vars_t])[0]
        
        preds_oos.append(pred)
        trues_oos.append(y_test_pt)
        hist_means.append(y_train.mean())

    # Calculate OOS R2 (Campbell & Thompson)
    mse_model = np.mean((np.array(trues_oos) - np.array(preds_oos))**2)
    mse_bench = np.mean((np.array(trues_oos) - np.array(hist_means))**2)
    r2_out = 1 - (mse_model / mse_bench)
    
    # --------------------------------------------------------------------------
    # C. TABLES
    # --------------------------------------------------------------------------
    table_data = []
    for var in top_5_vars:
        table_data.append({
            'Topic': labels_map.get(var, f"Topic {var}"),
            'Coeff.': model.params[var],
            'p-val': model.pvalues[var]
        })
    
    df_res = pd.DataFrame(table_data).sort_values(by='Coeff.', key=abs, ascending=False)
    
    print(f"\n{'-'*60}")
    print(f"TABLE : {target_name.upper()}")
    print(f"{'-'*60}")
    print(f"{'Topic':<30} | {'Coeff.':<8} | {'p-val':<8}")
    print(f"{'-'*60}")
    for _, row in df_res.iterrows():
        print(f"{row['Topic']:<30} | {row['Coeff.']:<8.2f} | {row['p-val']:<8.2f}")
    print(f"{'-'*60}")
    print(f"In-Sample R2     : {r2_in:.2f}")
    print(f"Out-of-Sample R2 : {r2_out:.2f}")
    print(f"{'-'*60}")
    
    # --------------------------------------------------------------------------
    # D. GRAPHS (ACTUAL vs PREDICTED)
    # --------------------------------------------------------------------------
    # The paper's graph shows "Actual" and "Predicted"
    # The data is "Standardized" (mean 0), so it oscillates around 0
    
    # 1. Calcul des prédictions In-Sample (sur tout l'historique avec le modèle final)
    # On refait une sélection sur TOUT X_std
    vars_final = select_exact_k_variables(X_std, y_std, k=N_VARS)
    reg_final = LinearRegression().fit(X_std[vars_final], y_std)
    preds_in_sample = reg_final.predict(X_std[vars_final])

# 2. Alignement des dates pour le plot
# OOS commence à MIN_TRAIN_SIZE
    dates_oos = y_std.index[MIN_TRAIN_SIZE:] 
    preds_oos_aligned = pd.Series(preds_oos, index=dates_oos)

    plt.figure(figsize=(12, 5))

# Données Réelles
    plt.plot(y_std.index, y_std, 'k-', lw=1, alpha=0.4, label='Actual data')

# In-Sample (Ce que montre le papier)
    plt.plot(y_std.index, preds_in_sample, 'b--', lw=1, alpha=0.8, label='In-Sample Fit (Explain)')

# Out-of-Sample (La vérité terrain)
    plt.plot(dates_oos, preds_oos_aligned, 'r-', lw=1.5, label='Out-of-Sample Forecast (Forecast)')

    plt.axvline(y_std.index[MIN_TRAIN_SIZE], color='gray', linestyle=':', label='Beginning OOS')
    plt.title(f"{target_name}: Explanation (blue) vs real forecast (red)")
    plt.legend()
    plt.tight_layout()
    plt.show()
# ==============================================================================
# EXECUTION
# ==============================================================================

# Exact reproduction of the 4 variables from Section 4.1
# 1. Industrial Production
run_analysis_section_4_1("Industrial Production Growth", data['IP'], theta)

# 2. Employment
run_analysis_section_4_1("Employment Growth", data['Emp'], theta)

# 3. Market Returns
run_analysis_section_4_1("Market Returns", data['MktRet'], theta)

# 4. Market Volatility
run_analysis_section_4_1("Market Volatility", data['MktVol'], theta)