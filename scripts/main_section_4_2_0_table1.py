import pandas as pd
import pandas_datareader.data as web
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import lars_path, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import warnings

# 0. CONFIGURATION
print("INITIALIZATION...")
warnings.filterwarnings("ignore")

MIN_TRAIN_SIZE = 120  # 10 years initialization
N_VARS = 5            # Five coefficients

def get_stars(p):
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.1: return "*"
    return ""

# 1. DATA LOADING (CORRECTED MAPPING)
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
    
    # --- INTELLIGENT MAPPING (THE FIX) ---
    # On utilise les noms de colonnes de PHI qui sont les "vrais" noms du papier
    labels_map = {}
    for i in range(min(theta.shape[1], phi.shape[1])):
        # phi.columns[i] est le nom propre (ex: "Recession")
        clean_name = phi.columns[i].strip()
        
        # Sécurité : si le nom est un chiffre (cas rare), on génère des mots-clés
        if str(clean_name).isdigit():
             words = phi.iloc[:, i].sort_values(ascending=False).head(2).index.tolist()
             clean_name = "-".join([str(w).lower() for w in words])
             
        # On associe l'index (str) au nom propre
        labels_map[str(i)] = clean_name
        
    print(f"   -> Mapping loaded for {len(labels_map)} topics.")

except Exception as e:
    raise ValueError(f"Error loading local data: {e}")

# B. MACRO FRED
tickers = {
    'INDPRO': 'IP',             # Industrial Production
    'PAYEMS': 'Emp',            # Employment
    'SPASTT01USM661N': 'MktRet',# S&P 500 Returns
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
    # Fallback simulation si FRED plante (juste pour que le code tourne)
    print("⚠️ FRED Error/Offline. Using dummy data for demonstration.")
    dates = theta.index
    data = pd.DataFrame(np.random.randn(len(dates), 4), index=dates, columns=['IP','Emp','MktRet','MktVol'])

# Alignment
common_idx = theta.index.intersection(data.index)
theta = theta.loc[common_idx]
data = data.loc[common_idx]

# LASSO 5 VARIABLES & OOS

def select_exact_k_variables(X, y, k=5):
    """ Uses LARS to find EXACTLY k variables. """
    alphas, active, coefs = lars_path(X.values, y.values, method='lasso')
    n_active = np.count_nonzero(coefs, axis=0)
    
    step_idx = np.where(n_active == k)[0]
    if len(step_idx) > 0: chosen_step = step_idx[0]
    else: chosen_step = np.argmin(np.abs(n_active - k))
    
    active_indices = np.where(coefs[:, chosen_step] != 0)[0]
    return X.columns[active_indices].tolist()

def run_analysis_section_4_1(target_name, y_raw, X_raw):
    print(f"\nProcessing: {target_name}...")
    
    # 1. STANDARDIZATION
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_std = pd.DataFrame(scaler_X.fit_transform(X_raw), index=X_raw.index, columns=X_raw.columns)
    y_std = pd.Series(scaler_y.fit_transform(y_raw.values.reshape(-1, 1)).flatten(), index=y_raw.index)
    
    # --------------------------------------------------------------------------
    # A. IN-SAMPLE ANALYSIS
    # --------------------------------------------------------------------------
    top_5_vars = select_exact_k_variables(X_std, y_std, k=N_VARS)
    
    X_ols = sm.add_constant(X_std[top_5_vars])
    model = sm.OLS(y_std, X_ols).fit()
    r2_in = model.rsquared
    
    # --------------------------------------------------------------------------
    # B. OUT-OF-SAMPLE ANALYSIS
    # --------------------------------------------------------------------------
    trues_oos = []
    preds_oos = []
    hist_means = []
    
    for i in range(MIN_TRAIN_SIZE, len(y_std)):
        y_train = y_std.iloc[:i]
        X_train = X_std.iloc[:i]
        
        # Test (1 point)
        y_test_pt = y_std.iloc[i]
        X_test_row = X_std.iloc[[i]]
        
        # Select & Fit on Train
        vars_t = select_exact_k_variables(X_train, y_train, k=N_VARS)
        reg = LinearRegression().fit(X_train[vars_t], y_train)
        
        # Predict
        pred = reg.predict(X_test_row[vars_t])[0]
        
        preds_oos.append(pred)
        trues_oos.append(y_test_pt)
        hist_means.append(y_train.mean())

    # Calculate OOS R2
    mse_model = np.mean((np.array(trues_oos) - np.array(preds_oos))**2)
    mse_bench = np.mean((np.array(trues_oos) - np.array(hist_means))**2)
    r2_out = 1 - (mse_model / mse_bench)
    
    # --------------------------------------------------------------------------
    # C. TABLES
    # --------------------------------------------------------------------------
    table_data = []
    for var in top_5_vars:
        # ICI LE MAPPING PREND EFFET
        clean_name = labels_map.get(var, f"Topic {var}")
        
        table_data.append({
            'Topic': clean_name,
            'Coeff.': model.params[var],
            'p-val': model.pvalues[var]
        })
    
    df_res = pd.DataFrame(table_data).sort_values(by='Coeff.', key=abs, ascending=False)
    
    print(f"\n{'-'*65}")
    print(f"TABLE : {target_name.upper()}")
    print(f"{'-'*65}")
    print(f"{'Topic':<35} | {'Coeff.':<8} | {'p-val':<8}")
    print(f"{'-'*65}")
    for _, row in df_res.iterrows():
        print(f"{row['Topic']:<35} | {row['Coeff.']:<8.2f} | {row['p-val']:<8.2f}")
    print(f"{'-'*65}")
    print(f"In-Sample R2     : {r2_in:.2f}")
    print(f"Out-of-Sample R2 : {r2_out:.2f}")
    print(f"{'-'*65}")
    
    # --------------------------------------------------------------------------
    # D. GRAPHS
    # --------------------------------------------------------------------------
    # In-Sample Fit (Full period)
    vars_final = select_exact_k_variables(X_std, y_std, k=N_VARS)
    reg_final = LinearRegression().fit(X_std[vars_final], y_std)
    preds_in_sample = reg_final.predict(X_std[vars_final])

    dates_oos = y_std.index[MIN_TRAIN_SIZE:] 
    preds_oos_aligned = pd.Series(preds_oos, index=dates_oos)

    plt.figure(figsize=(12, 5))
    plt.plot(y_std.index, y_std, 'k-', lw=1, alpha=0.3, label='Actual data')
    plt.plot(y_std.index, preds_in_sample, 'b--', lw=1, alpha=0.8, label='In-Sample Fit')
    plt.plot(dates_oos, preds_oos_aligned, 'r-', lw=1.5, label='Out-of-Sample Forecast')
    plt.axvline(y_std.index[MIN_TRAIN_SIZE], color='gray', linestyle=':', label='OOS Start')
    plt.title(f"{target_name}: Topic Model Performance")
    plt.legend()
    plt.tight_layout()
    plt.show()

# EXECUTION

# 1. Industrial Production
run_analysis_section_4_1("Industrial Production Growth", data['IP'], theta)

# 2. Employment
run_analysis_section_4_1("Employment Growth", data['Emp'], theta)

# 3. Market Returns
run_analysis_section_4_1("Market Returns", data['MktRet'], theta)

# 4. Market Volatility
run_analysis_section_4_1("Market Volatility", data['MktVol'], theta)