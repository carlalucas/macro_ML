import pandas as pd
import pandas_datareader.data as web
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import lars_path
from sklearn.preprocessing import StandardScaler
import warnings

# ==============================================================================
# 0. CONFIGURATION
# ==============================================================================
print(">>> INITIALIZATION...")
warnings.filterwarnings("ignore")

# END DATE OF PAPER
END_DATE_PAPER = '2017-06-01'

# 1. CFNAI 
CFNAI_IDS = {
    'PANDI': 'Prod_Inc',   
    'EUANDH': 'Emp_Hrs',   
    'CANDH': 'Cons_Hous',  
    'SOANDI': 'Sales_Ord'  
}

# 2. FRED-MD 
FRED_MD_IDS = [
    'IPNCONGD', 
    'INDPRO', 'IPFINAL', 'IPMAT', 'IPBUSEQ', 
    'PAYEMS', 'USGOOD', 'MANEMP', 'SRVPRD', 'USCONS', 'UNRATE', 'UEMPMEAN',
    'HOUST', 'PERMIT', 'HOUST5F', 'PERMIT5F',
    'PCEPI', 'CPIAUCSL', 'CPILFESL', 'PPIACO',
    'RETAIL', 'DPCERA3M086SBEA', # Real Personal Consumption
    'CMRMTSPL', # Real Manuf. & Trade Sales
    'TTLCONS'   # Construction Spending
]

def select_exact_k_lars(X, y, k=5):
    """Select k variables via LARS"""
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_s = scaler_X.fit_transform(X)
    y_s = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()
    
    alphas, active, coefs = lars_path(X_s, y_s, method='lasso')
    
    n_active = np.count_nonzero(coefs, axis=0)
    step_idx = np.where(n_active == k)[0]
    
    if len(step_idx) > 0: chosen_step = step_idx[0]
    else: chosen_step = np.argmin(np.abs(n_active - k))
    
    active_indices = np.where(coefs[:, chosen_step] != 0)[0]
    return active_indices

def get_ar1_innovations(df):
    """Compute AR(1) residuals"""
    if df.empty: return pd.DataFrame()
    innovations = pd.DataFrame(index=df.index)
    for col in df.columns:
        series = df[col].dropna()
        if len(series) < 24: continue
        y = series.iloc[1:]
        X = sm.add_constant(series.shift(1).iloc[1:])
        try:
            model = sm.OLS(y, X).fit()
            innovations[col] = model.resid
        except: pass
    return innovations.dropna()

def transform_fred_md(df):
    """Log-Diff to stationarize"""
    if df.empty: return pd.DataFrame()
    df_clean = pd.DataFrame(index=df.index)
    for col in df.columns:
        s = df[col]
        # Rates -> Simple Diff
        if col in ['UNRATE', 'UEMPMEAN']: 
            df_clean[col] = s.diff()
        # Levels -> Log Diff * 100
        else:
            try:
                s_safe = s.replace(0, np.nan).dropna()
                df_clean[col] = np.log(s_safe).diff() * 100
            except:
                df_clean[col] = s.diff()
    return df_clean.dropna()

# ==============================================================================
# 1. DOWNLOAD
# ==============================================================================
print(">>> 1. LOADING DATA...")

# A. Topics
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
    print("Error: Local files.")
    exit()

# B. Target (S&P 500 Returns)
try:
    sp500 = web.DataReader('SPASTT01USM661N', 'fred', start='1980-01-01')
    mkt_ret = np.log(sp500).diff() * 100
    mkt_ret.columns = ['MktRet']
    mkt_ret = mkt_ret.dropna()
except:
    print("Error: Target (SP500).")
    exit()

# C. Benchmarks
print("   -> CFNAI & FRED-MD...")
try:
    # CFNAI
    cfnai_raw = web.DataReader(list(CFNAI_IDS.keys()), 'fred', start='1980-01-01')
    cfnai_raw = cfnai_raw.rename(columns=CFNAI_IDS).resample('MS').mean()
    cfnai_innov = get_ar1_innovations(cfnai_raw)
    
    # FRED-MD (Hard Macro Only)
    fred_data = []
    for code in FRED_MD_IDS:
        try:
            s = web.DataReader(code, 'fred', start='1980-01-01').resample('MS').mean()
            fred_data.append(s)
        except: continue
    
    if fred_data:
        fred_raw = pd.concat(fred_data, axis=1)
        fred_stat = transform_fred_md(fred_raw)
        fred_innov = get_ar1_innovations(fred_stat)
    else:
        fred_innov = pd.DataFrame()
except:
    cfnai_innov = pd.DataFrame()
    fred_innov = pd.DataFrame()

# Global Alignment
common = mkt_ret.index.intersection(theta.index).intersection(cfnai_innov.index).intersection(fred_innov.index)

# --- CRITICAL: EXACT PAPER DATE CUTOFF ---
common = common[common <= END_DATE_PAPER]
# --------------------------------------------

y = mkt_ret.loc[common]
X_topics = theta.loc[common]
X_cfnai = cfnai_innov.loc[common]
X_fred = fred_innov.loc[common]

print(f"   -> Period: {len(common)} months ({common.min().date()} - {common.max().date()})")

# ==============================================================================
# 2. ANALYSIS (STANDARDIZED COEFFICIENTS)
# ==============================================================================

def run_standardized_analysis(y_target, X_news, X_bench, bench_name):
    if X_bench.empty: return

    # Standardization
    scaler_y = StandardScaler()
    scaler_news = StandardScaler()
    scaler_bench = StandardScaler()
    
    y_std = pd.Series(scaler_y.fit_transform(y_target.values.reshape(-1, 1)).flatten(), index=y_target.index)
    X_news_std = pd.DataFrame(scaler_news.fit_transform(X_news), index=X_news.index, columns=X_news.columns)
    X_bench_std = pd.DataFrame(scaler_bench.fit_transform(X_bench), index=X_bench.index, columns=X_bench.columns)

    # 1. Benchmark R2
    if X_bench.shape[1] <= 5:
        model_b = sm.OLS(y_std, sm.add_constant(X_bench_std)).fit()
    else:
        idx_b = select_exact_k_lars(X_bench_std, y_std, k=5)
        vars_b = X_bench_std.columns[idx_b]
        model_b = sm.OLS(y_std, sm.add_constant(X_bench_std[vars_b])).fit()
    
    bench_r2 = model_b.rsquared

    # 2. Full Model
    X_pool = pd.concat([X_news_std, X_bench_std], axis=1)
    top_indices = select_exact_k_lars(X_pool, y_std, k=5)
    selected_vars = X_pool.columns[top_indices]
    
    full_model = sm.OLS(y_std, sm.add_constant(X_pool[selected_vars])).fit()
    full_r2 = full_model.rsquared
    
    # --- DISPLAY ---
    print(f"\n{'-'*65}")
    print(f"TABLE 2: TOPIC MODEL & {bench_name}")
    print(f"{'-'*65}")
    print(f"{'Topic / Variable':<35} | {'Coeff.':<8} | {'p-val':<8}")
    print(f"{'-'*65}")
    
    res_df = pd.DataFrame({
        'Name': selected_vars,
        'Coeff': full_model.params[selected_vars],
        'Pval': full_model.pvalues[selected_vars]
    }).sort_values(by='Coeff', key=abs, ascending=False)
    
    for _, row in res_df.iterrows():
        var_name = str(row['Name'])
        # Name
        if var_name in labels_map:
            display_name = labels_map[var_name]
        elif var_name in CFNAI_IDS.values():
            display_name = f"[CFNAI] {var_name}"
        else:
            display_name = f"[MACRO] {var_name}"
            
        print(f"{display_name:<35} | {row['Coeff']:<8.2f} | {row['Pval']:<8.2f}")
        
    print(f"{'-'*65}")
    print(f"Full R2      : {full_r2:.2f}")
    print(f"Benchmark R2 : {bench_r2:.2f}")
    print(f"{'-'*65}")

print("\n>>> 3. RESULTS...")
run_standardized_analysis(y, X_topics, X_cfnai, "CFNAI")
run_standardized_analysis(y, X_topics, X_fred, "FRED-MD")