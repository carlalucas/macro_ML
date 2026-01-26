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

# Période cible du papier
TARGET_START = '1984-01-01'
TARGET_END = '2017-06-01'

# 1. CFNAI 
CFNAI_IDS = {
    'CFNAI': 'CFNAI', # L'indice global suffit souvent et est plus long
    'PANDI': 'Prod_Inc',   
    'EUANDH': 'Emp_Hrs',   
    'CANDH': 'Cons_Hous',  
    'SOANDI': 'Sales_Ord'  
}

# 2. FRED-MD (Auswahl plus robuste)
# On ne garde que ceux qui remontent surement à 1984
FRED_MD_IDS = [
    'INDPRO', 'PAYEMS', 'UNRATE', 'HOUST', 'PPIACO', 'PCEPI', 'M2SL', 'FEDFUNDS'
]

def select_exact_k_lars(X, y, k=5):
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

# ==============================================================================
# 1. DOWNLOAD & DIAGNOSTIC
# ==============================================================================
print(">>> 1. LOADING DATA & DIAGNOSTIC...")

# A. Topics
try:
    theta = pd.read_csv('data/theta_monthly.csv', sep=None, engine='python')
    date_col = next((c for c in theta.columns if 'date' in str(c).lower()), theta.columns[0])
    theta[date_col] = pd.to_datetime(theta[date_col])
    theta.set_index(date_col, inplace=True)
    theta = theta.select_dtypes(include=[np.number])
    theta.columns = [str(i) for i in range(theta.shape[1])]
    
    phi = pd.read_csv('data/phi_scaled.csv', sep=None, engine='python', index_col=0)
    
    # Mapping
    labels_map = {}
    for i in range(min(theta.shape[1], phi.shape[1])):
        clean_name = phi.columns[i].strip()
        if str(clean_name).isdigit():
             words = phi.iloc[:, i].sort_values(ascending=False).head(2).index.tolist()
             clean_name = "-".join([str(w).lower() for w in words])
        labels_map[str(i)] = clean_name
        
    print(f"   [THETA] Range: {theta.index.min().date()} -> {theta.index.max().date()} (Count: {len(theta)})")

except Exception as e:
    print(f"Error Local Files: {e}")
    exit()

# B. Target (S&P 500)
try:
    sp500 = web.DataReader('SPASTT01USM661N', 'fred', start='1950-01-01')
    mkt_ret = np.log(sp500).diff() * 100
    mkt_ret.columns = ['MktRet']
    mkt_ret = mkt_ret.dropna()
    print(f"   [TARGET] Range: {mkt_ret.index.min().date()} -> {mkt_ret.index.max().date()}")
except:
    print("Error Target.")
    exit()

# C. Benchmarks (CFNAI)
print("   -> Downloading CFNAI...")
try:
    # On télécharge large
    cfnai_raw = web.DataReader(list(CFNAI_IDS.keys()), 'fred', start='1960-01-01')
    cfnai_raw = cfnai_raw.rename(columns=CFNAI_IDS).resample('MS').mean()
    
    # Check colonnes vides
    cfnai_raw = cfnai_raw.dropna(axis=1, how='all')
    cfnai_innov = get_ar1_innovations(cfnai_raw)
    
    print(f"   [CFNAI] Range: {cfnai_innov.index.min().date()} -> {cfnai_innov.index.max().date()}")
except:
    print("   [CFNAI] Failed. Using Empty.")
    cfnai_innov = pd.DataFrame()

# D. Benchmarks (FRED-MD)
print("   -> Downloading FRED-MD (Proxy)...")
try:
    fred_list = []
    for code in FRED_MD_IDS:
        try:
            s = web.DataReader(code, 'fred', start='1960-01-01').resample('MS').mean()
            fred_list.append(s)
        except: pass
    
    fred_raw = pd.concat(fred_list, axis=1)
    
    # Transform (Log Diff simple pour robustesse)
    fred_stat = pd.DataFrame(index=fred_raw.index)
    for c in fred_raw.columns:
        if c in ['UNRATE', 'FEDFUNDS']: fred_stat[c] = fred_raw[c].diff()
        else: fred_stat[c] = np.log(fred_raw[c]).diff() * 100
        
    fred_innov = get_ar1_innovations(fred_stat)
    print(f"   [FRED] Range: {fred_innov.index.min().date()} -> {fred_innov.index.max().date()}")
except:
    print("   [FRED] Failed. Using Empty.")
    fred_innov = pd.DataFrame()

# ==============================================================================
# 2. ALIGNEMENT INTELLIGENT
# ==============================================================================

# On aligne d'abord Theta et Market (Les plus importants)
common = mkt_ret.index.intersection(theta.index)

# On filtre sur la période du papier
common = common[(common >= TARGET_START) & (common <= TARGET_END)]

# On reindexe les benchmarks sur cette période (avec fillna si besoin pour ne pas tout perdre)
# Astuce: si une macro manque un mois, on ne veut pas jeter tout le dataset.
# Mais pour la régression, on doit dropna.

# Intersection finale stricte
final_idx = common.intersection(cfnai_innov.index).intersection(fred_innov.index)

print(f"\n>>> PERIODE COMMUNE FINALE : {len(final_idx)} mois")
print(f"    {final_idx.min().date()} -> {final_idx.max().date()}")

if len(final_idx) < 200:
    print("⚠️ ATTENTION : Période trop courte ! Les résultats 'Recession' ne sortiront pas.")
    print("   Causes possibles :")
    if len(cfnai_innov) < len(common): print("   - CFNAI est trop court.")
    if len(fred_innov) < len(common): print("   - FRED est trop court.")

y = mkt_ret.loc[final_idx]
X_topics = theta.loc[final_idx]
X_cfnai = cfnai_innov.loc[final_idx]
X_fred = fred_innov.loc[final_idx]

# ==============================================================================
# 3. ANALYSIS
# ==============================================================================

def run_standardized_analysis(y_target, X_news, X_bench, bench_name):
    if X_bench.empty: return

    # Standardisation
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