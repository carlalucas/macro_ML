import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import lars_path
from sklearn.preprocessing import StandardScaler
import warnings
import os

# ==============================================================================
# 0. CONFIGURATION
# ==============================================================================
print("INITIALIZATION...")
warnings.filterwarnings("ignore")

# End date of the paper (June 2017)
END_DATE_PAPER = '2017-06-01'

def get_stars(p):
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.1: return "*"
    return ""

def select_exact_k_lars(X, y, k=5):
    """Select exactly k variables via LARS"""
    # Critical standardization
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_s = scaler_X.fit_transform(X)
    y_s = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()
    
    # LARS Path
    alphas, active, coefs = lars_path(X_s, y_s, method='lasso')
    
    # Find the k step
    n_active = np.count_nonzero(coefs, axis=0)
    step_idx = np.where(n_active == k)[0]
    
    if len(step_idx) > 0: chosen_step = step_idx[0]
    else: chosen_step = np.argmin(np.abs(n_active - k))
    
    active_indices = np.where(coefs[:, chosen_step] != 0)[0]
    return active_indices

# ==============================================================================
# 1. ROBUST LOADING
# ==============================================================================
print("1. LOADING DATA...")

# A. Topics
try:
    theta = pd.read_csv('data/theta_monthly.csv', sep=None, engine='python')
    # Intelligent date column detection
    date_col = next((c for c in theta.columns if 'date' in str(c).lower()), theta.columns[0])
    theta[date_col] = pd.to_datetime(theta[date_col])
    theta.set_index(date_col, inplace=True)
    # Keep only numeric columns (Topics)
    theta = theta.select_dtypes(include=[np.number])
    theta.columns = [str(i) for i in range(theta.shape[1])]
    
    phi = pd.read_csv('data/phi_scaled.csv', sep=None, engine='python', index_col=0)
    labels_map = {}
    for i in range(min(theta.shape[1], phi.shape[1])):
        words = phi.iloc[:, i].sort_values(ascending=False).head(2).index.tolist()
        labels_map[str(i)] = "-".join([str(w).lower() for w in words])
except:
    print("Error loading local files (theta/phi). Check 'data/'.")
    exit()

# B. EPU Categorical (Local)
file_path = 'data/Categorical_EPU_Data.xlsx'

if not os.path.exists(file_path):
    print(f"ERROR: File '{file_path}' not found.")
    print("   -> Download here: https://www.policyuncertainty.com/media/Categorical_EPU_Data.xlsx")
    print("   -> Place it in the data/ folder")
    exit()

try:
    print(f"   -> Reading {file_path}...")
    epu_raw = pd.read_excel(file_path)
    
    # --- CRITICAL FIX: DATE CLEANING ---
    # 1. Force to numeric (handles weird cases)
    epu_raw['Year'] = pd.to_numeric(epu_raw['Year'], errors='coerce')
    epu_raw['Month'] = pd.to_numeric(epu_raw['Month'], errors='coerce')
    
    # 2. Remove empty rows
    epu_raw = epu_raw.dropna(subset=['Year', 'Month'])
    
    # 3. Convert to pure INTEGER (1985 instead of 1985.0)
    epu_raw['Year'] = epu_raw['Year'].astype(int).astype(str)
    epu_raw['Month'] = epu_raw['Month'].astype(int).astype(str)
    
    # 4. Create clean Date
    epu_raw['Date'] = pd.to_datetime(epu_raw['Year'] + '-' + epu_raw['Month'] + '-01')
    epu_raw = epu_raw.set_index('Date').sort_index()
    
    # Select data columns (all except Year/Month)
    cols_to_keep = [c for c in epu_raw.columns if c not in ['Year', 'Month']]
    epu_cats = epu_raw[cols_to_keep]
    
    # Log Transformation (As suggested by EPU literature to normalize peaks)
    # Add 1 to avoid log(0)
    epu_cats_log = np.log(epu_cats + 1)
    
    print(f"      OK: {epu_cats.shape[1]} EPU indices loaded.")

except Exception as e:
    print(f"Error reading EPU: {e}")
    exit()

# Alignment
common = theta.index.intersection(epu_cats_log.index)
common = common[common <= END_DATE_PAPER] # Cut at 2017 as in the paper

X_topics = theta.loc[common]
Y_epu = epu_cats_log.loc[common]

print(f"   -> Common period: {len(common)} months ({common.min().date()} - {common.max().date()})")

# ==============================================================================
# 2. TABLE 5 ANALYSIS
# ==============================================================================
print("\n2. RESULTS...")

# Mapping Excel file names to paper names
# Excel file sometimes has different names (e.g., "National Security" vs "National Security Policy")
target_map = {
    'Entitlement Programs': 'Entitlement Programs', 
    'Financial Regulation': 'Financial Regulation',
    'Fiscal Policy': 'Fiscal Policy',
    'Government Spending': 'Government Spending',
    'Health Care': 'Health Care',
    'Monetary Policy': 'Monetary Policy',
    'National Security': 'National Security',
    'Regulation': 'Regulation',
    'Sovereign Debt': 'Sovereign Debt',
    'Taxes': 'Taxes',
    'Trade Policy': 'Trade Policy',
    'Economic Policy Uncertainty': 'Broad EPU'
}

def analyze_epu_category(cat_name, y_series):
    # Standardization
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_s = pd.DataFrame(scaler_X.fit_transform(X_topics), index=X_topics.index, columns=X_topics.columns)
    y_s = pd.Series(scaler_y.fit_transform(y_series.values.reshape(-1, 1)).flatten(), index=y_series.index)
    
    # Lasso Selection (5 vars)
    top_indices = select_exact_k_lars(X_s, y_s, k=5)
    selected_vars = X_s.columns[top_indices]
    
    # Final OLS
    model = sm.OLS(y_s, sm.add_constant(X_s[selected_vars])).fit()
    r2 = model.rsquared
    
    print(f"\n--- {cat_name.upper()} (R2 = {r2:.2f}) ---")
    
    res = pd.DataFrame({
        'Topic': selected_vars,
        'Coeff': model.params[selected_vars],
        'Pval': model.pvalues[selected_vars]
    }).sort_values(by='Coeff', key=abs, ascending=False)
    
    for _, row in res.iterrows():
        t_name = labels_map.get(row['Topic'], f"Topic {row['Topic']}")
        star = get_stars(row['Pval'])
        print(f"{t_name:<35} | {row['Coeff']:>6.2f} {star}")

# Execution for each found category
found = 0
for col in Y_epu.columns:
    # Fuzzy search to match Excel columns to targets
    for key, display_name in target_map.items():
        if key.lower() in col.lower():
            analyze_epu_category(display_name, Y_epu[col])
            found += 1
            break

if found == 0:
    print("Warning: Unrecognized column names. Displaying all:")
    for col in Y_epu.columns:
        analyze_epu_category(col, Y_epu[col])