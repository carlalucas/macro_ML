import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import lars_path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
import io
import requests
import zipfile

# ==============================================================================
# 0. CONFIGURATION
# ==============================================================================
print(">>> INITIALISATION (SECTION 4.2.2 - LOG VOLATILITY FIX)...")
warnings.filterwarnings("ignore")

START_DATE = '1984-01-01'
END_DATE = '2017-06-01'

# Mapping Cible (avec variantes orthographiques pour robustesse)
TARGET_SECTORS = {
    'Automotive': ['Autos', 'Auto'],
    'Banking': ['Banks', 'Bank'],
    'Pharmaceuticals': ['Drugs', 'Drug'],
    'Computer Hardware': ['Hrdwr', 'Hardw', 'Comps'],
    'Oil and Gas': ['Oil', 'Oil  '],
    'Tobacco': ['Smoke']
}

def get_stars(p):
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.1: return "*"
    return ""

def select_exact_k_lars(X, y, k=5):
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_s = scaler_X.fit_transform(X)
    y_s = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()
    alphas, active, coefs = lars_path(X_s, y_s, method='lasso')
    n_active = np.count_nonzero(coefs, axis=0)
    
    # Selection step k
    step_idx = np.where(n_active == k)[0]
    if len(step_idx) > 0: chosen_step = step_idx[0]
    else: chosen_step = np.argmin(np.abs(n_active - k))
    
    active_indices = np.where(coefs[:, chosen_step] != 0)[0]
    return active_indices

# ==============================================================================
# 1. TÉLÉCHARGEMENT KEN FRENCH
# ==============================================================================
print(">>> 1. CHARGEMENT KEN FRENCH...")

def get_ken_french_49_daily():
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/49_Industry_Portfolios_daily_CSV.zip"
    try:
        r = requests.get(url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        csv_filename = z.namelist()[0]
        with z.open(csv_filename) as f: lines = f.readlines()
        
        start_row = 0
        for i, line in enumerate(lines):
            if b"Average Value Weighted Returns" in line:
                start_row = i + 1
                break
                
        df = pd.read_csv(z.open(csv_filename), skiprows=start_row, header=0)
        df = df.rename(columns={df.columns[0]: 'Date'})
        df['Date'] = pd.to_numeric(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df['Date'] = pd.to_datetime(df['Date'].astype(int).astype(str), format='%Y%m%d')
        df = df.set_index('Date')
        df.columns = df.columns.str.strip()
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        return df
    except Exception as e:
        raise ValueError(f"Erreur Ken French: {e}")

kf_daily = get_ken_french_49_daily()

# ==============================================================================
# 2. TRAITEMENT FINANCIER (LOG-VOLATILITY + PCA)
# ==============================================================================
print(">>> 2. CALCUL LOG-VOLATILITÉ & ORTHOGONALISATION...")

# 1. Volatilité Mensuelle
vol_monthly = kf_daily.resample('MS').std()

# 2. Filtre Date (Strict 1984-2017)
mask = (vol_monthly.index >= START_DATE) & (vol_monthly.index <= END_DATE)
vol_monthly = vol_monthly.loc[mask]

# 3. TRANSFORMATION LOG (La clé pour supprimer l'effet multiplicatif du marché)
# On ajoute une minuscule constante pour éviter log(0) si jamais une volatilité est nulle (rare)
vol_log = np.log(vol_monthly + 1e-6)

# 4. PCA sur les LOG-VOLATILITÉS
pca = PCA(n_components=1)
common_factor = pca.fit_transform(StandardScaler().fit_transform(vol_log))

# 5. Orthogonalisation (sur les logs)
vol_ortho_log = pd.DataFrame(index=vol_log.index, columns=vol_log.columns)
for col in vol_log.columns:
    y_reg = vol_log[col]
    X_reg = sm.add_constant(common_factor)
    vol_ortho_log[col] = sm.OLS(y_reg, X_reg).fit().resid

# 6. Innovations AR(1) sur les résidus orthogonaux
industry_innov = pd.DataFrame(index=vol_ortho_log.index)
for col in vol_ortho_log.columns:
    series = vol_ortho_log[col].dropna()
    y_ar = series.iloc[1:]
    X_ar = sm.add_constant(series.shift(1).iloc[1:])
    industry_innov[col] = sm.OLS(y_ar, X_ar).fit().resid
industry_innov = industry_innov.dropna()

print(f"   -> Traitement Log-Vol terminé sur {len(industry_innov)} mois.")

# ==============================================================================
# 3. CHARGEMENT TOPICS (AVEC NOMS PROPRES)
# ==============================================================================
print(">>> 3. CHARGEMENT TOPICS...")

try:
    theta = pd.read_csv('data/theta_monthly.csv', sep=None, engine='python')
    date_col = next((c for c in theta.columns if 'date' in str(c).lower()), theta.columns[0])
    theta[date_col] = pd.to_datetime(theta[date_col])
    theta.set_index(date_col, inplace=True)
    theta = theta.select_dtypes(include=[np.number])
    theta.columns = [str(i) for i in range(theta.shape[1])]
    
    phi = pd.read_csv('data/phi_scaled.csv', sep=None, engine='python', index_col=0)
    
    # Mapping Noms Propres
    labels_map = {}
    for i in range(min(theta.shape[1], phi.shape[1])):
        clean_name = phi.columns[i].strip()
        if str(clean_name).isdigit():
             words = phi.iloc[:, i].sort_values(ascending=False).head(2).index.tolist()
             clean_name = "-".join([str(w).lower() for w in words])
        labels_map[str(i)] = clean_name

    # Innovations AR(1) sur Topics
    theta_innov = pd.DataFrame(index=theta.index)
    for col in theta.columns:
        s = theta[col]
        y_ar = s.iloc[1:]
        X_ar = sm.add_constant(s.shift(1).iloc[1:])
        try:
            theta_innov[col] = sm.OLS(y_ar, X_ar).fit().resid
        except: pass
    theta_innov = theta_innov.dropna()

except Exception as e:
    print(f"❌ Erreur Topics: {e}")
    exit()

# Alignement
common = theta_innov.index.intersection(industry_innov.index)
X_topics = theta_innov.loc[common]
Y_inds = industry_innov.loc[common]

# ==============================================================================
# 4. ANALYSE ET MAPPING FINAL
# ==============================================================================
print("\n>>> 4. RÉSULTATS (TABLE 4 - LOG VOLATILITY)...")

# Mapping automatique
FINAL_MAP = {}
available_cols = set(Y_inds.columns)
for paper_name, candidates in TARGET_SECTORS.items():
    for cand in candidates:
        if cand in available_cols:
            FINAL_MAP[cand] = paper_name
            break

def analyze_industry(kf_code, paper_name):
    y = Y_inds[kf_code]
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_s = pd.DataFrame(scaler_X.fit_transform(X_topics), index=X_topics.index, columns=X_topics.columns)
    y_s = pd.Series(scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten(), index=y.index)
    
    top_indices = select_exact_k_lars(X_s, y_s, k=5)
    selected_vars = X_s.columns[top_indices]
    
    model = sm.OLS(y_s, sm.add_constant(X_s[selected_vars])).fit()
    r2 = model.rsquared
    
    print(f"\n--- {paper_name.upper()} ({kf_code}) | R2 = {r2:.2f} ---")
    
    res = pd.DataFrame({
        'Topic': selected_vars,
        'Coeff': model.params[selected_vars],
        'Pval': model.pvalues[selected_vars]
    }).sort_values(by='Coeff', key=abs, ascending=False)
    
    for _, row in res.iterrows():
        t_name = labels_map.get(row['Topic'], f"Topic {row['Topic']}")
        star = get_stars(row['Pval'])
        print(f"{t_name:<30} | {row['Coeff']:>6.2f} {star}")

for kf_code, paper_name in FINAL_MAP.items():
    analyze_industry(kf_code, paper_name)