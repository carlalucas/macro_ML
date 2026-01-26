# src/data_loader.py
import pandas as pd
import pandas_datareader.data as web
import numpy as np
import requests
import zipfile
import io
import os

def load_topics(data_dir='data'):
    """
    Charge Theta et Phi, et génère le mapping intelligent des noms.
    """
    try:
        theta = pd.read_csv(f'{data_dir}/theta_monthly.csv', sep=None, engine='python')
        # Détection date
        date_col = next((c for c in theta.columns if 'date' in str(c).lower()), theta.columns[0])
        theta[date_col] = pd.to_datetime(theta[date_col])
        theta.set_index(date_col, inplace=True)
        theta = theta.select_dtypes(include=[np.number])
        theta.columns = [str(i) for i in range(theta.shape[1])]
        
        phi = pd.read_csv(f'{data_dir}/phi_scaled.csv', sep=None, engine='python', index_col=0)
        
        # --- MAPPING INTELLIGENT ---
        labels_map = {}
        for i in range(min(theta.shape[1], phi.shape[1])):
            clean_name = phi.columns[i].strip()
            # Sécurité si le nom est un chiffre
            if str(clean_name).isdigit():
                 words = phi.iloc[:, i].sort_values(ascending=False).head(2).index.tolist()
                 clean_name = "-".join([str(w).lower() for w in words])
            labels_map[str(i)] = clean_name
            
        return theta, labels_map
        
    except Exception as e:
        print(f"❌ Erreur chargement Topics: {e}")
        return None, None

def get_ken_french_49_daily():
    """Télécharge les 49 Industries (Daily) depuis le site de Ken French."""
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/49_Industry_Portfolios_daily_CSV.zip"
    try:
        r = requests.get(url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        csv_filename = z.namelist()[0]
        
        with z.open(csv_filename) as f:
            lines = f.readlines()
        
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
        df.columns = df.columns.str.strip() # Nettoyage essentiel
        return df.apply(pd.to_numeric, errors='coerce').dropna()
    except Exception as e:
        print(f"❌ Erreur Ken French: {e}")
        return None

def transform_fred_md(df):
    """Log-Diff pour stationnariser les données Macro FRED."""
    df_clean = pd.DataFrame(index=df.index)
    for col in df.columns:
        s = df[col]
        if col in ['UNRATE', 'UEMPMEAN', 'FEDFUNDS']: 
            df_clean[col] = s.diff()
        else:
            try:
                s_safe = s.replace(0, np.nan).dropna()
                df_clean[col] = np.log(s_safe).diff() * 100
            except:
                df_clean[col] = s.diff()
    return df_clean.dropna()