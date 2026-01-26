# src/utils.py
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import lars_path
from sklearn.preprocessing import StandardScaler

def get_stars(p):
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.1: return "*"
    return ""

def select_exact_k_lars(X, y, k=5):
    """
    Sélectionne exactement k variables via l'algorithme LARS.
    Entrée : DataFrame X, Series y
    Sortie : Liste des indices (noms de colonnes) sélectionnés.
    """
    # Standardisation critique pour le Lasso
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_s = scaler_X.fit_transform(X)
    y_s = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()
    
    # Chemin de régularisation LARS
    alphas, active, coefs = lars_path(X_s, y_s, method='lasso')
    
    # Trouver l'étape avec k variables actives
    n_active = np.count_nonzero(coefs, axis=0)
    
    # On cherche l'index où n_active == k
    step_idx = np.where(n_active == k)[0]
    
    if len(step_idx) > 0:
        chosen_step = step_idx[0]
    else:
        # Fallback : on prend le plus proche (rare)
        chosen_step = np.argmin(np.abs(n_active - k))
    
    # Indices des variables non nulles à cette étape
    active_indices = np.where(coefs[:, chosen_step] != 0)[0]
    
    return X.columns[active_indices]

def get_ar1_innovations(df):
    """
    Calcule les résidus d'un processus AR(1) pour chaque colonne.
    Utilisé pour stationnariser les séries (CFNAI, Volatilité).
    """
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
        except:
            pass # Skip si colinéarité
            
    return innovations.dropna()