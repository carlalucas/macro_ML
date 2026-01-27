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
    Selects k variables via LARS.
    Assumes X and y are already standardized.
    Returns the list of NAMES of the selected columns.
    """
    # Convert to numpy for scikit-learn
    X_val = X.values if hasattr(X, 'values') else X
    y_val = y.values if hasattr(y, 'values') else y.flatten()

    # LARS algorithm
    alphas, active, coefs = lars_path(X_val, y_val, method='lasso')
    
    # Find the step with k non-zero variables
    n_active = np.count_nonzero(coefs, axis=0)
    
    # We look for the exact index, otherwise the closest
    step_idx = np.where(n_active == k)[0]
    if len(step_idx) > 0:
        chosen_step = step_idx[0]
    else:
        chosen_step = np.argmin(np.abs(n_active - k))
    
    # Indices of active variables
    active_indices = np.where(coefs[:, chosen_step] != 0)[0]
    
    # Returns the NAMES of the columns if X is a DataFrame 
    if hasattr(X, 'columns'):
        return X.columns[active_indices].tolist()
    
    return active_indices

def get_ar1_innovations(df):
    """
    Calculates the residuals of an AR(1) process for each column.
    Used to stationarize the series (CFNAI, Volatility).
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
            pass 
            
    return innovations.dropna()