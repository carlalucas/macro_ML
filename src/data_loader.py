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
    Load Theta and Phi, and generate the intelligent mapping of names.
    """
    try:
        theta = pd.read_csv(f'{data_dir}/theta_monthly.csv', sep=None, engine='python')
        # Date detection
        date_col = next((c for c in theta.columns if 'date' in str(c).lower()), theta.columns[0])
        theta[date_col] = pd.to_datetime(theta[date_col])
        theta.set_index(date_col, inplace=True)
        theta = theta.select_dtypes(include=[np.number])
        theta.columns = [str(i) for i in range(theta.shape[1])]
        
        phi = pd.read_csv(f'{data_dir}/phi_scaled.csv', sep=None, engine='python', index_col=0)
        
        # --- INTELLIGENT MAPPING ---
        labels_map = {}
        for i in range(min(theta.shape[1], phi.shape[1])):
            clean_name = phi.columns[i].strip()
            # Safety if the name is a number
            if str(clean_name).isdigit():
                 words = phi.iloc[:, i].sort_values(ascending=False).head(2).index.tolist()
                 clean_name = "-".join([str(w).lower() for w in words])
            labels_map[str(i)] = clean_name
            
        return theta, labels_map
        
    except Exception as e:
        print(f"Error loading Topics: {e}")
        return None, None
