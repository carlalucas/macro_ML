# src/data_loader.py
import pandas as pd
import numpy as np
import requests
import io
from pathlib import Path

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
    
def fred_download_csv(series_id: str, start: str, end: str) -> pd.DataFrame:
    """
    Download a FRED series as CSV without API key using fredgraph endpoint.
    Returns DataFrame with DatetimeIndex and one column = series_id, float.
    """
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    params = {"id": series_id, "cosd": start, "coed": end}
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(pd.compat.StringIO(r.text)) if hasattr(pd.compat, "StringIO") else pd.read_csv(pd.io.common.StringIO(r.text))
    df.columns = ["date", series_id]
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
    return df

def stooq_download_spx_daily(start: str, end: str) -> pd.DataFrame:
    """
    Download S&P 500 index from Stooq as daily OHLCV.
    Symbol: ^SPX
    Returns DataFrame indexed by date with column 'SP500' (Close).
    Source: https://stooq.com/q/d/?s=%5Espx  (CSV download available)  :contentReference[oaicite:2]{index=2}
    """
    url = "https://stooq.com/q/d/l/"
    params = {"s": "^spx", "i": "d"}  # daily
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()

    df = pd.read_csv(io.StringIO(r.text))
    # Stooq columns typically: Date, Open, High, Low, Close, Volume
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()

    # Keep Close as SP500 level
    df = df.rename(columns={"Close": "SP500"})[["SP500"]]
    df["SP500"] = pd.to_numeric(df["SP500"], errors="coerce")

    # Filter sample
    df = df.loc[pd.to_datetime(start):pd.to_datetime(end)]
    return df

# Non-monthly data processing
def to_monthly(df: pd.DataFrame, how: str = "mean") -> pd.DataFrame:
    """
    Convert daily (or higher frequency) series to monthly.
    how: "mean" or "last"
    """
    if how == "mean":
        return df.resample("MS").mean()
    if how == "last":
        return df.resample("MS").last()
    raise ValueError("how must be 'mean' or 'last'")

# Transformations
def safe_log(x: pd.Series) -> pd.Series:
    x = x.replace(0, np.nan)
    return np.log(x)

def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=0)

# Loaders of topic model data
def load_theta(path: Path) -> pd.DataFrame:
    """
    Load theta_monthly.csv: monthly topic attention. 
    Transform date column to datetime index.
    Checked: start of the month index.
    """
    df = pd.read_csv(path)
    date_col = "date"
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    # Ensure month-start index
    df.index = df.index.to_period("M").to_timestamp(how="start")
    return df

def load_epu_xlsx(path: Path) -> pd.Series:
    """
    Load monthly US EPU from the Excel file (sheet: 'Main News Index') where:
      col1 = year
      col2 = month (1-12)
      col3 = EPU index
    Builds a month-start DatetimeIndex (YYYY-MM-01) and returns a Series named 'EPU'.
    """
    df = pd.read_excel(path, sheet_name="Main News Index", engine="openpyxl")

    # Take first 3 columns robustly (year, month, EPU)
    df = df.iloc[:, :3].copy()
    df.columns = ["year", "month", "EPU"]

    # Coerce types
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["month"] = pd.to_numeric(df["month"], errors="coerce")
    df["EPU"] = pd.to_numeric(df["EPU"], errors="coerce")

    # Drop invalid rows
    df = df.dropna(subset=["year", "month", "EPU"])
    df = df[(df["month"] >= 1) & (df["month"] <= 12)]

    # Build month-start dates
    df["date"] = pd.to_datetime(
        dict(year=df["year"].astype(int), month=df["month"].astype(int), day=1),
        errors="coerce",
    )
    df = df.dropna(subset=["date"]).set_index("date").sort_index()

    # Ensure month-start index
    df.index = df.index.to_period("M").to_timestamp(how="start")

    return df["EPU"].rename("EPU")
