"""
Reproduce sections 5.0–5.4 of "Business News and Business Cycles" using:
- theta_monthly.csv (monthly topic attention, 1984–2017, downloaded from the autors' site: https://structureofnews.com) 
- phi_scaled.csv    (scaled word weights by topic, downloaded from the autors' site: https://structureofnews.com)
- US_Policy_Uncertainty_Data.xlsx (downloaded from authors' site : https://www.policyuncertainty.com/us_monthly.html)

What this script does (5.0–5.4):
5.0  Build the "recession attention" series from theta_monthly (auto-detect topic via phi_scaled)
5.1  Estimate baseline monthly VAR(3) with {recession, SP500, FEDFUNDS, PAYEMS, INDPRO}
     Compute orthogonalized IRFs; scale shock from 5th->95th percentile; bootstrap bands (Kilian-style residual bootstrap)
     Robustness: ordering recession first / second / last
     Compare with EPU shock (replace recession series with EPU)
5.3  Group-lasso selection among 180 topics + (EPU,VIX,UMCSENT) to predict core macro variables
     following the paper’s idea: select variables as groups across lags
5.4  Prints interpretation hooks (selected topic, effect sizes).
"""
#%% 
# -----------------------------
# Imports
# -----------------------------

from __future__ import annotations

# import sys
# import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import io
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from group_lasso import GroupLasso
from sklearn.model_selection import TimeSeriesSplit

# -----------------------------
# Config
# -----------------------------

DATA_DIR = Path("data")
THETA_FILE = DATA_DIR / "theta_monthly.csv"
RECESSION_COLNAME = "Recession"
EPU_XLSX_FILE = DATA_DIR / "US_Policy_Uncertainty_Data.xlsx"

OUT_DIR = Path("outputs_section5")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FRED_START = "1984-01-01"
FRED_END = "2017-12-31"

# Baseline series (monthly)
FRED_SERIES = {
    "INDPRO": "INDPRO",        # Industrial Production Index
    "PAYEMS": "PAYEMS",        # All Employees: Total Nonfarm Payrolls
    "FEDFUNDS": "FEDFUNDS",    # Effective Federal Funds Rate
    "VIXCLS": "VIXCLS",        # VIX (daily; optional robustness & 5.3)
    "VXOCLS": "VXOCLS",        # VXO (daily; for backup when VIX missing, i.e. before 1990)
    "UMCSENT": "UMCSENT",      # Michigan consumer sentiment (monthly; optional robustness & 5.3)
}
# Note that SP500 is not available before 2016 on FRED-MD, so we download it separately.

# VAR params
VAR_START = "1985-01-01"
LAGS = 3
IRF_HORIZON = 36
BOOT_REPS = 500  # increase (e.g. 1000–2000) for paper-like smooth bands (slower)

np.random.seed(42)

# GL parameters
FAST_MODE = True

if FAST_MODE:
    LAM_GRID = np.logspace(np.log10(1e-3), np.log10(2e-1), 25) #np.logspace(-3, -0.3, 20)   
    CV_SPLITS = 3
    N_ITER_PATH = 800                      
    N_ITER_CV   = 800                      
    TOL_PATH    = 1e-3                     
    TOL_CV      = 1e-3
else:
    LAM_GRID = np.logspace(-3, -0.3, 40) 
    CV_SPLITS = 10
    N_ITER_PATH = 5000
    N_ITER_CV   = 4000
    TOL_PATH    = 1e-5
    TOL_CV      = 1e-5

VAR_GL_START = "1986-01-01"
LAGS_GL = 3
Y_VARS = ["SP500", "FEDFUNDS", "PAYEMS", "INDPRO"]


# Plots parameters
H = IRF_HORIZON

#%%
# -----------------------------
# Utilities & loading / downloading data
# -----------------------------

# Data downloaders
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

# Helpers for IRF extraction and saving
def get_median_path(res, ordering, shock_var, response_var):
    """
    Median (point) IRF path for response_var to shock_var, already scaled.
    res.irf_scaled is (H+1, K, K).
    """
    shock_idx = ordering.index(shock_var)
    resp_idx = ordering.index(response_var)
    return res.irf_scaled[:, resp_idx, shock_idx]

def get_band_paths(res, ordering, response_var):
    """
    5% and 95% bands for response_var (for the shock used in this res).
    res.lower/res.upper are (H+1, K) with columns aligned to ordering.
    """
    resp_idx = ordering.index(response_var)
    lo = res.lower[:, resp_idx]
    hi = res.upper[:, resp_idx]
    return lo, hi

def save_irf_and_bands_csv(res, ordering, shock_var, out_dir, key, horizon):
    """
    Save:
      - responses to shock_var: (H+1 x K) matrix, columns=ordering
      - lower90 / upper90: (H+1 x K) matrices, columns=ordering
    """
    H = horizon
    shock_idx = ordering.index(shock_var)

    irf_resp = pd.DataFrame(
        res.irf_scaled[:, :, shock_idx],
        columns=ordering,
        index=pd.Index(range(H + 1), name="h"),
    )
    irf_resp.to_csv(out_dir / f"irf_{key}_responses_to_{shock_var}_shock.csv")

    bands_lo = pd.DataFrame(
        res.lower,
        columns=ordering,
        index=irf_resp.index,
    )
    bands_hi = pd.DataFrame(
        res.upper,
        columns=ordering,
        index=irf_resp.index,
    )
    bands_lo.to_csv(out_dir / f"irf_{key}_lower90.csv")
    bands_hi.to_csv(out_dir / f"irf_{key}_upper90.csv")

def peak_drop(series: pd.Series):
    """
    Convenience: min value & argmin (horizon index).
    """
    vmin = float(series.min())
    hmin = int(series.idxmin())
    return vmin, hmin


# -----------------------------
# VAR + Bootstrap IRFs
# -----------------------------

@dataclass
class IRFResult:
    ordering: List[str]
    irf: np.ndarray          # shape (H+1, K, K) orth IRFs for 1-orth shock
    irf_scaled: np.ndarray   # scaled by 5-95 percentile shock
    lower: np.ndarray        # (H+1, K) bands for response variable(s) to recession shock
    upper: np.ndarray


def fit_var(df: pd.DataFrame, lags: int) -> sm.tsa.vector_ar.var_model.VARResults:
    model = VAR(df)
    res = model.fit(lags)
    return res


def orth_irf(res: sm.tsa.vector_ar.var_model.VARResults, horizon: int) -> np.ndarray:
    # returns orthogonalized impulse responses
    irf_obj = res.irf(horizon)
    return irf_obj.orth_irfs  # (h+1, K, K)


def scale_to_percentile_structural_shock(
    res,
    shock_idx: int,
    p_low: float = 5,
    p_high: float = 95,
) -> float:
    """
    Compute shock scale as (p95 - p05) of the orthogonalized (structural) shock ε_t.

    statsmodels orth_irfs are responses to a 1-unit ε shock.
    Therefore the scaling factor is simply the desired ε amplitude.
    """
    # Reduced-form residuals u_t (T-lags, K)
    u = res.resid.values

    # Cholesky factor P such that Sigma_u = P P'
    P = np.linalg.cholesky(res.sigma_u.values)

    # Structural shocks: eps = P^{-1} u
    eps = np.linalg.solve(P, u.T).T  # (T-lags, K)

    shock_eps = eps[:, shock_idx]
    desired = np.nanpercentile(shock_eps, p_high) - np.nanpercentile(shock_eps, p_low)
    return float(desired)


def residual_bootstrap_irf_bands(
    df: pd.DataFrame,
    ordering: List[str],
    lags: int,
    horizon: int,
    shock_var: str,
    reps: int = 500,
    alpha: float = 0.10,
    seed: int = 123,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Kilian-style residual bootstrap:
    - fit VAR on df
    - resample residuals with replacement
    - simulate synthetic series using estimated parameters + resampled residuals
    - re-fit VAR, compute orth IRFs
    - collect response paths to shock_var shock (scaled later by same percentile scaling rule per bootstrap draw)
    Returns lower/upper bands for each variable's response to shock_var shock: (h+1, K)
    """
    rng = np.random.default_rng(seed)
    df = df[ordering].dropna()
    base_res = fit_var(df, lags)
    K = df.shape[1]
    T = df.shape[0]

    # Fitted pieces
    coefs = base_res.coefs  # (lags, K, K)
    intercept = base_res.intercept  # (K,)
    resid = base_res.resid.values  # (T-lags, K)
    sigma_u = base_res.sigma_u.values  # (K, K)

    # Initial history
    y = df.values
    y0 = y[:lags].copy()

    shock_idx = ordering.index(shock_var)

    irf_paths = np.zeros((reps, horizon + 1, K), dtype=float)

    for b in range(reps):
        # resample residuals (T-lags rows)
        eps = resid[rng.integers(0, resid.shape[0], size=resid.shape[0])]
        # simulate
        ys = np.zeros_like(y)
        ys[:lags] = y0
        for t in range(lags, T):
            yhat = intercept.copy()
            for L in range(1, lags + 1):
                yhat += coefs[L - 1] @ ys[t - L]
            ys[t] = yhat + eps[t - lags]

        df_b = pd.DataFrame(ys, index=df.index, columns=ordering)

        try:
            res_b = fit_var(df_b, lags)
            irf_b = orth_irf(res_b, horizon)
            # scale to 5–95 shock for each bootstrap draw
            sf = scale_to_percentile_structural_shock(res_b, shock_idx)
            # store responses of all variables to shock_var shock (diagonal shock)
            irf_paths[b] = (irf_b[:, :, shock_idx] * sf)
        except Exception:
            irf_paths[b, :, :] = np.nan

    # percentile bands, ignoring failed draws
    lower = np.nanpercentile(irf_paths, 100 * (alpha / 2), axis=0)
    upper = np.nanpercentile(irf_paths, 100 * (1 - alpha / 2), axis=0)
    return lower, upper


def run_var_irf(
    df: pd.DataFrame,
    ordering: List[str],
    shock_var: str,
    lags: int = LAGS,
    horizon: int = IRF_HORIZON,
    reps: int = BOOT_REPS,
) -> IRFResult:
    df = df[ordering].dropna()
    res = fit_var(df, lags)
    irf_o = orth_irf(res, horizon)
    shock_idx = ordering.index(shock_var)
    sf = scale_to_percentile_structural_shock(res, shock_idx)

    lower, upper = residual_bootstrap_irf_bands(
        df=df, ordering=ordering, lags=lags, horizon=horizon,
        shock_var=shock_var, reps=reps, alpha=0.10, seed=123
    )

    return IRFResult(
        ordering=ordering,
        irf=irf_o,
        irf_scaled=irf_o * sf,
        lower=lower,
        upper=upper,
    )


# -----------------------------
# Group-lasso selection
# -----------------------------

def variance_standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Variance-standardize each column (mean 0, std 1, ddof=0)."""
    return (df - df.mean()) / df.std(ddof=0)

def build_group_lasso_var_design(y: pd.DataFrame, x: pd.DataFrame, lags: int):
    """
    Build design matrix for a VARX(lags):
      y_t = c + sum_{i=1..lags} A_i y_{t-i} + sum_{i=1..lags} B_i x_{t-i} + u_t

    Groups:
      - group 0: intercept + ALL y-lags (unpenalized)
      - group j>=1: x-variable j (all its lags share one group)

    Returns
      Xmat: (T-lags, P) regressors
      Ymat: (T-lags, K) targets
      groups: (P,) int group labels
      colmeta: DataFrame describing each regressor column (name, block, lag, group)
    """
    assert (y.index.equals(x.index)), "y and x must share same index and alignment"
    T = len(y)
    K = y.shape[1]
    M = x.shape[1]

    idx = y.index

    # Targets: y_t for t = lags..T-1
    Ymat = y.iloc[lags:].to_numpy()

    X_cols = []
    groups = []
    meta_rows = []

    # Intercept (unpenalized)
    X_cols.append(np.ones((T - lags, 1)))
    groups.append(0)
    meta_rows.append({"name": "const", "block": "const", "var": "const", "lag": 0, "group": 0})

    # y-lags (unpenalized, all in group 0)
    for ell in range(1, lags + 1):
        y_l = y.shift(ell).iloc[lags:].to_numpy()  # (T-lags, K)
        X_cols.append(y_l)
        for j, v in enumerate(y.columns):
            groups.append(0)
            meta_rows.append({
                "name": f"{v}_L{ell}",
                "block": "y",
                "var": v,
                "lag": ell,
                "group": 0
            })

    # x-lags (penalized, one group per x-variable across all lags)
    # group id: 1..M aligned to x.columns order
    for m, xv in enumerate(x.columns, start=1):
        for ell in range(1, lags + 1):
            x_l = x[[xv]].shift(ell).iloc[lags:].to_numpy()  # (T-lags, 1)
            X_cols.append(x_l)
            groups.append(m)
            meta_rows.append({
                "name": f"{xv}_L{ell}",
                "block": "x",
                "var": xv,
                "lag": ell,
                "group": m
            })

    Xmat = np.concatenate(X_cols, axis=1)
    groups = np.array(groups, dtype=int)
    colmeta = pd.DataFrame(meta_rows)

    # sanity
    assert Xmat.shape[0] == Ymat.shape[0]
    assert Xmat.shape[1] == len(groups) == len(colmeta)

    return Xmat, Ymat, groups, colmeta

def group_l2_norms_by_var(coef: np.ndarray, colmeta: pd.DataFrame, x_vars: list[str]) -> pd.Series:
    """
    coef: (P, K) coefficients from GroupLasso (P regressors, K equations)
    Return L2 norm per x-variable aggregating all its lagged coefficients across all equations:
      norm(var) = sqrt( sum_{cols in var lags} sum_{k} coef[col,k]^2 )
    """
    norms = {}
    for v in x_vars:
        cols = colmeta.index[(colmeta["block"] == "x") & (colmeta["var"] == v)].to_numpy()
        if cols.size == 0:
            norms[v] = 0.0
        else:
            norms[v] = float(np.sqrt(np.sum(coef[cols, :] ** 2)))
    return pd.Series(norms)

def save_coef_tables(coef: np.ndarray, colmeta: pd.DataFrame, y_vars: list[str], out_path: str):
    """
    Save coefficients with metadata:
      rows = regressors, cols = equations (y variables)
    """
    df_coef = pd.DataFrame(coef, columns=y_vars)
    df_out = pd.concat([colmeta.reset_index(drop=True), df_coef.reset_index(drop=True)], axis=1)
    df_out.to_csv(out_path, index=False)
    return df_out

def cv_mse_for_lambda(X, Y, groups, lam, n_splits=10):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    mses = []

    for tr, te in tscv.split(X):
        gl = GroupLasso(
            groups=groups,
            group_reg=float(lam),
            l1_reg=0.0,
            n_iter=4000,
            tol=1e-5,
            supress_warning=True,
            fit_intercept=False,
            scale_reg="none",
        )
        gl.fit(X[tr], Y[tr])
        pred = gl.predict(X[te])
        mses.append(np.mean((Y[te] - pred) ** 2))

    return float(np.mean(mses))


# -------------------------
# Plotting
# -------------------------

def plot_irf_fig7(
    results: Dict[str, IRFResult],
    specs: Dict[str, dict],
    out_dir: Path,
    horizon: int,
    response_vars: List[str] = ["INDPRO", "PAYEMS"],
    response_titles: List[str] = ["Industrial Production", "Employment"],
    panelA_key: str = "baseline",
    panelB_main_key: str = "news_last",
    panelB_overlay_keys: Optional[List[str]] = None,
    fnameA: str = "Figure7_PanelA.png",
    fnameB: str = "Figure7_PanelB.png",
    suptitleA: str = "Panel A: Baseline VAR",
    suptitleB: str = "Panel B: Robustness",
):
    """
    Create and save Panel A and Panel B IRF plots.

    Panel A:
      - main spec = panelA_key
      - line with round markers + 90% bands (alpha=0.2)

    Panel B:
      - main spec = panelB_main_key
      - line with round markers + 90% bands OF panelB_main_key (NOT baseline)
      - overlays = panelB_overlay_keys as dashed lines (median only)
    """
    if panelB_overlay_keys is None:
        # default overlays = everything except the main key
        panelB_overlay_keys = [k for k in specs.keys() if k != panelB_main_key]

    H = horizon
    h = np.arange(H + 1)

    # Panel A
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

    resA = results[panelA_key]
    ordA = specs[panelA_key]["ordering"]
    shockA = specs[panelA_key]["shock"]

    for ax, var, title in zip(axes, response_vars, response_titles):
        med = get_median_path(resA, ordA, shockA, var)
        lo, hi = get_band_paths(resA, ordA, var)

        ax.plot(h, med, marker="o", markersize=3, linewidth=1)
        ax.fill_between(h, lo, hi, alpha=0.2)
        ax.axhline(0, linewidth=1)

        ax.set_title(title)
        ax.set_xlabel("Months")
        ax.set_ylabel("%")

    plt.suptitle(suptitleA)
    plt.tight_layout()
    plt.savefig(out_dir / fnameA, dpi=200)
    plt.close(fig)

    # Panel B
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

    # Main = news_last (bands + markers)
    resB_main = results[panelB_main_key]
    ordB_main = specs[panelB_main_key]["ordering"]
    shockB_main = specs[panelB_main_key]["shock"]
    main_label = specs[panelB_main_key].get("label", panelB_main_key)

    for ax, var, title in zip(axes, response_vars, response_titles):
        med_main = get_median_path(resB_main, ordB_main, shockB_main, var)
        lo_main, hi_main = get_band_paths(resB_main, ordB_main, var)

        ax.fill_between(h, lo_main, hi_main, alpha=0.15)
        ax.plot(h, med_main, marker="o", markersize=3, linewidth=2, label=main_label)

        # Overlays = dashed lines only
        for k in panelB_overlay_keys:
            if k == panelB_main_key:
                continue
            res_k = results[k]
            ord_k = specs[k]["ordering"]
            shock_k = specs[k]["shock"]
            label_k = specs[k].get("label", k)

            med_k = get_median_path(res_k, ord_k, shock_k, var)
            ax.plot(h, med_k, linestyle="--", linewidth=1.5, label=label_k)

        ax.axhline(0, linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Months")
        ax.set_ylabel("%")
        ax.legend(fontsize=8)

    plt.suptitle(suptitleB)
    plt.tight_layout()
    plt.savefig(out_dir / fnameB, dpi=200)
    plt.close(fig)

def plot_irf_panel(
    results: Dict[str, IRFResult],
    specs: Dict[str, dict],
    out_path: Path,
    horizon: int,
    resp_var: str,
    subplots: List[Dict[str, Any]],
    panel_title: Optional[str] = None,
    figsize: Optional[tuple] = None,
    sharey: bool = True,
    xlabel: str = "Months",
    ylabel: str = "%",
    legend_fontsize: int = 9,
    default_band_alpha: float = 0.20,
):
    """
    Generic IRF panel plotter.

    Parameters
    ----------
    results/specs:
        - results[key] -> IRFResult
        - specs[key]["ordering"], specs[key]["shock"], optional specs[key]["label"]
    out_path:
        where to save the figure (png, pdf, etc.)
    horizon:
        IRF horizon H (plots 0..H)
    resp_var:
        response variable name (must belong to each ordering used)
    subplots:
        list of subplot configs (one per axis). Example for 2 subplots:
        [
          {
            "title": "Left title",
            "series": [
              {"key":"run_key_1", "label":"...", "linestyle":"-", "linewidth":2, "with_band":True, "marker":None},
              {"key":"run_key_2", "label":"...", "linestyle":"--","linewidth":2, "with_band":True},
            ]
          },
          {
            "title": "Right title",
            "series": [...]
          }
        ]
    """
    H = horizon
    h = np.arange(H + 1)

    ncols = len(subplots)
    if figsize is None:
        figsize = (6 * ncols, 4)

    fig, axes = plt.subplots(1, ncols, figsize=figsize, sharex=True, sharey=sharey)
    if ncols == 1:
        axes = [axes]  # uniform handling

    for ax, sp in zip(axes, subplots):
        ax.set_title(sp.get("title", ""))

        for s in sp.get("series", []):
            key = s["key"]
            res_k = results[key]
            ord_k = specs[key]["ordering"]
            shock_k = specs[key]["shock"]

            label = s.get("label", specs[key].get("label", key))
            linestyle = s.get("linestyle", "-")
            linewidth = s.get("linewidth", 2)
            marker = s.get("marker", None)

            med = get_median_path(res_k, ord_k, shock_k, resp_var)
            ax.plot(
                h, med,
                linestyle=linestyle,
                linewidth=linewidth,
                marker=marker,
                label=label,
            )

            if s.get("with_band", False):
                lo, hi = get_band_paths(res_k, ord_k, resp_var)
                band_alpha = s.get("band_alpha", default_band_alpha)
                ax.fill_between(h, lo, hi, alpha=band_alpha)

        ax.axhline(0, linewidth=1)
        ax.set_xlabel(xlabel)

    # y-label only on left-most axis (cleaner, esp. sharey=True)
    axes[0].set_ylabel(ylabel)

    if panel_title:
        plt.suptitle(panel_title)

    # Legend: put on each axis by default (as in your current code)
    for ax in axes:
        ax.legend(fontsize=legend_fontsize)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


#%% 1. Load topic model outputs and download FRED series

print("[1/4] Loading topics, and EPU, and downloading FRED series ...")
theta = load_theta(THETA_FILE)
recession_attention = theta[RECESSION_COLNAME].rename("recession_attn")

print("[1/4]bis Downloading FRED series ...")
fred = {}
# Monthly series
fred["INDPRO"] = fred_download_csv(FRED_SERIES["INDPRO"], FRED_START, FRED_END)
fred["PAYEMS"] = fred_download_csv(FRED_SERIES["PAYEMS"], FRED_START, FRED_END)
fred["FEDFUNDS"] = fred_download_csv(FRED_SERIES["FEDFUNDS"], FRED_START, FRED_END)

# Daily -> monthly
sp500_daily = stooq_download_spx_daily(FRED_START, FRED_END)
sp500_monthly = to_monthly(sp500_daily, how="last")

# Combine core macro dataset
df = pd.concat(
    [
        recession_attention,
        sp500_monthly["SP500"],
        fred["FEDFUNDS"]["FEDFUNDS"],
        fred["PAYEMS"]["PAYEMS"],
        fred["INDPRO"]["INDPRO"],
    ],
    axis=1,
).loc[FRED_START:FRED_END]

# Transformations: 100*log for levels; rates in levels; attention in percent
df["recession_attn"] = 100.0 * df["recession_attn"]
df["SP500"] = 100.0 * safe_log(df["SP500"])
df["PAYEMS"] = 100.0 * safe_log(df["PAYEMS"])
df["INDPRO"] = 100.0 * safe_log(df["INDPRO"])
# FEDFUNDS stays as percent rate (already)
df = df.dropna()

print("[1/4]ter Loading EPU series ...")
# Load EPU (monthly) from Excel and align to df sample
epu = load_epu_xlsx(EPU_XLSX_FILE).loc[df.index.min():df.index.max()]
# Aligne index exactly on the same dates
epu = epu.reindex(df.index)


# Combine for full df
df_all = pd.concat([df, epu], axis=1).dropna()

# Adjust to start exactly like in the paper
df = df.loc[VAR_START:]
df_all = df_all.loc[VAR_START:]

#%% 2. Estimate VAR(3) and IRFs

print("[2/4] Estimating VAR(3) and IRFs (baseline + robustness orderings + EPU checks)...")

# --- Define specifications for each case ---
specs = {
    # Panel A baseline
    "baseline": {
        "df": df_all,
        "ordering": ["recession_attn", "SP500", "FEDFUNDS", "PAYEMS", "INDPRO"],
        "shock": "recession_attn",
        "label": "Baseline",
    },

    # Panel B robustness (ordering tests)
    "news_last": {
        "df": df_all,
        "ordering": ["SP500", "FEDFUNDS", "PAYEMS", "INDPRO", "recession_attn"],
        "shock": "recession_attn",
        "label": "News Last",
    },
    "news_second": {
        "df": df_all,
        "ordering": ["SP500", "recession_attn", "FEDFUNDS", "PAYEMS", "INDPRO"],
        "shock": "recession_attn",
        "label": "News 2nd",
    },

    # Panel B robustness: EPU instead of recession_attn (shock becomes EPU)
    "epu_instead": {
        "df": df_all,
        "ordering": ["EPU", "SP500", "FEDFUNDS", "PAYEMS", "INDPRO"],
        "shock": "EPU",
        "label": "EPU",
    },

    # Panel B robustness: include EPU as control (shock stays recession_attn)
    "incl_epu": {
        "df": df_all,
        "ordering": ["recession_attn", "EPU", "SP500", "FEDFUNDS", "PAYEMS", "INDPRO"],
        "shock": "recession_attn",
        "label": "Incl. EPU",
    },
}

results = {}

for key, s in specs.items():
    order = s["ordering"]
    shock_var = s["shock"]
    df_use = s["df"][order].dropna()

    print(f"  - {key}: shock={shock_var} ordering={order}")
    r = run_var_irf(
        df_use,
        ordering=order,
        shock_var=shock_var,
        lags=LAGS,
        horizon=IRF_HORIZON,
        reps=BOOT_REPS,
    )
    results[key] = r

    # Save IRFs to disk (responses to the specified shock)
    shock_idx = order.index(shock_var)
    irf_resp = pd.DataFrame(
        r.irf_scaled[:, :, shock_idx],
        columns=order,
        index=pd.Index(range(IRF_HORIZON + 1), name="h"),
    )
    irf_resp.to_csv(OUT_DIR / f"irf_{key}_responses_to_{shock_var}_shock.csv")

    bands_lo = pd.DataFrame(r.lower, columns=order, index=irf_resp.index)
    bands_hi = pd.DataFrame(r.upper, columns=order, index=irf_resp.index)
    bands_lo.to_csv(OUT_DIR / f"irf_{key}_lower90.csv")
    bands_hi.to_csv(OUT_DIR / f"irf_{key}_upper90.csv")

base = results["baseline"]
base_order = specs["baseline"]["ordering"]
base_shock = specs["baseline"]["shock"]
base_shock_idx = base_order.index(base_shock)
base_resp = pd.DataFrame(base.irf_scaled[:, :, base_shock_idx], columns=base_order)

ip_min, ip_h = peak_drop(base_resp["INDPRO"])
emp_min, emp_h = peak_drop(base_resp["PAYEMS"])
print("\n[Baseline peak responses to a 5th->95th percentile shock]")
print(f"  INDPRO: {ip_min:.2f} at h={ip_h} months")
print(f"  PAYEMS: {emp_min:.2f} at h={emp_h} months")

# Plot equivalent of Fig 7 (Panel A (Baseline VAR) + Panel B (Robustness checks))
plot_irf_fig7(
    results=results,
    specs=specs,
    out_dir=OUT_DIR,
    horizon=IRF_HORIZON,
    response_vars=["INDPRO", "PAYEMS"],
    response_titles=["Industrial Production", "Employment"],
    panelA_key="baseline",
    panelB_main_key="news_last",
    panelB_overlay_keys=["baseline", "epu_instead", "news_second", "incl_epu"],
    fnameA="Figure7_PanelA.png",
    fnameB="Figure7_PanelB.png",
    suptitleA="Panel A: Baseline VAR",
    suptitleB="Panel B: Robustness (News Last bands)",
)


#%% 3. News vs SP500 shocks and News vs EPU shocks 

print("[3/4] Estimating VAR(3) and IRFs for (News vs SP500 shocks) and (News vs EPU shocks) ...")

# ---- Figure 8 orderings ----
order_news_first = ["recession_attn", "SP500", "FEDFUNDS", "PAYEMS", "INDPRO"]
order_sp_first   = ["SP500", "recession_attn", "FEDFUNDS", "PAYEMS", "INDPRO"]

order_epu_first  = ["EPU", "SP500", "FEDFUNDS", "PAYEMS", "INDPRO"]
order_epu_second = ["SP500", "EPU", "FEDFUNDS", "PAYEMS", "INDPRO"]

# --- Define specifications for Figure 8 runs ---
specs_figure8 = {
    # For Panel A/B (INDPRO & PAYEMS), we need both shocks (news shock, SP500 shock) under both orderings.
    "news_first__shock_news": {
        "df": df_all,
        "ordering": order_news_first,
        "shock": "recession_attn",
        "label": "News first — News shock",
    },
    "news_first__shock_sp500": {
        "df": df_all,
        "ordering": order_news_first,
        "shock": "SP500",
        "label": "News first — SP500 shock",
    },
    "sp_first__shock_news": {
        "df": df_all,
        "ordering": order_sp_first,
        "shock": "recession_attn",
        "label": "SP500 first — News shock",
    },
    "sp_first__shock_sp500": {
        "df": df_all,
        "ordering": order_sp_first,
        "shock": "SP500",
        "label": "SP500 first — SP500 shock",
    },

    # For Panel C right subplot (EPU shock): EPU first vs EPU second
    "epu_first__shock_epu": {
        "df": df_all,
        "ordering": order_epu_first,
        "shock": "EPU",
        "label": "EPU first — EPU shock",
    },
    "epu_second__shock_epu": {
        "df": df_all,
        "ordering": order_epu_second,
        "shock": "EPU",
        "label": "EPU second — EPU shock",
    },
}

results_figure8 = {}

for key, s in specs_figure8.items():
    order = s["ordering"]
    shock_var = s["shock"]
    df_use = s["df"][order].dropna()

    print(f"  - {key}: shock={shock_var} ordering={order}")

    r = run_var_irf(
        df_use,
        ordering=order,
        shock_var=shock_var,
        lags=LAGS,
        horizon=IRF_HORIZON,
        reps=BOOT_REPS,   # make sure run_var_irf uses alpha=0.10 internally (=> 5/95)
    )
    results_figure8[key] = r

    # Save to CSV: responses + bands
    save_irf_and_bands_csv(
        res=r,
        ordering=order,
        shock_var=shock_var,
        out_dir=OUT_DIR,
        key=key,
        horizon=IRF_HORIZON,
    )

# Quick peak stats (optional)
H = IRF_HORIZON
h = np.arange(H + 1)

# Baseline here is "news_first__shock_news" (news shock, news first)
base_key = "news_first__shock_news"
base_res = results_figure8[base_key]
base_order = specs_figure8[base_key]["ordering"]
base_shock = specs_figure8[base_key]["shock"]

base_irf = pd.DataFrame(
    base_res.irf_scaled[:, :, base_order.index(base_shock)],
    columns=base_order,
    index=pd.Index(range(H + 1), name="h")
)

ip_min, ip_h = peak_drop(base_irf["INDPRO"])
emp_min, emp_h = peak_drop(base_irf["PAYEMS"])

print("\n[Baseline peak responses to news shock]")
print(f"  INDPRO: {ip_min:.2f} at h={ip_h} months")
print(f"  PAYEMS: {emp_min:.2f} at h={emp_h} months")


# Plots

# Panel A: INDPRO
plot_irf_panel(
    results=results_figure8,
    specs=specs_figure8,
    out_path=OUT_DIR / "Figure8_PanelA_INDPRO.png",
    horizon=IRF_HORIZON,
    resp_var="INDPRO",
    subplots=[
        {
            "title": "Panel A: INDPRO — News first",
            "series": [
                {"key": "news_first__shock_news",  "label": "Recession attention shock", "linestyle": "-",  "linewidth": 2, "with_band": True},
                {"key": "news_first__shock_sp500", "label": "SP500 shock",              "linestyle": "--", "linewidth": 2, "with_band": True},
            ],
        },
        {
            "title": "Panel A: INDPRO — SP500 first",
            "series": [
                {"key": "sp_first__shock_news",  "label": "Recession attention shock", "linestyle": "-",  "linewidth": 2, "with_band": True},
                {"key": "sp_first__shock_sp500", "label": "SP500 shock",              "linestyle": "--", "linewidth": 2, "with_band": True},
            ],
        },
    ],
    sharey=True,
)

# Panel B: PAYEMS
plot_irf_panel(
    results=results_figure8,
    specs=specs_figure8,
    out_path=OUT_DIR / "Figure8_PanelB_PAYEMS.png",
    horizon=IRF_HORIZON,
    resp_var="PAYEMS",
    subplots=[
        {
            "title": "Panel B: PAYEMS — News first",
            "series": [
                {"key": "news_first__shock_news",  "label": "Recession attention shock", "linestyle": "-",  "linewidth": 2, "with_band": True},
                {"key": "news_first__shock_sp500", "label": "SP500 shock",              "linestyle": "--", "linewidth": 2, "with_band": True},
            ],
        },
        {
            "title": "Panel B: PAYEMS — SP500 first",
            "series": [
                {"key": "sp_first__shock_news",  "label": "Recession attention shock", "linestyle": "-",  "linewidth": 2, "with_band": True},
                {"key": "sp_first__shock_sp500", "label": "SP500 shock",              "linestyle": "--", "linewidth": 2, "with_band": True},
            ],
        },
    ],
    sharey=True,
)

# Panel C: SP500 response (News shock vs EPU shock)
plot_irf_panel(
    results=results_figure8,
    specs=specs_figure8,
    out_path=OUT_DIR / "Figure8_PanelC_SP500Response_News_vs_EPU.png",
    horizon=IRF_HORIZON,
    resp_var="SP500",
    subplots=[
        {
            "title": "Panel C: SP500 response — News shock",
            "series": [
                {"key": "news_first__shock_news", "label": "News shock (news 1st)", "linestyle": "-",  "linewidth": 2, "with_band": True},
                {"key": "sp_first__shock_news",   "label": "News shock (news 2nd)", "linestyle": "--", "linewidth": 2, "with_band": True},
            ],
        },
        {
            "title": "Panel C: SP500 response — EPU shock (BBD VAR)",
            "series": [
                {"key": "epu_first__shock_epu",  "label": "EPU shock (EPU 1st)", "linestyle": "-",  "linewidth": 2, "with_band": True},
                {"key": "epu_second__shock_epu", "label": "EPU shock (EPU 2nd)", "linestyle": "--", "linewidth": 2, "with_band": True},
            ],
        },
    ],
    sharey=True,
)

# %% 4. Group Lasso VAR selection

print("[4/4] Group Lasso VAR selection...")

# ---- Build design matrices for Group-Lasso VAR ---
print("[4.1] Building inputs for Group-Lasso VAR...")

# Topics (180): in percentages so, 100 * theta (like in df_all only for recession topic)
x_topics = 100.0 * theta.copy()
x_topics.index = x_topics.index.to_period("M").to_timestamp(how="start")

# EPU already in df_all
x_epu = df_all[["EPU"]].copy()

# VIX completed by VXO 
vix_daily = fred_download_csv(FRED_SERIES["VIXCLS"], FRED_START, FRED_END)
vix_monthly = to_monthly(vix_daily, how="last").rename(columns={"VIXCLS":"VIX"})
vxo_daily = fred_download_csv(FRED_SERIES["VXOCLS"], FRED_START, FRED_END)
vxo_monthly = to_monthly(vxo_daily, how="last").rename(columns={"VXOCLS":"VIX"})
cut = vix_monthly.index.min()
x_vix = pd.concat([vxo_monthly.loc[vxo_monthly.index < cut], vix_monthly])

# Michigan Consumer Sentiment (UMCSENT)
fred["UMCSENT"] = fred_download_csv(FRED_SERIES["UMCSENT"], FRED_START, FRED_END)
x_umcsent = fred["UMCSENT"][["UMCSENT"]].copy()
x_umcsent.index = x_umcsent.index.to_period("M").to_timestamp(how="start")
x_umcsent = x_umcsent.rename(columns={"UMCSENT":"UMCSENT"})

# Assemble x_t
x_all = pd.concat([x_topics, x_epu, x_vix, x_umcsent], axis=1)

# Assemble y_t
y_all = df_all[Y_VARS].copy()

# Align & drop NA
y_all = y_all.loc[VAR_START:]
x_all = x_all.loc[VAR_START:]

idx = y_all.index.intersection(x_all.index)
y_all = y_all.loc[idx]
x_all = x_all.loc[idx]

df_joint = pd.concat([y_all, x_all], axis=1).dropna()
y_all = df_joint[Y_VARS]
x_all = df_joint[x_all.columns]

# Variance-standardize ALL variables 
y_std = variance_standardize(y_all)
x_std = variance_standardize(x_all)

Xmat, Ymat, groups, colmeta = build_group_lasso_var_design(y=y_std, x=x_std, lags=LAGS_GL)

x_vars = list(x_std.columns)  # 180 topics + EPU + VIX
y_vars = list(y_std.columns)

print(f"  - Samples: {Xmat.shape[0]}")
print(f"  - Regressors: {Xmat.shape[1]}")
print(f"  - Equations (K: #y variables): {Ymat.shape[1]}")
print(f"  - #x variables (topic+EPU(+VIX+UMCSENT)): {len(x_vars)}")

# Sanity checks
K = y_std.shape[1]
M = x_std.shape[1]
L = LAGS_GL
print("K=", K, "M=", M, "L=", L, "theoretical regressors =", 1 + L*(K+M))

print("df_joint first date:", df_joint.index[0])
print("df_joint last date :", df_joint.index[-1])
print("T (rows)           :", len(df_joint))
print("lags               :", L)
print("expected samples   :", len(df_joint) - L)


# ---- Fit Group Lasso path and compute group norms ----
print("[4.2/4] Fitting lambda path and computing group norms (Figure 9 data)...")

coef_path = {}          # lambda -> coef matrix (P,K)
norms_path = []         # rows = lambda, cols = x_vars

for lam in LAM_GRID:
    gl = GroupLasso(
        groups=groups,
        group_reg=float(lam),
        l1_reg=0.0,
        n_iter=N_ITER_PATH,
        tol=TOL_PATH,
        supress_warning=True,
        fit_intercept=False,   # we already included 'const'
        scale_reg="none",
    )
    gl.fit(Xmat, Ymat)

    coef = gl.coef_.copy()  # (P, K)
    coef_path[float(lam)] = coef

    norms = group_l2_norms_by_var(coef, colmeta, x_vars)
    norms.name = float(lam)
    norms_path.append(norms)

norms_df = pd.DataFrame(norms_path)
norms_df.index.name = "lambda"
norms_df = norms_df.sort_index()

# Save full norms path
norms_df.to_csv(OUT_DIR / "Figure9_norms_path_all_predictors.csv")
print("  - Saved norms path:", OUT_DIR / "Figure9_norms_path_all_predictors.csv")


# ---- Select top-10 survivors at high penalty (lambda max) and plot Figure 9 ----
print("[4.3/4] Selecting top-10 survivors at high penalty (lambda max) and plotting Figure 9...")

# numerical threshold for "active" variable
EPS = 1e-12  

# active variables count per lambda
active_counts = (norms_df > EPS).sum(axis=1)

# lambda strong: largest lambda with at least 10 active variables
if (active_counts >= 10).any():
    lam_strong = active_counts[active_counts >= 10].index.max()
else:
    # fallback: at least 1 active variable
    lam_strong = active_counts[active_counts >= 1].index.max()

print("lambda_strong =", lam_strong, "active vars =", int(active_counts.loc[lam_strong]))

# top10 variables surviving at lambda_strong
top10 = norms_df.loc[lam_strong].sort_values(ascending=False).head(10).index.tolist()
pd.Series(top10, name="top10_survivors").to_csv(OUT_DIR / "Figure9_top10_survivors.csv", index=False)

print("Top10 at strong penalty:", top10)

plt.figure(figsize=(10, 5))
for v in top10:
    plt.plot(norms_df.index.values, norms_df[v].values, label=v)

plt.xscale("log")
plt.axvline(lam_strong, linestyle="--", linewidth=1, label=f"strong λ={lam_strong:.4g}")
plt.xlabel("lambda (log scale)")
plt.ylabel("L2 norm of coefficients (group norm)")
plt.title("Figure 9-like: Group-Lasso VAR selection path (top 10 survivors at strong penalty)")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT_DIR / "Figure9_like_Top10_L2norms_vs_lambda.png", dpi=200)
plt.close()

print("  - Saved:", OUT_DIR / "Figure9_like_Top10_L2norms_vs_lambda.png")
print("  - Saved:", OUT_DIR / "Figure9_top10_survivors.csv")


# ---- 10-fold time-series CV to select lambda ----
print("[4.4/4] 10-fold time-series cross-validation to choose lambda...")

# Compute CV curve
cv_rows = []
for lam in LAM_GRID:
    mse = cv_mse_for_lambda(Xmat, Ymat, groups, lam, n_splits=10)
    cv_rows.append({"lambda": float(lam), "mse": mse})

cv_df = pd.DataFrame(cv_rows).sort_values("lambda")
cv_df.to_csv(OUT_DIR / "GroupLasso_CV_curve.csv", index=False)

best_row = cv_df.loc[cv_df["mse"].idxmin()]
best_lam = float(best_row["lambda"])

print(f"  - Best lambda (min CV MSE): {best_lam:.6f}")
print("  - Saved CV curve:", OUT_DIR / "GroupLasso_CV_curve.csv")

print("[4.5/4] Re-fitting Group-Lasso VAR at optimal lambda and saving coefficients...")

gl_best = GroupLasso(
    groups=groups,
    group_reg=best_lam,
    l1_reg=0.0,
    n_iter=N_ITER_CV,
    tol=TOL_CV,
    supress_warning=True,
    fit_intercept=False,
    scale_reg="none",
)
gl_best.fit(Xmat, Ymat)
coef_best = gl_best.coef_.copy()

# Save full coefficient table (all regressors)
save_coef_tables(
    coef=coef_best,
    colmeta=colmeta,
    y_vars=y_vars,
    out_path=OUT_DIR / "GroupLasso_VAR_coefficients_best_lambda.csv",
)

# Also save reduced table: only x-vars in top10 (plus intercept + y-lags)
keep_mask = (colmeta["block"] != "x") | (colmeta["var"].isin(top10))
coef_best_reduced = coef_best[keep_mask.to_numpy(), :]
colmeta_reduced = colmeta.loc[keep_mask].reset_index(drop=True)

save_coef_tables(
    coef=coef_best_reduced,
    colmeta=colmeta_reduced,
    y_vars=y_vars,
    out_path=OUT_DIR / "GroupLasso_VAR_coefficients_best_lambda_reduced_top10.csv",
)

# Save selected x-vars at best lambda (non-zero norms)
norms_best = group_l2_norms_by_var(coef_best, colmeta, x_vars).sort_values(ascending=False)
norms_best.to_csv(OUT_DIR / "GroupLasso_selected_norms_best_lambda.csv")

print("  - Saved:", OUT_DIR / "GroupLasso_VAR_coefficients_best_lambda.csv")
print("  - Saved:", OUT_DIR / "GroupLasso_VAR_coefficients_best_lambda_reduced_top10.csv")
print("  - Saved:", OUT_DIR / "GroupLasso_selected_norms_best_lambda.csv")


# %%
