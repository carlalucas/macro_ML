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

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import io
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.api import VAR

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
    "UMCSENT": "UMCSENT",      # Michigan consumer sentiment (monthly; optional robustness & 5.3)
}
# Note that SP500 is not available before 2016 on FRED-MD, so we download it separately.

# VAR params
LAGS = 3
IRF_HORIZON = 36
BOOT_REPS = 500  # increase (e.g. 1000–2000) for paper-like smooth bands (slower)

np.random.seed(42)

# -----------------------------
# Utilities & loading / downloading data
# -----------------------------

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


def safe_log(x: pd.Series) -> pd.Series:
    x = x.replace(0, np.nan)
    return np.log(x)


def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=0)


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

def get_median_path(res_obj, ordering, shock_var, response_var):
    shock_idx = ordering.index(shock_var)
    resp_idx = ordering.index(response_var)
    return res_obj.irf_scaled[:, resp_idx, shock_idx]

def get_band_paths(res_obj, ordering, response_var):
    resp_idx = ordering.index(response_var)
    lo = res_obj.lower[:, resp_idx]
    hi = res_obj.upper[:, resp_idx]
    return lo, hi

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
# Group-lasso selection (5.3)
# -----------------------------

def build_lagged_design(
    y: pd.DataFrame,
    x: pd.DataFrame,
    lags: int,
) -> Tuple[np.ndarray, np.ndarray, List[int], List[str]]:
    """
    Build design matrix for MultiTask regression:
      Y_t (K_y) on [const, lags of Y, lags of each X_j] with group ids
    Grouping:
      group 0: unpenalized (const + all Y lags)
      group j: lags of X_j (j>=1), so each candidate enters as a group across lags.
    Returns:
      Xmat: (T-lags, P)
      Ymat: (T-lags, K_y)
      groups: length P list of group ids
      colnames: length P list of regressor names
    """
    y = y.copy()
    x = x.copy()
    df = pd.concat([y, x], axis=1).dropna()
    y = df[y.columns]
    x = df[x.columns]

    T = df.shape[0]
    Y = y.iloc[lags:].values  # (T-lags, K_y)

    cols = []
    groups = []
    colnames = []

    # intercept
    cols.append(np.ones((T - lags, 1)))
    groups.append(0)
    colnames.append("const")

    # lags of Y (unpenalized)
    for L in range(1, lags + 1):
        lag_block = y.shift(L).iloc[lags:].values
        cols.append(lag_block)
        for c in y.columns:
            groups.append(0)
            colnames.append(f"{c}_L{L}")

    # lags of each X candidate (grouped by variable)
    for j, var in enumerate(x.columns, start=1):
        for L in range(1, lags + 1):
            lag_col = x[[var]].shift(L).iloc[lags:].values  # (T-lags, 1)
            cols.append(lag_col)
            groups.append(j)
            colnames.append(f"{var}_L{L}")

    Xmat = np.hstack(cols)
    return Xmat, Y, groups, colnames


def group_lasso_cv_select(
    X: np.ndarray,
    Y: np.ndarray,
    groups: List[int],
    colnames: List[str],
    candidate_names: List[str],
    lams: List[float],
    n_splits: int = 5,
) -> Tuple[float, List[str]]:
    """
    Cross-validate Group Lasso (if available) with simple expanding-window splits.
    Requires: pip install group-lasso
    """
    try:
        from group_lasso import GroupLasso
    except ImportError as e:
        raise ImportError(
            "Missing dependency 'group-lasso'. Install it with:\n"
            "  pip install group-lasso\n"
            "Then rerun."
        ) from e

    # Create time splits (expanding window)
    T = X.shape[0]
    fold_sizes = np.linspace(int(0.6 * T), int(0.9 * T), n_splits, dtype=int)

    best_lam = None
    best_score = np.inf

    for lam in lams:
        scores = []
        for cut in fold_sizes:
            Xtr, Ytr = X[:cut], Y[:cut]
            Xte, Yte = X[cut:], Y[cut:]

            gl = GroupLasso(
                groups=np.array(groups),
                group_reg=lam, l1_reg=0.0,
                n_iter=2000, tol=1e-4,
                supress_warning=True,
                fit_intercept=False,  # we already included const
                scale_reg="none",
            )
            gl.fit(Xtr, Ytr)
            Yhat = gl.predict(Xte)
            mse = np.mean((Yte - Yhat) ** 2)
            scores.append(mse)

        score = float(np.mean(scores))
        if score < best_score:
            best_score = score
            best_lam = lam

    # Refit on full sample at best lambda, extract selected groups
    gl = GroupLasso(
        groups=np.array(groups),
        group_reg=best_lam, l1_reg=0.0,
        n_iter=4000, tol=1e-5,
        supress_warning=True,
        fit_intercept=False,
        scale_reg="none",
    )
    gl.fit(X, Y)

    # Identify which candidate X variables have non-zero coefficients in any of their lag columns
    coef = gl.coef_  # (P, K_y)
    selected = []
    # group id j corresponds to candidate_names[j-1]
    for j, name in enumerate(candidate_names, start=1):
        idxs = [i for i, g in enumerate(groups) if g == j]
        if np.any(np.abs(coef[idxs, :]) > 1e-8):
            selected.append(name)

    return float(best_lam), selected

#%% 1. Load topic model outputs and download FRED series

print("[1/6] Loading topics, and EPU, and downloading FRED series ...")
theta = load_theta(THETA_FILE)
recession_attention = theta[RECESSION_COLNAME].rename("recession_attn")

print("[1/6]bis Downloading FRED series ...")
fred = {}
# Monthly series
fred["INDPRO"] = fred_download_csv(FRED_SERIES["INDPRO"], FRED_START, FRED_END)
fred["PAYEMS"] = fred_download_csv(FRED_SERIES["PAYEMS"], FRED_START, FRED_END)
fred["FEDFUNDS"] = fred_download_csv(FRED_SERIES["FEDFUNDS"], FRED_START, FRED_END)
fred["UMCSENT"] = fred_download_csv(FRED_SERIES["UMCSENT"], FRED_START, FRED_END)

# Daily -> monthly
sp500_daily = stooq_download_spx_daily(FRED_START, FRED_END)
vix_daily = fred_download_csv(FRED_SERIES["VIXCLS"], FRED_START, FRED_END)
sp500_monthly = to_monthly(sp500_daily, how="mean")
fred["VIXCLS"] = to_monthly(vix_daily, how="mean")

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

print("[1/6]ter Loading EPU series ...")
# Load EPU (monthly) from Excel and align to df sample
epu = load_epu_xlsx(EPU_XLSX_FILE).loc[df.index.min():df.index.max()]
# Aligne index exactly on the same dates
epu = epu.reindex(df.index)


# Combine for full df
df_all = pd.concat([df, epu], axis=1).dropna()


#%% 2. Estimate VAR(3) and IRFs

print("[2/6] Estimating VAR(3) and IRFs (baseline + robustness orderings + EPU checks)...")

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
    # NB: l’ordre exact est un choix d’identification. Ici on met EPU après SP500.
    "incl_epu": {
        "df": df_all,
        "ordering": ["SP500", "EPU", "recession_attn", "FEDFUNDS", "PAYEMS", "INDPRO"],
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

# Peak effects for baseline (like paper discussion)
def peak_drop(series: pd.Series) -> Tuple[float, int]:
    v = series.values
    h = int(np.nanargmin(v))
    return float(v[h]), h

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

# Plots

H = IRF_HORIZON
h = np.arange(H + 1)

# ---- Panel A: baseline only, with 5/95 band ----
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

for ax, var, title in zip(
    axes,
    ["INDPRO", "PAYEMS"],
    ["Industrial Production", "Employment"],
):
    base_res = results["baseline"]
    base_order = specs["baseline"]["ordering"]
    base_shock = specs["baseline"]["shock"]

    med = get_median_path(base_res, base_order, base_shock, var)
    lo, hi = get_band_paths(base_res, base_order, var)

    ax.plot(h, med, marker="o", markersize=3, linewidth=1)
    ax.fill_between(h, lo, hi, alpha=0.2)
    ax.axhline(0, linewidth=1)

    ax.set_title(title)
    ax.set_xlabel("Months")
    ax.set_ylabel("%")

plt.suptitle("Panel A: Baseline VAR")
plt.tight_layout()
plt.savefig(OUT_DIR / "Figure7_PanelA.png", dpi=200)
plt.close(fig)

# ---- Panel B: baseline band + median lines for robustness checks ----
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

robust_keys = ["baseline", "epu_instead", "news_second", "news_last", "incl_epu"]
robust_labels = {k: specs[k]["label"] for k in robust_keys}

for ax, var, title in zip(
    axes,
    ["INDPRO", "PAYEMS"],
    ["Industrial Production", "Employment"],
):
    # baseline band + baseline line
    base_res = results["baseline"]
    base_order = specs["baseline"]["ordering"]
    base_shock = specs["baseline"]["shock"]

    med_base = get_median_path(base_res, base_order, base_shock, var)
    lo, hi = get_band_paths(base_res, base_order, var)

    ax.fill_between(h, lo, hi, alpha=0.15)
    ax.plot(h, med_base, linewidth=2, label="Baseline")

    # other robustness: median lines only
    for k in robust_keys[1:]:
        res_k = results[k]
        ord_k = specs[k]["ordering"]
        shock_k = specs[k]["shock"]

        med_k = get_median_path(res_k, ord_k, shock_k, var)

        # style: dashed for robustness
        ax.plot(h, med_k, linestyle="--", linewidth=1.5, label=robust_labels[k])

    ax.axhline(0, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Months")
    ax.set_ylabel("%")
    ax.legend(fontsize=8)

plt.suptitle("Panel B: Robustness")
plt.tight_layout()
plt.savefig(OUT_DIR / "Figure7_PanelB.png", dpi=200)
plt.close(fig)

#%%
# -----------------------------
# Main workflow
# -----------------------------

def main() -> int:
    warnings.filterwarnings("ignore", category=FutureWarning)

    if not THETA_FILE.exists():
        print(f"[ERROR] Missing {THETA_FILE}.", file=sys.stderr)
        return 1
    
    if not EPU_XLSX_FILE.exists():
        print(f"[ERROR] Missing {EPU_XLSX_FILE}.", file=sys.stderr)
        return 1
    

    ############################################################
    # --- Load topic model, EPU, Download FRED series ---
    ############################################################

    print("[1/6] Loading topic model outputs...")
    theta = load_theta(THETA_FILE)
    recession_attention = theta[RECESSION_COLNAME].rename("recession_attn")

    print("[1/6] Downloading FRED series ...")
    fred = {}
    # Monthly series
    fred["INDPRO"] = fred_download_csv(FRED_SERIES["INDPRO"], FRED_START, FRED_END)
    fred["PAYEMS"] = fred_download_csv(FRED_SERIES["PAYEMS"], FRED_START, FRED_END)
    fred["FEDFUNDS"] = fred_download_csv(FRED_SERIES["FEDFUNDS"], FRED_START, FRED_END)
    fred["UMCSENT"] = fred_download_csv(FRED_SERIES["UMCSENT"], FRED_START, FRED_END)

    # Daily -> monthly
    sp500_daily = stooq_download_spx_daily(FRED_START, FRED_END)
    vix_daily = fred_download_csv(FRED_SERIES["VIXCLS"], FRED_START, FRED_END)
    sp500_monthly = to_monthly(sp500_daily, how="mean")
    fred["VIXCLS"] = to_monthly(vix_daily, how="mean")

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

    # Load EPU (monthly) from Excel and align to df sample
    epu = load_epu_xlsx(EPU_XLSX_FILE).loc[df.index.min():df.index.max()]

    ############################################################
    # --- VAR estimation and IRFs ---
    ############################################################

    print("[3/6] Estimating VAR(3) and IRFs (baseline + robustness orderings)...")
    # Baseline ordering (paper baseline places "recession" early; we’ll do it first)
    baseline_order = ["recession_attn", "SP500", "FEDFUNDS", "PAYEMS", "INDPRO"]
    # Robustness: recession last
    recession_last = ["SP500", "FEDFUNDS", "PAYEMS", "INDPRO", "recession_attn"]
    # Robustness: recession second (after SP500)
    recession_second = ["SP500", "recession_attn", "FEDFUNDS", "PAYEMS", "INDPRO"]

    results = {}
    for name, order in [
        ("baseline_recession_first", baseline_order),
        ("robust_recession_last", recession_last),
        ("robust_recession_second", recession_second),
    ]:
        print(f"  - {name}: ordering={order}")
        r = run_var_irf(df, ordering=order, shock_var="recession_attn", lags=LAGS, horizon=IRF_HORIZON, reps=BOOT_REPS)
        results[name] = r

        # Save a quick CSV of IRFs to disk (responses to recession shock)
        shock_idx = order.index("recession_attn")
        irf_resp = pd.DataFrame(
            r.irf_scaled[:, :, shock_idx],
            columns=order,
            index=pd.Index(range(IRF_HORIZON + 1), name="h"),
        )
        irf_resp.to_csv(OUT_DIR / f"irf_{name}_responses_to_recession_shock.csv")

        bands_lo = pd.DataFrame(r.lower, columns=order, index=irf_resp.index)
        bands_hi = pd.DataFrame(r.upper, columns=order, index=irf_resp.index)
        bands_lo.to_csv(OUT_DIR / f"irf_{name}_lower90.csv")
        bands_hi.to_csv(OUT_DIR / f"irf_{name}_upper90.csv")

    # Print peak effects (like paper discussion)
    def peak_drop(series: pd.Series) -> Tuple[float, int]:
        v = series.values
        h = int(np.nanargmin(v))
        return float(v[h]), h

    base = results["baseline_recession_first"]
    order = base.ordering
    shock_idx = order.index("recession_attn")
    base_resp = pd.DataFrame(base.irf_scaled[:, :, shock_idx], columns=order)

    ip_min, ip_h = peak_drop(base_resp["INDPRO"])
    emp_min, emp_h = peak_drop(base_resp["PAYEMS"])
    sp_min, sp_h = peak_drop(base_resp["SP500"])
    print("\n[Baseline peak responses to a 5th->95th 'recession attention' shock]")
    print(f"  INDPRO: {ip_min:.2f} at h={ip_h} months")
    print(f"  PAYEMS: {emp_min:.2f} at h={emp_h} months")
    print(f"  SP500 : {sp_min:.2f} at h={sp_h} months")


    ############################################################
    # --- EPU VAR ---
    ############################################################

    print("\n[4/6] Comparison: replace recession with EPU (BBD-style comparison)...")
    # --- EPU benchmark VAR (like BBD comparison in the paper) ---
    if epu is not None and epu.notna().any():
        df_epu = df.drop(columns=["recession_attn"]).copy()
        df_epu = pd.concat([epu, df_epu], axis=1).dropna()

        # Option 1 (proche du papier) : choc 5e->95e en niveau d'EPU (pas de zscore)
        epu_order = ["EPU", "SP500", "FEDFUNDS", "PAYEMS", "INDPRO"]
        r_epu = run_var_irf(
            df_epu,
            ordering=epu_order,
        shock_var="EPU",
            lags=LAGS,
            horizon=IRF_HORIZON,
            reps=BOOT_REPS
        )
        # Sauvegarde des IRFs et bandes
        shock_idx = epu_order.index("EPU")
        irf_resp_epu = pd.DataFrame(
            r_epu.irf_scaled[:, :, shock_idx],
            columns=epu_order,
            index=pd.Index(range(IRF_HORIZON + 1), name="h"),
        )
        irf_resp_epu.to_csv(OUT_DIR / "irf_epu_responses_to_epu_shock.csv")
        pd.DataFrame(r_epu.lower, columns=epu_order, index=irf_resp_epu.index).to_csv(OUT_DIR / "irf_epu_lower90.csv")
        pd.DataFrame(r_epu.upper, columns=epu_order, index=irf_resp_epu.index).to_csv(OUT_DIR / "irf_epu_upper90.csv")
    else:
        print("[WARN] EPU series is missing or empty; skipping EPU VAR.")


    ############################################################
    # --- Group-lasso selection among topics + EPU/VIX/UMCSENT ---
    ############################################################

    print("\n[5/6] Group-lasso selection among topics + EPU/VIX/UMCSENT (section 5.3 idea)...")
    # Core macro variables y_t (paper uses a core set; here we use SP500, FEDFUNDS, PAYEMS, INDPRO)
    y_core = df[["SP500", "FEDFUNDS", "PAYEMS", "INDPRO"]].copy()

    # Candidate x_t: all topic attentions (scaled to percent) + VIX + UMCSENT (+ EPU if available)
    x_topics = theta.copy()
    # Convert to percent scale to match recession_attn handling
    x_topics = 100.0 * x_topics
    # Align to monthly start
    x_topics.index = x_topics.index.to_period("M").to_timestamp("MS")

    x_other = pd.concat(
        [
            fred["VIXCLS"]["VIXCLS"].rename("VIXCLS"),
            fred["UMCSENT"]["UMCSENT"].rename("UMCSENT"),
        ],
        axis=1,
    )
    x_other.index = x_other.index.to_period("M").to_timestamp("MS")

    x_all = pd.concat([x_topics, x_other], axis=1)

    # # Add EPU if present
    # if EPU_FILE is not None and Path(EPU_FILE).exists():
    #     epu = load_epu(Path(EPU_FILE))
    #     epu = epu.rename("EPU")
    #     epu.index = epu.index.to_period("M").to_timestamp("MS")
    #     x_all = pd.concat([x_all, epu], axis=1)

    # Restrict to sample
    x_all = x_all.loc[y_core.index.min():y_core.index.max()]
    y_core = y_core.loc[x_all.index.min():x_all.index.max()]

    # Build lagged regression matrices
    Xmat, Ymat, groups, colnames = build_lagged_design(y=y_core, x=x_all, lags=LAGS)

    candidate_names = list(x_all.columns)

    # Lambda grid (you can widen/narrow depending on selection aggressiveness)
    lam_grid = np.logspace(-3, 0.5, 12).tolist()

    try:
        best_lam, selected = group_lasso_cv_select(
            X=Xmat, Y=Ymat, groups=groups, colnames=colnames,
            candidate_names=candidate_names, lams=lam_grid, n_splits=5
        )
        print(f"  Best lambda (group_reg) = {best_lam:.6f}")
        print(f"  Selected candidates (count={len(selected)}):")
        for s in selected[:30]:
            print(f"    - {s}")
        if len(selected) > 30:
            print("    ...")

        # Save selection
        pd.Series(selected, name="selected").to_csv(OUT_DIR / "group_lasso_selected_candidates.csv", index=False)

        # Check if recession topic is selected
        # We compare using theta column name that corresponded to recession_col
        rec_in_selected = recession_col in selected
        print(f"\n  Recession topic column '{recession_col}' selected? {rec_in_selected}")

    except ImportError as e:
        print("  [Skipped group-lasso] " + str(e))

    print("\n[6/6] Done. Outputs saved in:", OUT_DIR.resolve())
    print("Key files:")
    print("  - irf_*_responses_to_recession_shock.csv (+ lower/upper 90% bands)")
    print("  - irf_epu_responses_to_epu_shock.csv")
    print("  - group_lasso_selected_candidates.csv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# %%
