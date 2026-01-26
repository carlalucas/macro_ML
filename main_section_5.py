#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproduce sections 5.0–5.4 of "Business News and Business Cycles" using:
- theta_monthly.csv (monthly topic attention, 1984–2017)
- phi_scaled.csv    (scaled word weights by topic)
- (optional) EPU csv you download from authors' site

What this script does (5.0–5.4):
5.0  Build the "recession attention" series from theta_monthly (auto-detect topic via phi_scaled)
5.1  Estimate baseline monthly VAR(3) with {recession, SP500, FEDFUNDS, PAYEMS, INDPRO}
     Compute orthogonalized IRFs; scale shock from 5th->95th percentile; bootstrap bands (Kilian-style residual bootstrap)
     Robustness: ordering recession first / second / last
     Compare with EPU shock (replace recession series with EPU)
5.3  Group-lasso selection among 180 topics + (EPU,VIX,UMCSENT) to predict core macro variables
     following the paper’s idea: select variables as groups across lags
5.4  Prints interpretation hooks (selected topic, effect sizes); narrative discussion is not computational.

No pandas-datareader used. FRED download is automated via fredgraph.csv (no API key).
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
from statsmodels.tsa.api import VAR

# -----------------------------
# Config
# -----------------------------

DATA_DIR = Path("data")
THETA_FILE = DATA_DIR / "theta_monthly.csv"
PHI_FILE = DATA_DIR / "phi_scaled.csv"

# Optional: your EPU csv you download from authors' site (set to None if not used)
# (Expected: a date column + a value column, see load_epu() below.)
EPU_FILE = DATA_DIR / "epu_monthly.csv"  # change if needed; or set to None

OUT_DIR = Path("outputs_section5")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FRED_START = "1984-01-01"
FRED_END = "2017-12-31"

# Baseline series (monthly)
FRED_SERIES = {
    "INDPRO": "INDPRO",        # Industrial Production Index
    "PAYEMS": "PAYEMS",        # All Employees: Total Nonfarm Payrolls
    "FEDFUNDS": "FEDFUNDS",    # Effective Federal Funds Rate
    "SP500": "SP500",          # S&P 500 (daily; will be converted to monthly)
    "VIXCLS": "VIXCLS",        # VIX (daily; optional robustness & 5.3)
    "UMCSENT": "UMCSENT",      # Michigan consumer sentiment (monthly; optional robustness & 5.3)
}

# Heuristic to identify the "recession" topic from phi_scaled
RECESSION_KEYWORDS = {
    "recession", "downturn", "slowdown", "slump", "contraction", "jobless",
    "unemployment", "layoffs", "bankrupt", "bankruptcy", "foreclosure",
    "weak", "decline", "falling", "drop", "credit", "crisis"
}

# If you already KNOW the recession topic id/column, set it here to bypass auto-detection.
# Examples: "topic_37" or "37" or whatever your theta columns look like.
RECESSION_TOPIC_OVERRIDE: Optional[str] = None

# VAR params
LAGS = 3
IRF_HORIZON = 36
BOOT_REPS = 500  # increase (e.g. 1000–2000) for paper-like smooth bands (slower)

np.random.seed(42)


# -----------------------------
# Utilities
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


# -----------------------------
# Load topic model outputs
# -----------------------------

def load_theta(path: Path) -> pd.DataFrame:
    """
    Load theta_monthly.csv: monthly topic attention. Tries to infer date column.
    Expected: one date-like column and many topic columns (180).
    """
    df = pd.read_csv(path)
    # Find a date column
    date_col = None
    for c in df.columns:
        if c.lower() in {"date", "dt", "month", "time"}:
            date_col = c
            break
    if date_col is None:
        # try first column
        date_col = df.columns[0]

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    # Ensure monthly start frequency
    df.index = df.index.to_period("M").to_timestamp("MS")
    return df


def load_phi(path: Path) -> pd.DataFrame:
    """
    Load phi_scaled.csv: scaled word weights by topic.
    Handles two common formats:
      (A) columns: ["topic", "word", "phi_scaled"] (long)
      (B) wide: index=word, columns=topic ids
    Returns a tidy long DataFrame with columns: topic, word, weight
    """
    df = pd.read_csv(path)

    cols_lower = {c.lower() for c in df.columns}
    if {"topic", "word"}.issubset(cols_lower):
        # long format
        topic_col = [c for c in df.columns if c.lower() == "topic"][0]
        word_col = [c for c in df.columns if c.lower() == "word"][0]
        # weight col guess
        weight_col = None
        for c in df.columns:
            if c.lower() in {"phi_scaled", "weight", "phi", "phi_tilde", "score", "value"}:
                weight_col = c
                break
        if weight_col is None:
            # last column as fallback
            weight_col = df.columns[-1]

        out = df[[topic_col, word_col, weight_col]].copy()
        out.columns = ["topic", "word", "weight"]
        out["topic"] = out["topic"].astype(str)
        out["word"] = out["word"].astype(str)
        out["weight"] = pd.to_numeric(out["weight"], errors="coerce")
        return out.dropna(subset=["weight"])

    # wide format fallback: first col words, rest topics
    word_col = df.columns[0]
    out = df.melt(id_vars=[word_col], var_name="topic", value_name="weight")
    out = out.rename(columns={word_col: "word"})
    out["topic"] = out["topic"].astype(str)
    out["word"] = out["word"].astype(str)
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce")
    return out.dropna(subset=["weight"])


def top_words_by_topic(phi_long: pd.DataFrame, topn: int = 30) -> Dict[str, List[str]]:
    """
    Returns dict: topic -> list of topn words by weight (descending)
    """
    d: Dict[str, List[str]] = {}
    for t, g in phi_long.groupby("topic"):
        g2 = g.sort_values("weight", ascending=False).head(topn)
        d[t] = g2["word"].str.lower().tolist()
    return d


def detect_recession_topic(phi_long: pd.DataFrame) -> str:
    """
    Heuristic: choose topic with max keyword matches in its top words,
    breaking ties by total matched weights.
    """
    topw = top_words_by_topic(phi_long, topn=50)

    # Precompute weights map for tie-breaker
    phi_long["word_l"] = phi_long["word"].str.lower()
    weights = (
        phi_long[phi_long["word_l"].isin(RECESSION_KEYWORDS)]
        .groupby("topic")["weight"]
        .sum()
        .to_dict()
    )

    best_topic = None
    best_score = (-1, -np.inf)

    for t, words in topw.items():
        hits = sum(1 for w in words if w in RECESSION_KEYWORDS)
        wsum = weights.get(t, 0.0)
        score = (hits, wsum)
        if score > best_score:
            best_score = score
            best_topic = t

    if best_topic is None:
        raise RuntimeError("Could not detect recession topic from phi_scaled.csv.")

    return str(best_topic)


# -----------------------------
# Load EPU (optional)
# -----------------------------

def load_epu(path: Path) -> pd.Series:
    """
    Load an EPU csv you downloaded from authors' site.
    Flexible parsing:
      - finds a date-like column and a numeric value column
    Returns monthly-start indexed Series named "EPU".
    """
    df = pd.read_csv(path)
    # date col
    date_col = None
    for c in df.columns:
        if c.lower() in {"date", "dt", "month", "time"}:
            date_col = c
            break
    if date_col is None:
        date_col = df.columns[0]

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    df.index = df.index.to_period("M").to_timestamp("MS")

    # value col: first numeric column
    val_col = None
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) or pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.9:
            val_col = c
            break
    if val_col is None:
        val_col = df.columns[-1]

    s = pd.to_numeric(df[val_col], errors="coerce").rename("EPU")
    return s


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


def scale_to_percentile_shock(
    irf_orth: np.ndarray,
    df: pd.DataFrame,
    shock_var: str,
    shock_idx: int,
    p_low: float = 5,
    p_high: float = 95,
) -> float:
    """
    Scale factor so that impact on shock_var at h=0 equals (p_high - p_low) in units of the variable.
    """
    desired = np.nanpercentile(df[shock_var], p_high) - np.nanpercentile(df[shock_var], p_low)
    impact = irf_orth[0, shock_idx, shock_idx]  # response of shock_var to its own orth shock at h=0
    if impact == 0 or np.isnan(impact):
        raise RuntimeError("Zero/NaN impact in IRF scaling.")
    return desired / impact


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
            sf = scale_to_percentile_shock(irf_b, df_b, shock_var, shock_idx)
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
    sf = scale_to_percentile_shock(irf_o, df, shock_var, shock_idx)

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


# -----------------------------
# Main workflow
# -----------------------------

def main() -> int:
    warnings.filterwarnings("ignore", category=FutureWarning)

    if not THETA_FILE.exists():
        print(f"[ERROR] Missing {THETA_FILE}.", file=sys.stderr)
        return 1
    if not PHI_FILE.exists():
        print(f"[ERROR] Missing {PHI_FILE}.", file=sys.stderr)
        return 1

    print("[1/6] Loading topic model outputs...")
    theta = load_theta(THETA_FILE)
    phi_long = load_phi(PHI_FILE)

    if RECESSION_TOPIC_OVERRIDE is not None:
        recession_topic = str(RECESSION_TOPIC_OVERRIDE)
        print(f"  Using RECESSION_TOPIC_OVERRIDE={recession_topic}")
    else:
        recession_topic = detect_recession_topic(phi_long)
        print(f"  Auto-detected recession topic: {recession_topic}")

    # Map recession topic to a theta column (handle common naming mismatches)
    theta_cols = list(map(str, theta.columns))
    if recession_topic in theta_cols:
        recession_col = recession_topic
    else:
        # try common patterns
        candidates = [
            f"topic_{recession_topic}",
            f"Topic_{recession_topic}",
            f"t{recession_topic}",
            recession_topic.zfill(3),
        ]
        match = next((c for c in candidates if c in theta_cols), None)
        if match is None:
            # last resort: try numeric match
            numeric = "".join(ch for ch in recession_topic if ch.isdigit())
            match = next((c for c in theta_cols if "".join(ch for ch in c if ch.isdigit()) == numeric), None)
        if match is None:
            print("[ERROR] Could not match recession topic to a theta column.", file=sys.stderr)
            print("  Detected topic:", recession_topic, file=sys.stderr)
            print("  Theta columns sample:", theta_cols[:10], "...", file=sys.stderr)
            return 1
        recession_col = match

    recession_attention = theta[recession_col].rename("recession_attn")

    print("[2/6] Downloading FRED series (no pandas-datareader)...")
    fred = {}
    # Monthly series
    fred["INDPRO"] = fred_download_csv(FRED_SERIES["INDPRO"], FRED_START, FRED_END)
    fred["PAYEMS"] = fred_download_csv(FRED_SERIES["PAYEMS"], FRED_START, FRED_END)
    fred["FEDFUNDS"] = fred_download_csv(FRED_SERIES["FEDFUNDS"], FRED_START, FRED_END)
    fred["UMCSENT"] = fred_download_csv(FRED_SERIES["UMCSENT"], FRED_START, FRED_END)

    # Daily -> monthly
    sp500_daily = fred_download_csv(FRED_SERIES["SP500"], FRED_START, FRED_END)
    vix_daily = fred_download_csv(FRED_SERIES["VIXCLS"], FRED_START, FRED_END)
    fred["SP500"] = to_monthly(sp500_daily, how="mean")
    fred["VIXCLS"] = to_monthly(vix_daily, how="mean")

    # Combine core macro dataset
    df = pd.concat(
        [
            recession_attention,
            fred["SP500"]["SP500"],
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

    print("\n[4/6] Optional comparison: replace recession with EPU (BBD-style comparison)...")
    if EPU_FILE is not None and Path(EPU_FILE).exists():
        epu = load_epu(Path(EPU_FILE)).loc[df.index.min():df.index.max()]
        # Align & transform (typical: log or level; we keep level and zscore for comparability)
        # If your EPU is already an index, you can keep in levels. Here: z-score helps scale.
        epu = zscore(epu).rename("EPU")
        df_epu = df.copy()
        df_epu = df_epu.drop(columns=["recession_attn"])
        df_epu = pd.concat([epu, df_epu], axis=1).dropna()

        epu_order = ["EPU", "SP500", "FEDFUNDS", "PAYEMS", "INDPRO"]
        print(f"  - EPU VAR ordering={epu_order}")
        r_epu = run_var_irf(df_epu, ordering=epu_order, shock_var="EPU", lags=LAGS, horizon=IRF_HORIZON, reps=BOOT_REPS)

        shock_idx2 = epu_order.index("EPU")
        irf_resp_epu = pd.DataFrame(r_epu.irf_scaled[:, :, shock_idx2], columns=epu_order)
        irf_resp_epu.to_csv(OUT_DIR / "irf_epu_responses_to_epu_shock.csv")
        pd.DataFrame(r_epu.lower, columns=epu_order).to_csv(OUT_DIR / "irf_epu_lower90.csv")
        pd.DataFrame(r_epu.upper, columns=epu_order).to_csv(OUT_DIR / "irf_epu_upper90.csv")

        ip_min2, ip_h2 = peak_drop(irf_resp_epu["INDPRO"])
        emp_min2, emp_h2 = peak_drop(irf_resp_epu["PAYEMS"])
        print("  [EPU peak responses to 5–95 shock (z-scored EPU; interpret magnitudes cautiously)]")
        print(f"    INDPRO: {ip_min2:.2f} at h={ip_h2}")
        print(f"    PAYEMS: {emp_min2:.2f} at h={emp_h2}")
    else:
        print("  (Skipped: EPU_FILE not found. Put your downloaded EPU csv in data/ and set EPU_FILE.)")

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

    # Add EPU if present
    if EPU_FILE is not None and Path(EPU_FILE).exists():
        epu = load_epu(Path(EPU_FILE))
        epu = epu.rename("EPU")
        epu.index = epu.index.to_period("M").to_timestamp("MS")
        x_all = pd.concat([x_all, epu], axis=1)

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
    print("  - (optional) irf_epu_responses_to_epu_shock.csv")
    print("  - group_lasso_selected_candidates.csv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
