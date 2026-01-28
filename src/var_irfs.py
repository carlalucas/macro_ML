# src/var_irfs.py

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.api import VAR

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
    lags: int = 3,
    horizon: int = 36,
    reps: int = 500,
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