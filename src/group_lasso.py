# src/group_lasso.py

import numpy as np
import pandas as pd

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