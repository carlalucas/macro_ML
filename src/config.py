# src/config.py
from pathlib import Path
import numpy as np

# -----------------------------
# LARS config
# -----------------------------
START_DATE = '1984-01-01'
END_DATE = '2017-06-01'

N_VARS_LASSO = 5

# -----------------------------
# Paths / sample
# -----------------------------
DATA_DIR = Path("data")
THETA_FILE = DATA_DIR / "theta_monthly.csv"
EPU_XLSX_FILE = DATA_DIR / "US_Policy_Uncertainty_Data.xlsx"

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FRED_START = "1984-01-01"
FRED_END   = "2017-12-31"

# -----------------------------
# Series IDs
# -----------------------------
FRED_SERIES = {
    "INDPRO":   "INDPRO",
    "FEDFUNDS": "FEDFUNDS",
    "PCEPI":    "PCEPI",
    "PCEPILFE": "PCEPILFE",
    "CPIAUCSL": "CPIAUCSL",      
    "VIXCLS":   "VIXCLS",
    "VXOCLS":   "VXOCLS",
    "UMCSENT":  "UMCSENT",
}

# -----------------------------
# VAR / IRF params
# -----------------------------
LAGS = 3
IRF_HORIZON = 36
BOOT_REPS = 500
np.random.seed(42)

# -----------------------------
# Group-Lasso params
# -----------------------------
CV_SPLITS   = 5
N_ITER_PATH = 1200
TOL_PATH    = 5e-4
N_ITER_CV   = 800
TOL_CV      = 1e-3

LAM_HI = 2e-1
LAM_LO = 4e-2
LAM_POINTS = 40
LAM_GRID_SCALED = np.geomspace(LAM_HI, LAM_LO, LAM_POINTS)

REL_EPS = 1e-3
ABS_EPS = 1e-8

VAR_GL_START = "1986-01-01"

# y_t demandé
Y_ORDER = ["pi_pce", "logIP", "FFR", "logSP500"]
Y_ORDER_CORE = ["pi_pce_core", "logIP", "FFR", "logSP500"]