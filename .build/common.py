"""Common config, data loader, and leak-free helpers for the Bangladesh DFAA study.

All paths hardcoded (workspace convention). Train-only fits everywhere.
Notation matches EXPERIMENT_DESIGN.md. No experiment numbers are produced here;
this is the shared, smoke-testable core that the notebooks reuse.
"""
from pathlib import Path
import os
import json
import numpy as np
import pandas as pd

# ROOT overridable so the shipped notebooks run on Colab/local (set env DFAA_ROOT).
ROOT = Path(os.environ.get("DFAA_ROOT", r"d:\BUET RESEARCH WORK\Bangladesh Flood"))
RAW_CSV = ROOT / "Bangladesh Meterological data.csv"
ART = ROOT / "artefacts"
ART.mkdir(exist_ok=True)

# ---- locked study constants (EXPERIMENT_DESIGN.md defaults) ----
SEED = 0
VARS_Z = ["Rainfall_mm", "Soil_moisture_mm"]          # standardized by train climatology
# robust standardization: per-(s,m) sd floored at SD_FLOOR_FRAC * station-pooled train sd,
# then z clipped to +-Z_CLIP. Guards the soil-moisture saturation / dry-month near-zero-sd
# pathology (otherwise z -> ~-40000). Fixed/train-only transforms => no leakage; train cells
# (|z|<3.6) are untouched. Documented in RESULTS_LOG S1/S2.
SD_FLOOR_FRAC = 0.15
Z_CLIP = 4.0
W_WEIGHTS = (1 / 3, 1 / 3, 1 / 3)                     # w1*z_P + w2*z_SM + w3*SPEI
ALPHA_DFAA = 1.8                                       # Wu (2006) constant in Eq. 2
LEADS = (1, 2, 3)                                      # symmetric window scale = lead h
TAUS = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95], dtype=np.float64)
THETA_PCT = 80.0                                       # theta_D = 80th pct of |DFAA| on train

# time-ordered split by ORIGIN year (the month t at which DFAA(s,t) is anchored)
TRAIN_YEARS = (2000, 2014)
VAL_YEARS = (2015, 2017)
TEST_YEARS = (2018, 2022)

# BMD station latitudes/longitudes (deg) for PET extraterrestrial radiation + maps.
# Standard BMD station coordinates; for PET only latitude matters (Ra is ~flat to +-0.2 deg).
# Provenance flagged for final verification before the .tex (CLAUDE.md Rule 3).
STATION_LATLON = {
    "Barisal":              (22.70, 90.37),
    "Bogra":                (24.85, 89.37),
    "Chittagong(Air-port)": (22.25, 91.81),
    "Comilla":              (23.43, 91.18),
    "Cox's Bazar":          (21.45, 91.97),
    "Dhaka":                (23.78, 90.38),
    "Faridpur":             (23.60, 89.85),
    "Jessore":              (23.18, 89.16),
    "Khulna":               (22.78, 89.53),
    "Mymensingh":           (24.75, 90.43),
    "Rajshahi":             (24.37, 88.70),
    "Rangpur":              (25.73, 89.23),
    "Sylhet":               (24.90, 91.88),
}


def load_clean():
    """Load raw CSV, drop the trailing all-NaN row, sort, add integer month index t.

    Returns a tidy long DataFrame with columns:
      Station_Name, Station_Code, Year, Month, Max_Temp, Min_Temp, Rainfall_mm,
      Soil_moisture_mm, SPEI_3, s (0..12 station id), t (0-based global month index),
      origin_year, split.
    Raw file is never modified.
    """
    df = pd.read_csv(RAW_CSV)
    # drop rows that are entirely NaN in the value columns (the trailing NaN row)
    val_cols = ["Max_Temp", "Min_Temp", "Rainfall_mm", "Soil_moisture_mm", "SPEI_3"]
    before = len(df)
    df = df.dropna(subset=["Station_Name", "Year", "Month"], how="any").copy()
    df = df.dropna(subset=val_cols, how="all").copy()
    dropped = before - len(df)

    df["Year"] = df["Year"].astype(int)
    df["Month"] = df["Month"].astype(int)
    df = df.sort_values(["Station_Name", "Year", "Month"]).reset_index(drop=True)

    stations = sorted(df["Station_Name"].unique())
    sid = {name: i for i, name in enumerate(stations)}
    df["s"] = df["Station_Name"].map(sid)

    # global 0-based month index over 2000-01 .. 2022-12
    df["t"] = (df["Year"] - 2000) * 12 + (df["Month"] - 1)

    def split_of(y):
        if TRAIN_YEARS[0] <= y <= TRAIN_YEARS[1]:
            return "train"
        if VAL_YEARS[0] <= y <= VAL_YEARS[1]:
            return "val"
        return "test"

    df["origin_year"] = df["Year"]
    df["split"] = df["Year"].map(split_of)
    return df, stations, sid, dropped


def to_grid(df, col):
    """Return an (S, T) float array of `col` indexed by [station s, month index t],
    NaN where missing. S=13 stations, T=276 months (2000-01..2022-12)."""
    S = df["s"].nunique()
    T = 276
    g = np.full((S, T), np.nan, dtype=np.float64)
    g[df["s"].to_numpy(), df["t"].to_numpy()] = df[col].to_numpy(dtype=float)
    return g


def train_mask_t(years=TRAIN_YEARS):
    """Boolean length-276 mask of month indices whose calendar year is in `years`."""
    t = np.arange(276)
    yr = 2000 + t // 12
    return (yr >= years[0]) & (yr <= years[1])


def fit_climatology(grid, train_t):
    """Per (station s, calendar month m) mean/std on TRAIN months only, with a robust
    std floor at SD_FLOOR_FRAC * station-pooled train sd (guards saturated/dry near-zero-sd
    cells). grid: (S,T); train_t: bool length T. Returns mu,sd as (S,12)."""
    S, T = grid.shape
    mu = np.full((S, 12), np.nan)
    sd = np.full((S, 12), np.nan)
    months = np.arange(T) % 12
    for s in range(S):
        for m in range(12):
            sel = (months == m) & train_t
            vals = grid[s, sel]
            vals = vals[~np.isnan(vals)]
            if len(vals) >= 2:
                mu[s, m] = vals.mean()
                sd[s, m] = vals.std(ddof=1)
    glob = np.nanstd(grid[:, train_t])
    for s in range(S):
        stat_sd = np.nanstd(grid[s, train_t])
        floor = SD_FLOOR_FRAC * stat_sd if np.isfinite(stat_sd) and stat_sd > 1e-6 else glob
        floor = max(floor, 1e-6)
        for m in range(12):
            if not np.isfinite(sd[s, m]) or sd[s, m] < floor:
                sd[s, m] = floor
            if not np.isfinite(mu[s, m]):
                mu[s, m] = np.nanmean(grid[s, train_t])
    return mu, sd


def standardize(grid, mu, sd):
    """z_v(s,t) = clip( (x - mu[s,m]) / sd[s,m], -Z_CLIP, +Z_CLIP ), m = t%12. NaNs propagate."""
    S, T = grid.shape
    months = np.arange(T) % 12
    z = (grid - mu[:, months]) / sd[:, months]
    return np.clip(z, -Z_CLIP, Z_CLIP)
