"""Step 1 - clean, profile, gap report, train-only climatology fit.

Reads raw CSV (never edits it), drops trailing NaN row, asserts grid shape,
reports gaps + distributions, fits per-(station,calendar-month) mean/std on TRAIN
years (2000-2014) only for {Rainfall_mm, Soil_moisture_mm}, persists artefacts/clima.json.

NOT an experiment (no paper number); data prep only.
"""
import json
import numpy as np
import pandas as pd
from common import (ROOT, ART, load_clean, to_grid, train_mask_t, fit_climatology,
                    standardize, VARS_Z, TRAIN_YEARS, VAL_YEARS, TEST_YEARS, SEED)

np.random.seed(SEED)
print("=" * 70)
print("STEP 1 - clean / profile / climatology  (train-only fits)")
print("=" * 70)

df, stations, sid, dropped = load_clean()
print(f"\nDropped non-data rows (trailing NaN): {dropped}")
print(f"Clean rows: {len(df)}  | stations: {len(stations)}")
print("Stations:", stations)

# ---- grid + gap report ----
S, T = len(stations), 276
print(f"\nExpected full grid: {S} x {T} = {S*T} station-months")
present = np.zeros((S, T), dtype=bool)
present[df["s"].to_numpy(), df["t"].to_numpy()] = True
n_missing = int((~present).sum())
print(f"Present: {present.sum()}  | Missing cells: {n_missing}")
miss_by_station = {stations[s]: int((~present[s]).sum()) for s in range(S) if (~present[s]).any()}
print("Missing by station:", miss_by_station)
for s in range(S):
    gaps = np.where(~present[s])[0]
    for t in gaps:
        print(f"   GAP: {stations[s]}  {2000 + t//12}-{t%12+1:02d}  (t={t})")

# ---- per-variable profile (full record) ----
print("\nVariable profile (full record):")
for c in ["Max_Temp", "Min_Temp", "Rainfall_mm", "Soil_moisture_mm", "SPEI_3"]:
    x = df[c].to_numpy(float)
    print(f"  {c:18s} n={np.isfinite(x).sum():4d}  min={np.nanmin(x):8.2f}  "
          f"med={np.nanmedian(x):8.2f}  mean={np.nanmean(x):8.2f}  max={np.nanmax(x):8.2f}  "
          f"nan={np.isnan(x).sum()}")

# tail context quoted in EXPERIMENT_DESIGN (sanity)
rain = df["Rainfall_mm"].to_numpy(float)
spei = df["SPEI_3"].to_numpy(float)
print(f"\n  station-months with Rainfall > 500 mm : {(rain > 500).sum()}")
print(f"  station-months SPEI_3 <= -1 (moderate+ drought): {(spei <= -1).sum()} "
      f"({100*(spei <= -1).mean():.1f}%)")
print(f"  station-months SPEI_3 >=  1 (moderate+ wet)    : {(spei >= 1).sum()} "
      f"({100*(spei >= 1).mean():.1f}%)")

# ---- split sizes (by origin year) ----
print("\nSplit sizes (by calendar year of the month):")
print(df["split"].value_counts().reindex(["train", "val", "test"]).to_string())
print(f"  train years {TRAIN_YEARS}, val {VAL_YEARS}, test {TEST_YEARS}")

# ---- train-only climatology for z-vars ----
train_t = train_mask_t()
clima = {"vars": VARS_Z, "train_years": list(TRAIN_YEARS), "stations": stations,
         "mu": {}, "sd": {}}
for v in VARS_Z:
    grid = to_grid(df, v)
    mu, sd = fit_climatology(grid, train_t)
    clima["mu"][v] = mu.tolist()
    clima["sd"][v] = sd.tolist()
    z = standardize(grid, mu, sd)
    # report standardized anomaly sanity on TRAIN (should be ~0 mean, ~1 std)
    zt = z[:, train_t]
    print(f"\n  {v}: train-z mean={np.nanmean(zt):+.3f} std={np.nanstd(zt):.3f} "
          f"(target ~0, ~1)  | mu range [{mu.min():.1f},{mu.max():.1f}] "
          f"sd range [{sd.min():.2f},{sd.max():.2f}]")

out = ART / "clima.json"
out.write_text(json.dumps(clima, indent=2))
print(f"\nWrote {out}")
print("\nSTEP 1 OK.")
