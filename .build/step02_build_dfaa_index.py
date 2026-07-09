"""Step 2 - composite wetness W, DFAA index (Eq. 2), event labels, climatology of events.

Implements EXPERIMENT_DESIGN.md Step 2 with the locked disambiguation:
  * symmetric window scale = lead h in {1,2,3}
  * W_E(s,t,h) = mean W over [t-h+1 .. t]   (known at origin t)
  * W_L(s,t,h) = mean W over [t+1 .. t+h]   (the forecast-unknown future)
  * DFAA(s,t,h) = (W_L - W_E)(|W_E|+|W_L|) * alpha^(-|W_E+W_L|),  alpha=1.8
  * target modeled directly is DFAA(s,t,h)
  * theta_D(h) = 80th pct of |DFAA| over TRAIN origins only
Persists artefacts/dfaa.npz + artefacts/dfaa_meta.json. Not yet a forecast (no skill metric);
this builds the label/target the later experiments predict.
"""
import json
import numpy as np
import pandas as pd
from common import (ROOT, ART, load_clean, to_grid, train_mask_t, fit_climatology,
                    standardize, VARS_Z, W_WEIGHTS, ALPHA_DFAA, LEADS, THETA_PCT,
                    TRAIN_YEARS, SEED)

np.random.seed(SEED)
print("=" * 70)
print("STEP 2 - composite wetness W + DFAA index + event labels")
print("=" * 70)

df, stations, sid, _ = load_clean()
S, T = len(stations), 276
train_t = train_mask_t()

# ---- composite wetness W(s,t) = w1 z_P + w2 z_SM + w3 SPEI (train climatology) ----
zP = standardize(to_grid(df, "Rainfall_mm"), *fit_climatology(to_grid(df, "Rainfall_mm"), train_t))
zSM = standardize(to_grid(df, "Soil_moisture_mm"), *fit_climatology(to_grid(df, "Soil_moisture_mm"), train_t))
SPEI = to_grid(df, "SPEI_3")
w1, w2, w3 = W_WEIGHTS
W = w1 * zP + w2 * zSM + w3 * SPEI                      # (S,T), NaN where any input missing
print(f"\nComposite W: weights {W_WEIGHTS}")
print(f"  W finite cells: {np.isfinite(W).sum()} / {S*T}")
print(f"  W (all):  mean={np.nanmean(W):+.3f}  std={np.nanstd(W):.3f}  "
      f"min={np.nanmin(W):+.2f}  max={np.nanmax(W):+.2f}")
print(f"  W (train) mean={np.nanmean(W[:, train_t]):+.3f}  std={np.nanstd(W[:, train_t]):.3f}")
print(f"  component corr on train: "
      f"corr(zP,SPEI)={np.corrcoef(zP[:,train_t][np.isfinite(zP[:,train_t]*SPEI[:,train_t])], SPEI[:,train_t][np.isfinite(zP[:,train_t]*SPEI[:,train_t])])[0,1]:+.2f}  "
      f"corr(zSM,SPEI)={np.corrcoef(zSM[:,train_t][np.isfinite(zSM[:,train_t]*SPEI[:,train_t])], SPEI[:,train_t][np.isfinite(zSM[:,train_t]*SPEI[:,train_t])])[0,1]:+.2f}")


def window_mean(arr, s, lo, hi):
    """mean of arr[s, lo..hi] inclusive; NaN if out of range or any element missing."""
    if lo < 0 or hi >= T:
        return np.nan
    seg = arr[s, lo:hi + 1]
    if np.any(~np.isfinite(seg)):
        return np.nan
    return float(seg.mean())


# ---- DFAA per lead ----
DFAA = {h: np.full((S, T), np.nan) for h in LEADS}
W_E = {h: np.full((S, T), np.nan) for h in LEADS}
W_L = {h: np.full((S, T), np.nan) for h in LEADS}
yr = 2000 + np.arange(T) // 12

for h in LEADS:
    for s in range(S):
        for t in range(T):
            we = window_mean(W, s, t - h + 1, t)          # early (known)
            wl = window_mean(W, s, t + 1, t + h)           # late (future)
            if not (np.isfinite(we) and np.isfinite(wl)):
                continue
            W_E[h][s, t] = we
            W_L[h][s, t] = wl
            flip = wl - we
            mag = abs(we) + abs(wl)
            supp = ALPHA_DFAA ** (-abs(we + wl))
            DFAA[h][s, t] = flip * mag * supp

# ---- event thresholds theta_D(h) on TRAIN origins only ----
meta = {"weights": list(W_WEIGHTS), "alpha": ALPHA_DFAA, "leads": list(LEADS),
        "theta_pct": THETA_PCT, "theta_D": {}, "stations": stations,
        "window": "symmetric: E=[t-h+1..t], L=[t+1..t+h]"}
print("\nPer-lead DFAA distribution, threshold theta_D, and event counts:")
print(f"{'h':>2} {'n_valid':>8} {'n_train':>8} {'DFAA std':>9} {'|DFAA|max':>10} "
      f"{'theta_D':>8} {'DTF_all':>8} {'FTD_all':>8} {'evt%':>6}")
ev_labels = {}   # h -> (S,T) in {-1,0,+1}; +1=DTF, -1=FTD, 0=none, NaN where no DFAA
for h in LEADS:
    d = DFAA[h]
    valid = np.isfinite(d)
    train_cells = valid & train_t[None, :]
    theta = float(np.percentile(np.abs(d[train_cells]), THETA_PCT))
    meta["theta_D"][str(h)] = theta
    lab = np.full((S, T), np.nan)
    lab[valid] = 0.0
    lab[valid & (d >= theta)] = 1.0     # DTF (drought->flood)
    lab[valid & (d <= -theta)] = -1.0   # FTD (flood->drought)
    ev_labels[h] = lab
    n_dtf = int(np.nansum(lab == 1.0))
    n_ftd = int(np.nansum(lab == -1.0))
    n_valid = int(valid.sum())
    print(f"{h:>2} {n_valid:>8} {int(train_cells.sum()):>8} {np.nanstd(d):>9.3f} "
          f"{np.nanmax(np.abs(d)):>10.3f} {theta:>8.3f} {n_dtf:>8} {n_ftd:>8} "
          f"{100*(n_dtf+n_ftd)/n_valid:>5.1f}%")

# ---- event counts by split (h=2 headline) + per-station hotspots ----
def split_mask(name):
    if name == "train":
        return (yr >= 2000) & (yr <= 2014)
    if name == "val":
        return (yr >= 2015) & (yr <= 2017)
    return (yr >= 2018) & (yr <= 2022)

print("\nEvent counts by split (DTF / FTD), per lead:")
for h in LEADS:
    lab = ev_labels[h]
    row = []
    for sp in ["train", "val", "test"]:
        m = split_mask(sp)[None, :] & np.isfinite(lab)
        row.append(f"{sp}: DTF={int(np.nansum((lab==1)&m))} FTD={int(np.nansum((lab==-1)&m))}")
    print(f"  h={h}  " + " | ".join(row))

print("\nPer-station event frequency (h=2, all years) - hotspot check:")
lab2 = ev_labels[2]
for s in range(S):
    v = np.isfinite(lab2[s])
    if v.sum() == 0:
        continue
    dtf = int(np.nansum(lab2[s] == 1)); ftd = int(np.nansum(lab2[s] == -1))
    print(f"  {stations[s]:22s} n={int(v.sum()):3d}  DTF={dtf:3d}  FTD={ftd:3d}  "
          f"event%={100*(dtf+ftd)/v.sum():4.1f}")

# ---- persist ----
np.savez_compressed(
    ART / "dfaa.npz",
    W=W, zP=zP, zSM=zSM, SPEI=SPEI,
    **{f"DFAA_h{h}": DFAA[h] for h in LEADS},
    **{f"WE_h{h}": W_E[h] for h in LEADS},
    **{f"WL_h{h}": W_L[h] for h in LEADS},
    **{f"lab_h{h}": ev_labels[h] for h in LEADS},
    stations=np.array(stations), yr=yr,
)
(ART / "dfaa_meta.json").write_text(json.dumps(meta, indent=2))
print(f"\nWrote {ART/'dfaa.npz'} and {ART/'dfaa_meta.json'}")
print("\nSTEP 2 OK.")
