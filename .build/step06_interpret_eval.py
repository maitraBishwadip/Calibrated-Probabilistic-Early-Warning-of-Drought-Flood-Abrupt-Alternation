"""Step 6 - interpretation & evaluation.

A. Lead-time skill collation (CRPSS vs h: GBQ / bucket / LSTM).
B. Physics-guided feature ABLATION (GBQ full vs no-physics) - the de-risking decision.
C. Permutation importance (GBQ, pinball-based) - driver attribution.
D. Cost-loss decision value V(r) (GBQ+CQR event probs) - the adaptation/services lens.
E. LOSO transfer (GBQ, h=2) - per-held-out-station skill ("ungauged" angle).

Writes artefacts/interpret_metrics.json + appends RESULTS_LOG.
"""
import json
import time
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from common import ART, load_clean, train_mask_t, LEADS, TAUS, SEED
import evalkit as ek

np.random.seed(SEED)
print("=" * 70); print("STEP 6 - interpretation & evaluation"); print("=" * 70)

df, stations, sid, _ = load_clean()
S, T = len(stations), 276
train_t = train_mask_t()
yr = 2000 + np.arange(T) // 12

fz = np.load(ART / "features.npz", allow_pickle=True)
feat = fz["feat"].astype(np.float64); fnames = list(fz["names"]); Fdim = feat.shape[-1]
dz = np.load(ART / "dfaa.npz", allow_pickle=True)
DFAA = {h: dz[f"DFAA_h{h}"] for h in LEADS}; WE = {h: dz[f"WE_h{h}"] for h in LEADS}
meta = json.loads((ART / "dfaa_meta.json").read_text()); THETA = {h: meta["theta_D"][str(h)] for h in LEADS}
base_m = json.loads((ART / "baseline_metrics.json").read_text())
model_m = json.loads((ART / "model_metrics.json").read_text())
conf_m = json.loads((ART / "conformal_metrics.json").read_text())

fmu = np.array([np.nanmean(feat[:, train_t, j]) for j in range(Fdim)])
feat_fill = np.where(np.isfinite(feat), feat, fmu[None, None, :])
we_mu = {h: float(np.nanmean(WE[h][:, train_t])) for h in LEADS}
PHYS = ["PET", "D", "aridity", "API", "dSM", "dSPEI", "roll3", "roll6"]
phys_idx = [fnames.index(n) for n in PHYS]
allcols = list(range(Fdim)) + ["WE"]   # WE appended as last column


def split_of(t):
    y = yr[t]; return "train" if y <= 2014 else ("val" if y <= 2017 else "test")


def build_Xy(h, idx, drop_cols=()):
    rows = []
    for s, t in idx:
        v = list(feat_fill[s, t]) + [WE[h][s, t] if np.isfinite(WE[h][s, t]) else we_mu[h]]
        rows.append(v)
    X = np.array(rows)
    if drop_cols:
        keep = [c for c in range(X.shape[1]) if c not in drop_cols]
        X = X[:, keep]
    y = np.array([DFAA[h][s, t] for s, t in idx])
    return X, y


def origins(h, sp):
    return [(s, t) for s in range(S) for t in range(T) if np.isfinite(DFAA[h][s, t]) and split_of(t) == sp]


def fit_gbq(Xtr, ytr, Xte, n_est=300):
    cols = []
    for tau in TAUS:
        gb = GradientBoostingRegressor(loss="quantile", alpha=float(tau), n_estimators=n_est,
                                       max_depth=3, learning_rate=0.05, subsample=0.8, random_state=SEED)
        gb.fit(Xtr, ytr); cols.append(gb.predict(Xte))
    return np.sort(np.stack(cols, 1), 1)


out = {}

# ---------- A. lead-time skill ----------
out["lead_time"] = {"gbq_crpss": {h: model_m["gbq"][str(h)]["test"]["crpss"] for h in LEADS},
                    "lstm_crpss": {h: model_m["lstm"][str(h)]["test"]["crpss"] for h in LEADS},
                    "bucket_crpss": {h: base_m[str(h)]["test"]["bucket"]["crpss"] for h in LEADS},
                    "gbq_dtf_auc": {h: model_m["gbq"][str(h)]["test"]["event"]["DTF"]["auc"] for h in LEADS},
                    "gbq_ftd_auc": {h: model_m["gbq"][str(h)]["test"]["event"]["FTD"]["auc"] for h in LEADS}}
print("\n[A] Lead-time CRPSS (test): GBQ / bucket / LSTM")
for h in LEADS:
    print(f"  h={h}  GBQ {out['lead_time']['gbq_crpss'][h]:+.3f} | bucket "
          f"{out['lead_time']['bucket_crpss'][h]:+.3f} | LSTM {out['lead_time']['lstm_crpss'][h]:+.3f}")

# ---------- B. physics-feature ablation ----------
print("\n[B] Physics-guided feature ablation (GBQ full vs no-physics):")
out["ablation"] = {}
for h in LEADS:
    tr, te = origins(h, "train"), origins(h, "test")
    ytr = np.array([DFAA[h][s, t] for s, t in tr]); yte = np.array([DFAA[h][s, t] for s, t in te])
    ref = ek.crps(np.tile(np.quantile(ytr, TAUS), (len(yte), 1)), yte)
    Xtr_f, _ = build_Xy(h, tr); Xte_f, _ = build_Xy(h, te)
    Xtr_n, _ = build_Xy(h, tr, drop_cols=phys_idx); Xte_n, _ = build_Xy(h, te, drop_cols=phys_idx)
    Qf = fit_gbq(Xtr_f, ytr, Xte_f); Qn = fit_gbq(Xtr_n, ytr, Xte_n)
    mf = ek.all_metrics(Qf, yte, THETA[h], crps_ref=ref); mn = ek.all_metrics(Qn, yte, THETA[h], crps_ref=ref)
    out["ablation"][h] = {"full": mf, "no_phys": mn}
    print(f"  h={h}  CRPSS full {mf['crpss']:+.3f} vs no-phys {mn['crpss']:+.3f}  (d={mf['crpss']-mn['crpss']:+.3f}) | "
          f"DTF-AUC full {mf['event']['DTF']['auc']:.3f} vs {mn['event']['DTF']['auc']:.3f}")

# ---------- C. permutation importance (GBQ, h=2, pinball-based) ----------
print("\n[C] Permutation importance (GBQ h=2, increase in pinball when feature shuffled):")
h = 2
tr, te = origins(h, "train"), origins(h, "test")
ytr = np.array([DFAA[h][s, t] for s, t in tr]); yte = np.array([DFAA[h][s, t] for s, t in te])
Xtr, _ = build_Xy(h, tr); Xte, _ = build_Xy(h, te)
gbs = []
for tau in TAUS:
    gb = GradientBoostingRegressor(loss="quantile", alpha=float(tau), n_estimators=300, max_depth=3,
                                   learning_rate=0.05, subsample=0.8, random_state=SEED).fit(Xtr, ytr)
    gbs.append(gb)
def pinball_of(X):
    Q = np.sort(np.stack([gb.predict(X) for gb in gbs], 1), 1)
    return ek.pinball(Q, yte)
base_pin = pinball_of(Xte)
colnames = fnames + ["WE"]
rng = np.random.default_rng(SEED)
imp = {}
for j in range(Xte.shape[1]):
    deltas = []
    for _ in range(5):
        Xp = Xte.copy(); Xp[:, j] = rng.permutation(Xp[:, j]); deltas.append(pinball_of(Xp) - base_pin)
    imp[colnames[j]] = float(np.mean(deltas))
ranked = sorted(imp.items(), key=lambda kv: -kv[1])
out["perm_importance_h2"] = dict(ranked)
for name, val in ranked[:10]:
    tag = " [physics]" if name in PHYS else ""
    print(f"   {name:10s} Δpinball={val:+.5f}{tag}")

# ---------- D. cost-loss value (GBQ+CQR event probs, h=2) ----------
print("\n[D] Cost-loss decision value V(r) (GBQ+CQR, h=2):")
pp = np.load(ART / "model_preds.npz", allow_pickle=True)
Qcal = pp["gbq_2_val_Q"]; ycal = pp["gbq_2_val_y"]; Qte = pp["gbq_2_test_Q"]; yte2 = pp["gbq_2_test_y"]
# apply CQR (reuse Step 5 logic: widen 90/80/50 pairs)
PAIRS = [(0, 6, 0.10), (1, 5, 0.20), (2, 4, 0.50)]
Qc = Qte.copy()
for lo_i, hi_i, a in PAIRS:
    E = np.maximum(Qcal[:, lo_i] - ycal, ycal - Qcal[:, hi_i])
    k = min(int(np.ceil((len(E) + 1) * (1 - a))), len(E)); e = np.sort(E)[k - 1]
    Qc[:, lo_i] -= e; Qc[:, hi_i] += e
Qc = np.sort(Qc, 1)
th = THETA[2]
p_dtf = 1 - ek.cdf_at(Qc, th); occ_dtf = (yte2 >= th).astype(float)
p_ftd = ek.cdf_at(Qc, -th); occ_ftd = (yte2 <= -th).astype(float)
p_clim_dtf = np.full_like(p_dtf, occ_dtf.mean()); p_clim_ftd = np.full_like(p_ftd, occ_ftd.mean())


def cost_loss_curve(p, occ):
    rs = np.linspace(0.02, 0.98, 49); s = occ.mean(); vals = []
    for r in rs:
        act = (p > r).astype(float)
        E_fc = np.mean(act * r + (1 - act) * occ)
        E_clim = min(r, s); E_perf = s * r; den = E_clim - E_perf
        vals.append((E_clim - E_fc) / den if den > 1e-9 else np.nan)
    return rs, np.array(vals)


out["cost_loss_h2"] = {}
for nm, p, occ in [("DTF", p_dtf, occ_dtf), ("FTD", p_ftd, occ_ftd)]:
    rs, v = cost_loss_curve(p, occ)
    vmax = float(np.nanmax(v)); rstar = float(rs[np.nanargmax(v)])
    out["cost_loss_h2"][nm] = {"r": rs.tolist(), "value": v.tolist(), "vmax": vmax, "r_at_vmax": rstar,
                               "base_rate": float(occ.mean())}
    print(f"   {nm}: base rate {occ.mean():.3f}  max value V={vmax:.3f} at C/L={rstar:.2f}  "
          f"(positive V over r in [{rs[np.where(v>0)[0][0]] if np.any(v>0) else float('nan'):.2f}, "
          f"{rs[np.where(v>0)[0][-1]] if np.any(v>0) else float('nan'):.2f}])")

# ---------- E. LOSO transfer (GBQ, h=2) ----------
print("\n[E] LOSO transfer (GBQ h=2): train on 12 stations, test held-out station (2018-22)...")
t0 = time.time(); out["loso_h2"] = {}
for s_out in range(S):
    tr = [(s, t) for (s, t) in origins(2, "train") if s != s_out]
    te = [(s, t) for (s, t) in origins(2, "test") if s == s_out]
    if len(te) < 10:
        continue
    Xtr, ytr = build_Xy(2, tr); Xte, yte_s = build_Xy(2, te)
    Q = fit_gbq(Xtr, ytr, Xte, n_est=200)
    ref = ek.crps(np.tile(np.quantile(ytr, TAUS), (len(yte_s), 1)), yte_s)
    m = ek.all_metrics(Q, yte_s, THETA[2], crps_ref=ref)
    out["loso_h2"][stations[s_out]] = {"crpss": m["crpss"], "crps": m["crps"],
                                       "dtf_auc": m["event"]["DTF"]["auc"], "ftd_auc": m["event"]["FTD"]["auc"]}
    print(f"   {stations[s_out]:22s} CRPSS {m['crpss']:+.3f}  DTF-AUC {m['event']['DTF']['auc']:.3f}  "
          f"FTD-AUC {m['event']['FTD']['auc']:.3f}")
cr = [v["crpss"] for v in out["loso_h2"].values()]
print(f"   LOSO CRPSS mean {np.mean(cr):+.3f}  (min {np.min(cr):+.3f}, max {np.max(cr):+.3f})  in {time.time()-t0:.0f}s")

(ART / "interpret_metrics.json").write_text(json.dumps(out, indent=2, default=float))
print(f"\nWrote {ART/'interpret_metrics.json'}")
print("\nSTEP 6 OK.")
