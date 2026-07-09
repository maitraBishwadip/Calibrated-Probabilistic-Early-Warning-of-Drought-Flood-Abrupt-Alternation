"""Step 3 - PET (Hargreaves), physics-guided features, and probabilistic baselines.

Baselines (all predict the predictive distribution of DFAA(s,t,h) directly, so they are
CRPS/pinball-comparable; spreads from TRAIN residual quantiles -> no leakage):
  * climatology       : unconditional train DFAA quantiles  (CRPS reference)
  * seasonal-clim     : train DFAA quantiles per origin calendar month
  * persistence       : point = DFAA(s,t-h,h) + train residual quantiles
  * water-balance bucket : roll Eq. 4 forward h months (climatological forcing) -> DFAA point + train resid
  * SARIMA            : per-station SARIMA on W -> h-step W_L point -> DFAA point + train resid

Writes artefacts/features.npz (for Step 4) and artefacts/baseline_metrics.json; appends to RESULTS_LOG.
"""
import json
import time
import warnings
import numpy as np
from common import (ROOT, ART, load_clean, to_grid, train_mask_t, fit_climatology, standardize,
                    STATION_LATLON, ALPHA_DFAA, LEADS, TAUS, SEED)
import evalkit as ek

warnings.filterwarnings("ignore")
np.random.seed(SEED)
print("=" * 70); print("STEP 3 - PET + physics features + baselines"); print("=" * 70)

df, stations, sid, _ = load_clean()
S, T = len(stations), 276
train_t = train_mask_t()
yr = 2000 + np.arange(T) // 12
mo = np.arange(T) % 12  # 0-based calendar month

z = np.load(ART / "dfaa.npz", allow_pickle=True)
W = z["W"]
DFAA = {h: z[f"DFAA_h{h}"] for h in LEADS}
WE = {h: z[f"WE_h{h}"] for h in LEADS}
meta = json.loads((ART / "dfaa_meta.json").read_text())
THETA = {h: meta["theta_D"][str(h)] for h in LEADS}

P = to_grid(df, "Rainfall_mm")
SM = to_grid(df, "Soil_moisture_mm")
Tx = to_grid(df, "Max_Temp")
Tn = to_grid(df, "Min_Temp")
SPEI = to_grid(df, "SPEI_3")

# ---------------- PET (Hargreaves-Samani) ----------------
DAYS = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
MIDDOY = np.array([15, 45, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349])  # mid-month day-of-year


def Ra_mm_per_day(lat_deg, J):
    """Extraterrestrial radiation in mm/day equivalent (FAO-56 Eq. 21-28)."""
    phi = np.radians(lat_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi * J / 365)
    dec = 0.409 * np.sin(2 * np.pi * J / 365 - 1.39)
    x = np.clip(-np.tan(phi) * np.tan(dec), -1, 1)
    ws = np.arccos(x)
    Ra = (24 * 60 / np.pi) * 0.0820 * dr * (ws * np.sin(phi) * np.sin(dec)
                                            + np.cos(phi) * np.cos(dec) * np.sin(ws))
    return Ra * 0.408  # MJ/m2/day -> mm/day


PET = np.full((S, T), np.nan)
for s, name in enumerate(stations):
    lat = STATION_LATLON[name][0]
    for t in range(T):
        m = mo[t]
        Tmean = (Tx[s, t] + Tn[s, t]) / 2.0
        dT = max(Tx[s, t] - Tn[s, t], 0.0)
        pet_day = 0.0023 * Ra_mm_per_day(lat, MIDDOY[m]) * (Tmean + 17.8) * np.sqrt(dT)
        PET[s, t] = max(pet_day, 0.0) * DAYS[m]   # monthly PET mm
print(f"\nPET (Hargreaves) mm/month: mean={np.nanmean(PET):.1f} range[{np.nanmin(PET):.0f},{np.nanmax(PET):.0f}]")
print(f"  Climatic water balance D=P-PET: mean={np.nanmean(P-PET):.1f} "
      f"(monsoon surplus / dry-season deficit expected)")

# ---------------- physics-guided features ----------------
zP = standardize(P, *fit_climatology(P, train_t))
zSM = standardize(SM, *fit_climatology(SM, train_t))
D = P - PET
aridity = P / (PET + 1.0)
beta = 0.9
API = np.full((S, T), np.nan)
for s in range(S):
    for t in range(T):
        acc = 0.0; ok = True
        for k in range(1, 7):
            if t - k < 0 or not np.isfinite(P[s, t - k]):
                ok = False; break
            acc += beta ** k * P[s, t - k]
        API[s, t] = acc if ok else np.nan
dSM = np.full((S, T), np.nan); dSM[:, 1:] = SM[:, 1:] - SM[:, :-1]
dSPEI = np.full((S, T), np.nan); dSPEI[:, 1:] = SPEI[:, 1:] - SPEI[:, :-1]
roll3 = np.full((S, T), np.nan); roll6 = np.full((S, T), np.nan)
for s in range(S):
    for t in range(T):
        if t >= 2 and np.all(np.isfinite(P[s, t - 2:t + 1])): roll3[s, t] = P[s, t - 2:t + 1].sum()
        if t >= 5 and np.all(np.isfinite(P[s, t - 5:t + 1])): roll6[s, t] = P[s, t - 5:t + 1].sum()

sinm = np.sin(2 * np.pi * mo / 12)[None, :].repeat(S, 0)
cosm = np.cos(2 * np.pi * mo / 12)[None, :].repeat(S, 0)
lat = np.array([STATION_LATLON[n][0] for n in stations])[:, None].repeat(T, 1)
lon = np.array([STATION_LATLON[n][1] for n in stations])[:, None].repeat(T, 1)
sid_grid = np.arange(S)[:, None].repeat(T, 1).astype(float)

feat_names = ["zP", "zSM", "SPEI", "Tx", "Tn", "PET", "D", "aridity", "API", "dSM",
              "dSPEI", "roll3", "roll6", "W", "sinm", "cosm", "lat", "lon", "sid"]
feat_stack = np.stack([zP, zSM, SPEI, Tx, Tn, PET, D, aridity, API, dSM, dSPEI,
                       roll3, roll6, W, sinm, cosm, lat, lon, sid_grid], axis=-1)  # (S,T,F)
np.savez_compressed(ART / "features.npz", feat=feat_stack, names=np.array(feat_names),
                    PET=PET, yr=yr, mo=mo, stations=np.array(stations))
print(f"Saved features.npz: feat {feat_stack.shape}, {len(feat_names)} features")

# ---------------- water-balance bucket ----------------
Pc_mu, _ = fit_climatology(P, train_t)        # climatological P per (s,m)
PETc_mu, _ = fit_climatology(PET, train_t)    # climatological PET per (s,m)
Smax = np.array([np.nanmax(SM[s, train_t]) for s in range(S)])   # field-capacity proxy (train)
S0 = np.array([np.nanmean(SM[s, train_t]) for s in range(S)])

# actual storage trajectory
Sa = np.full((S, T), np.nan)
for s in range(S):
    st = S0[s]
    for t in range(T):
        if np.isfinite(P[s, t]) and np.isfinite(PET[s, t]):
            g = np.clip(st / Smax[s], 0, 1)
            st = np.clip(st + P[s, t] - PET[s, t] * g, 0, Smax[s])
        Sa[s, t] = st
# standardize storage by its TRAIN climatology -> z_S (soil-wetness proxy)
zS = standardize(Sa, *fit_climatology(Sa, train_t))


def bucket_future_WL(s, t, h):
    """Roll bucket forward h months from actual storage at t using climatological forcing;
    return mean standardized-storage anomaly over [t+1..t+h] as the W_L proxy."""
    st = Sa[s, t]
    if not np.isfinite(st):
        return np.nan
    mu_s, sd_s = fit_clima_cache[s]
    vals = []
    for j in range(1, h + 1):
        tt = t + j
        if tt >= T:
            return np.nan
        m = mo[tt]
        g = np.clip(st / Smax[s], 0, 1)
        st = np.clip(st + Pc_mu[s, m] - PETc_mu[s, m] * g, 0, Smax[s])
        vals.append((st - mu_s[m]) / sd_s[m])
    return float(np.mean(vals))


# cache per-station climatology of Sa for standardizing the rolled storage
fit_clima_cache = {}
_mu_all, _sd_all = fit_climatology(Sa, train_t)
for s in range(S):
    fit_clima_cache[s] = (_mu_all[s], _sd_all[s])


def dfaa_from(we, wl):
    if not (np.isfinite(we) and np.isfinite(wl)):
        return np.nan
    return (wl - we) * (abs(we) + abs(wl)) * ALPHA_DFAA ** (-abs(we + wl))


# ---------------- SARIMA per station on W ----------------
def sarima_WL():
    """Per-station SARIMA(1,0,0)(1,0,0,12) on W; returns dict h-> (S,T) point W_L forecast."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    out = {h: np.full((S, T), np.nan) for h in LEADS}
    t0_val = (2015 - 2000) * 12  # first val origin month index
    for s in range(S):
        ser = W[s].astype(float)
        tr = ser[train_t].copy()
        if np.any(~np.isfinite(tr)):
            tr = np.where(np.isfinite(tr), tr, np.nanmean(tr))
        try:
            res_tr = SARIMAX(tr, order=(1, 0, 0), seasonal_order=(1, 0, 0, 12),
                             enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        except Exception as e:
            print(f"   SARIMA fit failed station {stations[s]}: {e}"); continue
        params = res_tr.params
        full = np.where(np.isfinite(ser), ser, np.nan)
        # forecast h-step from each origin t>=t0_val-1, filtering full series with train params
        for t in range(t0_val - max(LEADS) - 1, T):
            hist = full[:t + 1]
            if np.any(~np.isfinite(hist)):
                hist = np.where(np.isfinite(hist), hist, np.nanmean(hist[np.isfinite(hist)]))
            try:
                r = res_tr.apply(hist, refit=False)
                fc = r.forecast(max(LEADS))
            except Exception:
                continue
            for h in LEADS:
                if t + h < T:
                    out[h][s, t] = float(np.mean(fc[:h]))
    return out


print("\nFitting SARIMA per station (this is the slow baseline)...")
t0 = time.time()
SAR_WL = sarima_WL()
print(f"  SARIMA done in {time.time()-t0:.0f}s")

# ---------------- assemble baselines & evaluate ----------------
def split_of(t):
    y = yr[t]
    if y <= 2014: return "train"
    if y <= 2017: return "val"
    return "test"


results = {}
for h in LEADS:
    d = DFAA[h]
    # origins with valid target
    origins = [(s, t) for s in range(S) for t in range(T) if np.isfinite(d[s, t])]
    spl = {sp: np.array([(s, t) for (s, t) in origins if split_of(t) == sp]) for sp in ["train", "val", "test"]}
    y_tr = np.array([d[s, t] for s, t in spl["train"]])

    # --- climatology (unconditional) ---
    clim_q = np.quantile(y_tr, TAUS)
    # --- seasonal climatology (per origin month) ---
    seas_q = {}
    for m in range(12):
        ys = np.array([d[s, t] for s, t in spl["train"] if mo[t] == m])
        seas_q[m] = np.quantile(ys, TAUS) if len(ys) >= 10 else clim_q
    # --- persistence point + train residuals ---
    def persist_point(s, t):
        return d[s, t - h] if t - h >= 0 and np.isfinite(d[s, t - h]) else np.nan
    pp_tr = np.array([persist_point(s, t) for s, t in spl["train"]])
    res_p = y_tr - pp_tr
    rq_p = ek.residual_quantiles(res_p[np.isfinite(res_p)])
    # --- bucket point + train residuals ---
    def bucket_point(s, t):
        return dfaa_from(WE[h][s, t], bucket_future_WL(s, t, h))
    bp_tr = np.array([bucket_point(s, t) for s, t in spl["train"]])
    res_b = y_tr - bp_tr
    rq_b = ek.residual_quantiles(res_b[np.isfinite(res_b)])
    # --- sarima point + train residuals ---
    def sarima_point(s, t):
        wl = SAR_WL[h][s, t]
        return dfaa_from(WE[h][s, t], wl)
    sp_tr = np.array([sarima_point(s, t) for s, t in spl["train"]])
    res_s = y_tr - sp_tr
    rq_s = ek.residual_quantiles(res_s[np.isfinite(res_s)]) if np.isfinite(res_s).sum() > 20 else None

    results[h] = {}
    crps_ref = {}
    for sp in ["val", "test"]:
        idx = spl[sp]
        y = np.array([d[s, t] for s, t in idx])
        # build Q per method (fill non-finite point with climatology median)
        def pq(points, rq):
            pts = np.array(points)
            med = clim_q[3]
            pts = np.where(np.isfinite(pts), pts, med)
            return ek.point_to_quantiles(pts, rq)
        Q_clim = np.tile(clim_q, (len(idx), 1))
        Q_seas = np.array([seas_q[mo[t]] for s, t in idx])
        Q_pers = pq([persist_point(s, t) for s, t in idx], rq_p)
        Q_buck = pq([bucket_point(s, t) for s, t in idx], rq_b)
        ref = ek.crps(Q_clim, y)
        crps_ref[sp] = ref
        methods = {"climatology": Q_clim, "seasonal_clim": Q_seas,
                   "persistence": Q_pers, "bucket": Q_buck}
        if rq_s is not None:
            methods["sarima"] = pq([sarima_point(s, t) for s, t in idx], rq_s)
        results[h][sp] = {name: ek.all_metrics(Q, y, THETA[h], crps_ref=ref) for name, Q in methods.items()}

(ART / "baseline_metrics.json").write_text(json.dumps(results, indent=2))

# ---------------- print summary ----------------
print("\n" + "=" * 70)
print("BASELINE SUMMARY (TEST split) - CRPS / CRPSS / PICP90 / DTF-AUC / FTD-AUC")
print("=" * 70)
for h in LEADS:
    print(f"\n-- lead h={h}  (theta_D={THETA[h]:.3f}) --")
    print(f"{'method':16s} {'CRPS':>7s} {'CRPSS':>7s} {'PICP80':>7s} {'PICP90':>7s} "
          f"{'pinball':>8s} {'DTF-AUC':>8s} {'FTD-AUC':>8s}")
    for name, m in results[h]["test"].items():
        ev = m["event"]
        print(f"{name:16s} {m['crps']:7.4f} {m.get('crpss', float('nan')):7.3f} "
              f"{m['picp80']:7.3f} {m['picp90']:7.3f} {m['pinball']:8.4f} "
              f"{ev['DTF']['auc']:8.3f} {ev['FTD']['auc']:8.3f}")
print(f"\nWrote {ART/'baseline_metrics.json'}")
print("\nSTEP 3 OK.")
