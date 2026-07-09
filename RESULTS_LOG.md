# RESULTS_LOG.md — source of truth (append-only)

Every run that produces a number the paper could cite is logged here, one block per run.
The paper reads **only** from this file. Negatives are recorded plainly (CLAUDE.md Rule 1, §2).

Environment (recorded once; re-noted if it changes):
`Python 3.14.2 · numpy 2.4.0 · pandas 2.3.3 · scipy 1.17.1 · scikit-learn 1.8.0 ·
statsmodels 0.14.6 · torch 2.11.0+cpu (CPU)`. No ngboost/xgboost installed →
gradient-boosting quantile learner is **scikit-learn `GradientBoostingRegressor(loss="quantile")`**.

Locked defaults in force (EXPERIMENT_DESIGN.md): seed=0; composite weights 1/3·z_P + 1/3·z_SM + 1/3·SPEI;
α=1.8; symmetric window scale = lead h ∈ {1,2,3}; τ-grid {.05,.10,.25,.50,.75,.90,.95};
θ_D = 80th pct of |DFAA| on **train origins**; split train 2000–2014 / val 2015–2017 / test 2018–2022;
all climatology/θ/scalers/conformal fit on **train only**.

---

## run_id: S1 — data clean + train-only climatology (data prep, not a paper metric)
- **date:** 2026-06-18 · **script:** `.build/step01_explore_clean.py` v1 → `artefacts/clima.json`
- Dropped 1 trailing NaN-station row → **3587 clean station-months** (13 stations × 276 months − 1 gap).
- **Gap:** Khulna **2018-02** missing (the single grid hole; flagged, not imputed).
- Profile (full record): Rainfall mean 186.9 mm (max 1573); Soil_moisture mean 157.2 mm; SPEI_3 mean −0.17 (min −3.58, max 3.07); Max_Temp mean 30.9 °C; Min_Temp mean 21.5 °C. No NaNs in value columns.
- **Tail sanity (matches EXPERIMENT_DESIGN context):** 342 station-months with Rainfall > 500 mm; 692 (19.3%) with SPEI_3 ≤ −1 (moderate+ drought); 352 (9.8%) with SPEI_3 ≥ +1.
- Split sizes (by month's calendar year): train **2340**, val **468**, test **779** station-months.
- Train-only climatology fit per (station, calendar-month) for {Rainfall, Soil_moisture}; standardized train anomalies mean ≈0, std 0.97 (rain) / 0.86 (soil moisture; slightly <1 because the zero-variance-cell guard substitutes a station-wide σ for a few constant cells). SPEI used as-is.
- **Caveat:** station lat/lon table (for PET Ra) is standard BMD coordinates, flagged for final verification before .tex (Rule 3). EO provenance of each column still to be pinned (Rule 3).
- **Robust standardization (added after a blow-up was found):** raw per-(s,m) z exploded to ~−39618 because soil moisture **saturates at field capacity in monsoon months** (e.g. Rajshahi Sep train σ=0.0006 mm) and dry-month rainfall has near-zero σ. Fix: per-(s,m) σ floored at `0.15 × station-pooled train σ`, then z clipped to ±4. Both fixed/train-only transforms (no leakage); train cells (|z|<3.6) untouched. Post-fix train-z std 0.87 (rain) / 0.79 (soil). Sensitivity to the floor fraction to be reported as robustness.

---

## run_id: S2 — composite wetness W + DFAA index + event labels (target build, not yet a skill metric)
- **date:** 2026-06-18 · **script:** `.build/step02_build_dfaa_index.py` v1 → `artefacts/dfaa.npz`, `artefacts/dfaa_meta.json`
- **Disambiguation locked (mine, within approved design):** symmetric window scale = lead h; `W_E`=mean W over [t−h+1..t] (known at origin t), `W_L`=mean W over [t+1..t+h] (future); **target = DFAA(s,t,h) modeled directly** (Eq. 2) with the monotone quantile head → valid non-crossing CDF for event probs. α=1.8, equal weights.
- **Composite W** = ⅓z_P+⅓z_SM+⅓SPEI: finite on 3587/3588 cells, all-record range [−2.83, +3.39], std 0.68. Train-component corr: corr(z_P,SPEI)=+0.25, corr(z_SM,SPEI)=+0.41 (all positive; SPEI tracks soil moisture more than rainfall — physically sensible).
- **DFAA distribution & θ_D (80th pct |DFAA| on train):** h1 σ=0.59, θ=0.535; h2 σ=0.47, θ=0.415; h3 σ=0.49, θ=0.382. |DFAA|max 3.5–4.4.
- **Event counts (DTF / FTD), train·val·test:** h2 — train 224/242, val 67/52, test 127/127. DTF and FTD roughly balanced overall.
- **⚠ NOTABLE — non-stationarity / train-test shift:** with θ_D fixed on train (→20.0% train event rate by construction), the **test period 2018–2022 has ~33% event rate** (h2) vs 20% train, i.e. abrupt alternations are **more frequent/intense in recent years**. Real finding (whiplash intensification) *and* a modeling challenge (test harder than train; covariate/label shift). Flagged for interpretation; may motivate reporting skill on a per-period basis.
- **Spatial hotspots (h2 event %):** highest **Faridpur 29.3, Barisal 28.6, Dhaka 25.6, Comilla/Khulna 25.3**; lowest **Cox's Bazar 19.0, Mymensingh 19.8, Rangpur 21.2**. NB: this differs from the design's a-priori "NE haor/Sylhet vs NW" guess (Sylhet is mid-pack at 21.6) — reported as measured, not as expected.

---

## run_id: S3 — PET + physics-guided features + probabilistic baselines
- **date:** 2026-06-19 · **script:** `.build/step03_features_baselines.py` v1 → `artefacts/features.npz`, `artefacts/baseline_metrics.json`
- **PET (Hargreaves):** mean 124.5 mm/month (range 72–204); climatic water balance D=P−PET mean +62.4 mm (net annual surplus, monsoon-driven — physically sensible). 19-feature table saved for Step 4.
- **Baselines** all predict the predictive distribution of DFAA(s,t,h) directly; point methods probabilized with **train residual quantiles** (no leakage). Reference for CRPSS = unconditional train **climatology**.
- **TEST metrics** (CRPS / CRPSS / PICP80 / PICP90 / DTF-AUC / FTD-AUC):

| h | method | CRPS | CRPSS | PICP80 | PICP90 | DTF-AUC | FTD-AUC |
|--|--|--|--|--|--|--|--|
| 1 | climatology | 0.248 | 0.000 | 0.787 | 0.881 | 0.500 | 0.500 |
| 1 | seasonal_clim | 0.250 | −0.007 | 0.761 | 0.871 | 0.579 | 0.544 |
| 1 | persistence | 0.408 | −0.642 | 0.793 | 0.894 | 0.393 | 0.331 |
| 1 | **bucket** | 0.320 | −0.287 | 0.720 | 0.848 | **0.646** | **0.744** |
| 1 | sarima | 0.397 | −0.597 | 0.278 | 0.363 | 0.513 | 0.764 |
| 2 | climatology | 0.267 | 0.000 | 0.667 | 0.798 | 0.500 | 0.500 |
| 2 | seasonal_clim | 0.263 | +0.014 | 0.660 | 0.768 | 0.481 | 0.668 |
| 2 | persistence | 0.382 | −0.430 | 0.691 | 0.830 | 0.476 | 0.476 |
| 2 | **bucket** | 0.302 | −0.130 | 0.630 | 0.758 | **0.732** | **0.745** |
| 2 | sarima | 0.336 | −0.261 | 0.364 | 0.602 | 0.677 | 0.764 |
| 3 | climatology | 0.362 | 0.000 | 0.569 | 0.688 | 0.500 | 0.500 |
| 3 | seasonal_clim | 0.357 | +0.013 | 0.566 | 0.692 | 0.527 | 0.607 |
| 3 | persistence | 0.561 | −0.551 | 0.550 | 0.678 | 0.401 | 0.417 |
| 3 | **bucket** | 0.336 | **+0.072** | 0.541 | 0.697 | **0.786** | **0.769** |
| 3 | sarima | 0.373 | −0.031 | 0.311 | 0.493 | 0.705 | 0.780 |

- **Reading:** (i) **bucket = strong event detector** (AUC 0.65→0.79 rising with lead) — the physics bar the ML model must beat; (ii) on full-distribution **CRPS, climatology is hard to beat** at short leads (bucket only wins CRPS at h=3); (iii) **persistence is anti-skillful** (AUC<0.5 → DFAA reverts; negative lag-h autocorrelation — an honest property of the alternation index); (iv) **SARIMA under-covers badly** (PICP90 0.36–0.60, intervals too narrow) though it ranks FTD well; (v) **train-fit intervals under-cover test** and worsen with lead (climatology PICP90 0.88→0.69) → confirms the 2018–22 non-stationarity and **motivates the conformal step**.
- **Caveat:** CRPS is the 7-τ quantile estimator (coarse but identical across methods). SARIMA fit on train params, filtered leak-free via `apply(refit=False)`.

---

## run_id: S4 — quantile ML models (GBQ + QR-LSTM)
- **date:** 2026-06-19 · **script:** `.build/step04_quantile_models.py` v1 → `artefacts/model_metrics.json`, `artefacts/model_preds.npz`
- **GBQ:** sklearn `GradientBoostingRegressor(loss='quantile', n_estimators=300, max_depth=3, lr=0.05, subsample=0.8)` per (lead, τ); predicted quantiles sorted for non-crossing. Features = 19 physics-guided + W_E (mean-filled with train means). Same per-lead origin set as baselines.
- **QR-LSTM:** LSTM(hidden 48)+station emb(6) over L=12-month standardized window → monotone cumsum-softplus head (Cannon 2018), multi-lead, masked pinball; early-stopped on val pinball (best scaled val pinball 0.2824). Common origin set (valid for all leads + full window): train 2197 / val 466 / test 737.
- **TEST metrics:**

| h | model | CRPS | CRPSS | PICP80 | PICP90 | DTF-AUC | FTD-AUC |
|--|--|--|--|--|--|--|--|
| 1 | **GBQ** | 0.225 | **+0.094** | 0.669 | 0.791 | 0.670 | 0.780 |
| 1 | QR-LSTM | 0.252 | +0.002 | 0.628 | 0.779 | 0.661 | 0.792 |
| 2 | **GBQ** | 0.234 | **+0.122** | 0.574 | 0.707 | 0.745 | 0.769 |
| 2 | QR-LSTM | 0.255 | +0.056 | 0.509 | 0.670 | 0.733 | 0.771 |
| 3 | **GBQ** | 0.303 | **+0.162** | 0.465 | 0.596 | 0.819 | 0.746 |
| 3 | QR-LSTM | 0.322 | +0.109 | 0.430 | 0.575 | 0.779 | 0.749 |

- **Reading (headline forming):** (i) **GBQ beats climatology on CRPS at every lead, and the skill grows with lead** (+0.094→+0.122→+0.162) because climatology degrades under the 2018–22 shift while the conditional model holds; (ii) GBQ **matches/edges the physical bucket on event AUC** (DTF 0.67/0.75/0.82 vs bucket 0.65/0.73/0.79) while also being a calibrated-target full-distribution forecaster → ML adds value over *both* climatology and physics; (iii) **QR-LSTM is competitive but consistently behind GBQ** — small-data (~2.2k train origins) favors boosting; honest finding, LSTM kept as the deep-model reference; (iv) **both models under-cover** (PICP90 0.79→0.60, worse at long lead) → the raw quantiles are sharp but mis-calibrated under shift → **Step 5 conformal must restore coverage** (the paper's calibration differentiator).
- **Caveat:** GBQ/baselines share per-lead origin sets; QR-LSTM uses the slightly smaller all-leads windowed set (737 vs ~764 test) — each CRPSS referenced to climatology on its own set, so skill scores are internally valid; cross-model ranking re-checked on a common set in Step 6 if needed.

---

## run_id: S5 — split-conformal calibration (CQR) + reliability
- **date:** 2026-06-19 · **script:** `.build/step05_conformal.py` v1 → `artefacts/conformal_metrics.json`
- **Setup:** calibration set C = **VAL block 2015–2017** (disjoint from train-fit and test-eval); CQR (Eq. 7–8) per symmetric τ-pair (90%/80%/50% central intervals), conformal radius Ê on C, widen test quantiles, re-sort, rebuild calibrated CDF.
- **GBQ — test PICP before → after CQR / ECE before → after:**

| h | PICP80 b→a | PICP90 b→a | width90 b→a | ECE b→a |
|--|--|--|--|--|
| 1 | 0.669 → **0.791** | 0.791 → **0.928** | 1.26 → 1.83 | 0.036 → 0.024 |
| 2 | 0.574 → **0.706** | 0.707 → **0.863** | 0.95 → 1.42 | 0.072 → 0.040 |
| 3 | 0.465 → **0.624** | 0.596 → **0.844** | 0.85 → 1.51 | 0.106 → 0.063 |

- **QR-LSTM — test PICP90 before → after:** h1 0.779→0.951, h2 0.670→0.900, h3 0.575→0.848; ECE h1 0.068→0.041, h2 0.089→0.037, h3 0.104→0.048.
- **Event detection after CQR (GBQ):** DTF-AUC 0.705/0.762/0.860, FTD-AUC 0.805/0.777/0.779; DTF-Brier 0.097/0.117/0.121.
- **Reading (calibration differentiator):** (i) **CQR substantially restores coverage** — GBQ PICP90 0.60–0.79 → 0.84–0.93, ECE roughly halved at every lead — at the expected cost of wider intervals; (ii) **residual under-coverage at h2/h3 (0.86/0.84 < 0.90)** is the real 2018–22 distribution shift: split-conformal's exchangeability is violated by the trend, so val-calibrated radii slightly under-correct test — an honest result that motivates the climate-adaptation framing and adaptive/time-series conformal (EnbPI) as future work; (iii) event AUC stays strong post-calibration. This is the calibration capability absent from the BD point-forecast literature.

---

## run_id: S6 — interpretation & evaluation (ablation, drivers, cost–loss, LOSO)
- **date:** 2026-06-19 · **script:** `.build/step06_interpret_eval.py` v1 → `artefacts/interpret_metrics.json`
- **[A] Lead-time CRPSS (test):** GBQ +0.094/+0.122/+0.162 > LSTM +0.002/+0.056/+0.109 > bucket −0.287/−0.130/+0.072 (h=1/2/3). GBQ best at all leads; skill rises with lead (climatology reference degrades under shift).
- **[B] Physics-guided feature ABLATION (GBQ full vs no-physics), the de-risking result:**

| h | CRPSS full | CRPSS no-phys | Δ | DTF-AUC full | DTF-AUC no-phys |
|--|--|--|--|--|--|
| 1 | +0.094 | +0.093 | +0.001 | 0.670 | 0.682 |
| 2 | +0.122 | +0.131 | −0.008 | 0.745 | 0.751 |
| 3 | +0.162 | +0.175 | −0.013 | 0.819 | 0.826 |

  → **Physics-guided features are NEUTRAL (marginally unhelpful at h2/h3).** The raw EO variables + boosting already capture the moisture signal; the engineered PET/water-balance/API/rolling features are largely redundant. **Headline does not depend on them** — this is the planned "physics neutral → one honest ablation" path (CLAUDE.md §4). Reported as a finding, not hidden.
- **[C] Permutation importance (GBQ h=2, Δpinball when shuffled):** **W_E +0.0458 (dominant)**, then sinm +0.0020 (season), zSM +0.0020 (soil-moisture anomaly), dSPEI +0.0014, Tn/lon/roll6/API/PET all ≤0.0003. Forecast is driven by the **known current wetness state W_E** + seasonality; physics-derived features marginal — consistent with [B]. (Honest caveat: DFAA contains W_E by construction and W_E is observed at origin, so much skill = current-state + reversion; legitimate but stated plainly.)
- **[D] Cost–loss decision value (GBQ+CQR, h=2):** DTF Vmax **0.367** at C/L=0.14 (V>0 for C/L∈[0.06,0.64]); FTD Vmax **0.412** at C/L=0.18 (V>0 for [0.06,0.72]). Calibrated probabilities carry genuine operational value across a broad range of user cost structures (AIMSIR/services lens).
- **[E] LOSO transfer (GBQ h=2, train on 12 stations → held-out station test):** **all 13 stations beat climatology**, mean CRPSS **+0.126** (min +0.024 Rangpur, max +0.209 Mymensingh); per-station DTF-AUC 0.61–0.87, FTD-AUC 0.64–0.93. LOSO ≈ in-sample skill → **robust spatial transfer to "ungauged" stations** (EO gap-filling angle).

---

## run_id: S6-supp — per-station table tabulated for paper Table II (no new model run)
- **date:** 2026-06-19 · descriptive tabulation of recorded labels (`dfaa.npz` lab_h2) + `interpret_metrics.json` loso_h2. Not an experiment (no fit/eval); fills the 4 station event-% values not previously listed in S2.
- **Per-station DFAA event % (h=2, full record) | LOSO CRPSS | DTF-AUC | FTD-AUC:**
  Faridpur 29.3 | 0.098 | 0.679 | 0.685 · Barisal 28.6 | 0.156 | 0.736 | 0.932 · Dhaka 25.6 | 0.144 | 0.793 | 0.804 ·
  Comilla 25.3 | 0.104 | 0.712 | 0.697 · Khulna 25.3 | 0.160 | 0.787 | 0.831 · Rajshahi 23.8 | 0.088 | 0.778 | 0.707 ·
  Bogra 23.4 | 0.124 | 0.764 | 0.779 · Chittagong 22.3 | 0.145 | 0.866 | 0.728 · Jessore 22.3 | 0.125 | 0.611 | 0.729 ·
  Sylhet 21.6 | 0.145 | 0.846 | 0.833 · Rangpur 21.2 | 0.024 | 0.645 | 0.843 · Mymensingh 19.8 | 0.209 | 0.725 | 0.857 ·
  Cox's Bazar 19.0 | 0.121 | 0.839 | 0.641. Event% range 19.0–29.3 (mean 23.6); LOSO CRPSS mean +0.126, all 13 > 0.

---

## Summary for the paper (traceable headline, all from runs above)
1. First **DFAA index + event climatology for Bangladesh** (S2); abrupt alternations **intensify 2018–22** (~33% vs 20% event rate).
2. **GBQ calibrated quantile model beats climatology on CRPS at every lead** (CRPSS +0.09→+0.16, S4) and **matches the physical water-balance bucket on event AUC** (≤0.82, S3–S4) → AI value over both climatology and physics.
3. **Conformal (CQR) restores coverage** (PICP90 →0.84–0.93, ECE halved, S5); residual gap = honest non-stationarity.
4. **Physics-guided features neutral** (S6-B) — headline robust without them, as designed.
5. **Transferable** (LOSO all-positive, S6-E) and **operationally valuable** (cost–loss, S6-D).
6. **QR-LSTM competitive but below GBQ** — small-data favors boosting (honest, S4).
