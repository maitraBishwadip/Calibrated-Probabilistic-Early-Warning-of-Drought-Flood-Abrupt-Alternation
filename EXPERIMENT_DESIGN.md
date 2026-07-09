# EXPERIMENT_DESIGN.md — Math Model, Plan & Interpretation

**Status:** Plan approved 2026-06-18. **No experiments run yet** — each step below follows
EXPLAIN → WAIT FOR APPROVAL → EXECUTE (see `CLAUDE.md §1`). This is the architecture & plan of record;
results go to `RESULTS_LOG.md` (created on first run); citations live in `LITERATURE_REVIEW.md`.

---

## Context (why this work)

Bangladesh's monsoon-driven hydroclimate produces **both** severe droughts and floods, often in rapid
succession ("monsoon whiplash"). Our monthly satellite-derived record (13 BMD stations, 2000–2022:
`Rainfall_mm`, `Soil_moisture_mm`, `SPEI_3`, `Max_Temp`, `Min_Temp`) shows 19.3% of station-months in
moderate+ drought and 342 months with >500 mm rain — both tails coexist. **Drought–Flood Abrupt
Alternation (DFAA)** is active in China but **unstudied for Bangladesh**; existing BD drought-ML papers
give **point forecasts without calibrated uncertainty**. We deliver the first DFAA early-warning study
for Bangladesh: a **physics-guided, calibrated probabilistic** forecast of a DFAA index, from which event
probabilities are derived — fitting IEEE InGARSS 2026 (Natural Hazards / Digital Earth) and the UCD
AIMSIR profile (AI + meteorological physics + operational, climate-adaptive services).

**Outcome:** a 5-page IEEE InGARSS paper whose load-bearing contribution (first BD DFAA study + calibration)
holds regardless of whether the physics-guided features help, plus a reproducible notebook pipeline.

## Locked decisions

| Decision | Value |
|---|---|
| Method family | **NOT a PINN.** Calibrated probabilistic quantile forecasting + conformal calibration |
| Physics role | **Physics-guided features (inputs) + water-balance bucket baseline.** No physics in the loss. "physics-**guided**" |
| Forecast target | **Predictive distribution of a continuous DFAA index → derive P(DFAA event) by exceedance** |
| Venue / format | IEEE InGARSS 2026, **5 pages**, primary track Natural Hazards/Disaster Mgmt (secondary AI/ML) |
| Integrity | Every number traces to `RESULTS_LOG.md`; every DOI verified before `.tex` (CLAUDE.md Rules 1–10) |

---

## Notation (stable across code, this doc, and the .tex — Rule 7)

```
s        station index (1..13)            t      month index (2000-01 .. 2022-12)
m(t)     calendar month of t (1..12)      h      forecast lead (months): h ∈ {1,2,3}
P,SM     rainfall (mm), soil moisture (mm)   SPEI = SPEI_3 (already standardized)
Tx,Tn    Max_Temp, Min_Temp (°C)          PET    potential ET (Hargreaves, mm)
z_v      standardized anomaly of var v (train climatology, per (s,m))
W        composite "wetness" anomaly      DFAA   drought–flood abrupt alternation index (Eq. 2)
y        forecast target (= DFAA or W_L)  τ      quantile level (τ ∈ T_grid)
Q_τ(·)   predictive τ-quantile            F̂      predictive CDF        ρ_τ  pinball loss
θ_D      DFAA event magnitude threshold (train-fit)   α   conformal miscoverage (e.g. 0.1)
```
All μ, σ, climatologies, thresholds θ, scalers, conformal quantiles are **fit on the training split only**.

---

## Step-by-step math model

### Step 1 — Clean & standardize (no leakage)
1. Drop the trailing all-NaN row; sort by `(s, Year, Month)`; assert 276 contiguous months/station
   (Khulna has 275 — flag the gap).
2. **Train climatology** per `(station s, calendar month m)` on TRAIN years only: `μ_v(s,m), σ_v(s,m)` for
   v ∈ {P, SM}. Standardized anomaly:
   ```
   z_v(s,t) = ( x_v(s,t) − μ_v(s, m(t)) ) / σ_v(s, m(t))            (1)
   ```
   `SPEI` is used as-is (already standardized). Persist all `μ,σ` as artefacts.

### Step 2 — Composite wetness & the DFAA index (the target)
3. **Composite wetness anomaly** (all three signed so + = wet):
   ```
   W(s,t) = w1·z_P(s,t) + w2·z_SM(s,t) + w3·SPEI(s,t),  Σw=1  (default 1/3 each; sensitivity-tested)
   ```
4. **DFAA index** adapting the established LDFAI (Wu 2006) to a monthly multivariate form. For an *early*
   window E = mean W over [t−k+1 … t] and a *late* window L = mean W over [t+1 … t+h] (k = h = scale, default 2):
   ```
   DFAA(s,t) = (W_L − W_E) · (|W_E| + |W_L|) · α^(−|W_E + W_L|),   α = 1.8        (2)
   ```
   - `(W_L−W_E)`: direction & magnitude of the flip; **>0 ⇒ drought→flood (DTF)**, **<0 ⇒ flood→drought (FTD)**.
   - `(|W_E|+|W_L|)`: requires both windows anomalous (a real swing, not noise).
   - `α^(−|W_E+W_L|)`: **suppresses persistent same-sign** (non-alternating) cases. α=1.8 is the Wu (2006)
     constant; report sensitivity to α and to a copula-based MSDFI variant (Bai 2024) as robustness.
5. **Event labels** (train-fit θ_D = 80th pct of |DFAA| on train, or rule |W_E|≥1 ∧ |W_L|≥1 ∧ sign-flip):
   `DTF event ⇔ DFAA ≥ +θ_D ;  FTD event ⇔ DFAA ≤ −θ_D`.

### Step 3 — Physics-guided features + water-balance baseline (no PINN)
6. **PET (Hargreaves–Samani):** Ra = extraterrestrial radiation from station latitude + day-of-year,
   ```
   PET = 0.0023 · Ra · (T̄ + 17.8) · sqrt(Tx − Tn),   T̄ = (Tx+Tn)/2                 (3)
   ```
7. **Physics-guided input features:** climatic water balance `D = P − PET`; antecedent precipitation index
   `API_t = Σ_{k≥1} β^k P_{t−k}` (β≈0.9); soil-moisture tendency `ΔSM`; aridity `P/PET`; SPEI tendency;
   rolling 3/6-month sums; `sin/cos` of month; station id; lat/lon.
8. **Water-balance bucket BASELINE** (parameter-light physical forecaster to beat):
   ```
   S_t = clip( S_{t−1} + P_t − PET_t·g(S_{t−1}) − Q_t,  0,  S_max ),  Q_t = overflow above S_max   (4)
   ```
   Roll Eq. (4) forward h months (climatological P,PET for the future window), convert S to wetness, compute
   a deterministic DFAA forecast via Eq. (2). The physics benchmark.

### Step 4 — Probabilistic forecasting model (quantile, calibrated)
9. **Target:** `y = DFAA(s,t)` at lead h (or forecast `W_L`, then apply Eq. 2 with known `W_E`).
   **Features:** windowed past of {z_P, z_SM, SPEI, Tx, Tn, PET, D, API, ΔSM, calendar, station, lat/lon}.
10. **Monotone non-crossing quantile head** (reused from QR-PINN; Cannon 2018). Grid
    `T_grid = {0.05,0.1,0.25,0.5,0.75,0.9,0.95}` (extend 0.01/0.99 for tails):
    ```
    Q_{τ_1} = base(x);   Q_{τ_k} = Q_{τ_1} + Σ_{j=2}^{k} softplus(δ_j(x))      (5)
    ```
11. **Pinball loss** (Koenker–Bassett), averaged over τ, leads, observed samples:
    ```
    ρ_τ(u) = max(τ·u, (τ−1)·u),  u = y − Q_τ ;   L = mean_{i,h,τ} ρ_τ(y − Q_τ)   (6)
    ```
12. **Model stack** (small data ⇒ lead with robust learners; report which wins honestly):
    - Baselines: climatology, persistence, SARIMA, **water-balance bucket** (Step 3).
    - Core: **gradient-boosting quantile / NGBoost** and a **compact QR-LSTM** (Eq. 5–6 head).
    - Optional (upside, not load-bearing; report as such): **TFT**, **ST-GNN** over the 13-station graph,
      **foundation-model probe** (Prithvi-WxC features).

### Step 5 — Calibration via split-conformal quantile regression (CQR)
13. Split TRAIN into proper-train + calibration C. Fit on proper-train. For interval `[Q_{αlo}, Q_{αhi}]`,
    nonconformity on C:
    ```
    E_i = max( Q_{αlo}(x_i) − y_i ,  y_i − Q_{αhi}(x_i) )                          (7)
    k = ⌈(n+1)(1−α)⌉ ;  Ê = k-th smallest E_i ;  interval = [Q_{αlo} − Ê,  Q_{αhi} + Ê]   (8)
    ```
    Guarantees marginal coverage ≥ 1−α (distribution-free), per lead h. Also report PIT-recalibration
    (Kuleshov 2018) as an alternative.

### Step 6 — Index distribution → event probability (the early-warning output)
14. From calibrated quantiles build predictive CDF `F̂` (monotone interpolation):
    ```
    P(DTF) = 1 − F̂(+θ_D) ;   P(FTD) = F̂(−θ_D)                                   (9)
    ```
    The operational warning probabilities at lead h, per station-month.

### Step 7 — Metrics
- Probabilistic: **pinball**, **CRPS ≈ 2·mean_τ(pinball)** (Gneiting–Raftery), **CRPSS = 1 − CRPS/CRPS_clim**.
- Calibration: **PICP** & mean width @80/90%; **reliability/PIT**; expected calibration error.
- Event detection: **Brier**, **ROC-AUC**, **PR-AUC** (rare events), event-probability reliability.
- Operational: **lead-time vs skill** (CRPS/AUC vs h); **cost–loss value** `V = (E_clim − E_fc)/(E_clim − E_perfect)`.

### Step 8 — Splits, seeds, leakage
- **Time-ordered split** (e.g. train 2000–2014, val 2015–2017, test 2018–2022) + **Leave-One-Station-Out**.
  **Never random-shuffle** (autocorrelated rows).
- All climatology/scalers/θ_D/conformal Ê fit on **train (or proper-train/calibration) only**; applied
  unchanged to val/test/held-out station. Fix & log seeds; persist artefacts; write metrics JSON per run.

---

## Interpretation plan (how each result is read — Rule 8, before anything enters the paper)

1. **Skill vs baselines (headline):** does the calibrated model beat climatology/persistence/SARIMA and the
   **water-balance bucket** on CRPS/CRPSS and event AUC/PR? Beating the physical bucket = "AI adds value over
   physics"; if not, report plainly (honest negative).
2. **Calibration:** reliability/PIT near diagonal and PICP≈nominal ⇒ trustworthy probabilities. CQR judged by
   whether it restores coverage the raw model misses — the differentiator vs BD point-forecast literature.
3. **Lead-time decay:** skill vs h gives the usable warning horizon; state the lead where CRPSS crosses 0.
4. **DTF vs FTD asymmetry:** compare skill/drivers/frequency of drought→flood vs flood→drought (literature:
   drivers differ — precip/pressure/NDVI for DTF; precip/ET for FTD). Report which is more predictable here.
5. **Driver attribution:** SHAP (boosting) + permutation importance, integrated-gradients for the LSTM; rank
   features (rainfall anomaly, API, ΔSM, PET/aridity, SPEI). Require **cross-method agreement** before claiming a
   driver. Tie any temperature/PET dominance to the warming trend in the record.
6. **Spatial:** map per-station skill and DFAA-event frequency (expect NE haor/Sylhet vs NW contrast). LOSO skill
   = transferability to "ungauged" stations (EO gap-filling angle).
7. **Physics-guided ablation:** with vs without the Step-3 features. Report the delta honestly (help/neutral/hurt);
   the headline does not depend on it.
8. **Cost–loss / adaptation:** translate probabilities into decision value across user cost/loss ratios — the
   "climate-adaptive operational service" interpretation for InGARSS/AIMSIR.

---

## Execution sequence (notebooks — Rule 9; each approve-before-run)

| Notebook | Produces | Writes |
|---|---|---|
| `01_explore_clean.ipynb` | profiling, gap report, climatology fit (train-only) | `artefacts/clima.json` |
| `02_build_dfaa_index.ipynb` | W, DFAA (Eq. 2), event labels, hotspot maps | `artefacts/dfaa.npz`, figs |
| `03_features_baselines.ipynb` | PET, physics-guided features, climatology/persistence/SARIMA/**bucket** baselines | `RESULTS_LOG.md` |
| `04_quantile_models.ipynb` | NGBoost/GBQ + QR-LSTM quantiles (Eq. 5–6) | metrics JSON → `RESULTS_LOG.md` |
| `05_conformal_calibration.ipynb` | CQR (Eq. 7–8), reliability/PIT, event probs (Eq. 9) | metrics JSON, figs |
| `06_interpret_eval.ipynb` | CRPSS, lead-time, cost–loss, SHAP/IG, LOSO, physics ablation | figs, `docs/YYYY-MM-DD_*.md` |
| (optional) `07_extensions.ipynb` | TFT / ST-GNN / foundation-model probe | only if pursued |
| `make_paper_figs.ipynb` | final manuscript figures from logged results | `paper/figs/` |

Each notebook: device auto-detect (Colab GPU/CPU/local), guarded installs, train-only fits, seeds logged, ends
with a printed plain-language summary for the Rule-8 review.

## 5-page paper mapping (Rule 5)
Intro+gap (0.75 p) · Data + DFAA index Eq.2 & PET (1 p) · Method: quantile head + CQR Eq.5–9 (1 p) ·
Results: skill/calibration/lead-time/maps, ~3 figs + 1 table (1.5 p) · Interpretation + adaptation +
conclusion (0.5 p) · refs (IEEEtran `thebibliography`). Extensions only if space + results justify.

## Verification (end-to-end)
- **Index sanity:** Eq. (2) reproduces known BD wet/dry swings; DTF/FTD counts non-trivial and spatially sensible.
- **Leakage probes:** shuffling test-time future features destroys skill; assert no test months touch μ,σ,θ,Ê;
  persistence/climatology baselines reproduce known reference scores.
- **Calibration gate:** post-CQR PICP within ±2–3% of nominal on test.
- **Reproducibility:** re-running with logged seed reproduces `RESULTS_LOG.md` metrics.

## Open items to confirm (before the relevant step)
- DFAA window scale k=h (default 2) and forecast leads (default 1–3 months).
- Composite weights w1:w2:w3 (default equal) and θ_D rule (percentile vs sign-flip).
- Exact train/val/test year boundaries.
- Whether to run any optional extension (TFT / GNN / foundation-model) for v1 or defer.
- Confirm official InGARSS 2026 page limit (5 pp per researcher; IGARSS-family sometimes 4).
- Pin EO provenance of each column (product/resolution) + verify ⚠/● DOIs via doi.org (Rule 3).
