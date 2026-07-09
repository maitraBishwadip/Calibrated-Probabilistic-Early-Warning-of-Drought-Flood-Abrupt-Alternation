# CLAUDE.md — Project Governance & Working Agreement

**Project:** Physics-informed, **calibrated probabilistic early warning** of compound
**"monsoon-whiplash" drought–flood extremes** (Drought–Flood Abrupt Alternation, **DFAA**) over
Bangladesh, from satellite Earth-observation data.
**Researcher:** Bishwadip Maitra · BUET.
**Primary venue:** **IEEE InGARSS 2026** (Hyderabad, 01–04 Dec 2026; theme *"Digital Earth — Modeling,
Mapping, Monitoring"*). **Secondary goal:** the writeup/methodology must also strengthen the
researcher's application to **UCD AIMSIR** (Centre for AI, Meteorological Services, Innovation and
Research — AI + meteorological physics + operational forecasting + climate adaptation).
**Format constraint:** **5-page IEEE InGARSS paper** (IEEEtran, 2-column). Verify the exact page/format
rules against the official InGARSS 2026 author kit before finalising.

**Data:** `Bangladesh Meterological data.csv` — **13 BMD stations, monthly, 2000–2022**
(~3,588 rows incl. **one trailing NaN row to drop**). Columns: `Station_Name, Station_Code, Year, Month,
Max_Temp, Min_Temp, Rainfall_mm, Soil_moisture_mm, SPEI_3`. The researcher states **all variables are
satellite/EO-derived** (this is what makes the study InGARSS-appropriate). **Provenance of each product
(which satellite/reanalysis source, native resolution) is not yet confirmed and MUST be identified and
cited before any claim about the data is written** (see Rule 3).

This file is the **standing instruction set** for any AI assistant working in this repository. It
overrides default behaviour. Read it before doing anything.

---

## 0. The researcher's 10 standing rules (authoritative — never violate)

> These are the rules the researcher set for this project. Everything below elaborates them.

1. **No unsupported claims.** No statement in any `.md` or `.tex` writeup unless it is **explicitly
   backed by a recorded result** in `RESULTS_LOG.md` (for our experiments) or a verified citation in
   `LITERATURE_REVIEW.md` (for prior work). No placeholder/"expected" numbers, no values borrowed from
   other papers presented as ours.
2. **Document everything.** Every result, its methodology, and the literature review behind it is
   written down — in `RESULTS_LOG.md` (results), `EXPERIMENT_DESIGN.md` (methods/plans), and
   `LITERATURE_REVIEW.md` (cited review). A number with no recorded method/provenance does not exist.
3. **Verify every DOI.** Every citation's DOI is checked to be **real and resolving (100% verified)**
   before it enters `LITERATURE_REVIEW.md` or the `.tex`. No fabricated or guessed DOIs. Record the
   verification (resolved title/authors/year) next to each entry. The same applies to the **data
   product sources** (which EO/reanalysis dataset each column comes from).
4. **Literature review before experiments.** Do a **thorough, written literature review for the
   specific step** before running anything that trains/fits/evaluates. No model is run before its
   relevant review exists.
5. **5-page IEEE InGARSS paper — always.** Every scope, figure, table, and ablation decision is made
   under the 5-page constraint. Favour one headline contribution + tight ablations over breadth.
6. **Be sure of the implementation.** Before running any model, confirm its implementation is correct
   (library, version, API, loss, shapes, splits). Smoke-test on real data first; never run a model you
   cannot vouch for line-by-line.
7. **Be 100% sure of the maths.** All mathematical equations, indices, and notation are verified
   correct and used consistently to describe the method. Define every symbol once; keep notation stable
   across code, `EXPERIMENT_DESIGN.md`, and the `.tex`.
8. **Report-then-correct loop.** After every experiment, give the researcher a **plain-language
   overview of the experiment and its results** (what was run, on what, what came out, caveats) so the
   researcher can correct before anything is treated as final or written into the paper.
9. **Deliverables are runnable notebooks.** Write Python as **Jupyter notebooks (`.ipynb`)** that run on
   **Google Colab GPU/CPU or a local machine** (auto-detect device, pip-install guarded, self-contained
   data load). See § 6 for the notebook workflow.
10. **Document all plans properly.** Every plan is written to `EXPERIMENT_DESIGN.md` (or a dated
    `docs/YYYY-MM-DD_*.md`) before execution, with enough detail to reproduce.
11.**Generate Publication Grade And Meaningful Plots Only** Before generating a plot look for how this can 
    be made publicTION grade search make sure the quality of the plot and then crerate all images save PNG PDF TIFF
12. **Import Writting Skills** While writting in any Latex file you have to make sure you are     writting      like a phd holder so check the skills before writting and the explainations should be well structured and naturally flows from experiment to experiment the story telling must be top notch

---

## 1. The single most important workflow rule — EXPLAIN → APPROVE → EXECUTE

> **For EVERY experiment, follow this protocol — no exceptions** (this operationalises Rules 4, 6, 8, 10):
> 1. **LITERATURE REVIEW** — written, cited, DOI-verified review for this specific step (Rules 3, 4).
> 2. **EXPLAIN / PLAN** — state *what* you will do and *how* (data, model, loss, metric, equations,
>    expected artefacts), written to `EXPERIMENT_DESIGN.md` (Rule 10). Confirm the implementation and
>    the maths first (Rules 6, 7).
> 3. **WAIT FOR APPROVAL** — do not run anything until the researcher explicitly approves.
> 4. **EXECUTE** — run it, record the run in `RESULTS_LOG.md` (Rule 2).
> 5. **REPORT** — give the plain-language overview for correction (Rule 8) *before* anything is written
>    into the paper.

"Experiment" = anything that trains, fits, evaluates, tunes, or produces a number that could end up in
the paper. **Reading / profiling the data is NOT an experiment** and may proceed freely. When in doubt,
treat it as an experiment and ask first.

---

## 2. Sources of truth (what reads from what)

- **Results of record → `RESULTS_LOG.md`** (create on first run). Append-only, one entry per run:
  `run_id`, date, notebook/script + version, config (seed, quantile grid, horizons, loss weights,
  split), every metric value, and notes/caveats. **The paper reads only from here.**
- **Architecture & plans of record → `EXPERIMENT_DESIGN.md`** (intentions, equations, model/loss/data
  pipeline, baselines, metrics). Keep in sync with the notebooks.
- **Literature of record → `LITERATURE_REVIEW.md`** (cited, thematic, every DOI verified per Rule 3).
- **Negatives are results.** A model that underperforms a baseline, a physics term that does not help,
  a foundation model that overfits the small data — record it plainly. Do not drop unfavourable runs.
  (This project's physics and foundation-model components are explicitly *ablation, not load-bearing* —
  see § 4.)

---

## 3. Locked study definition (decided with the researcher so far)

| Decision | Value |
|---|---|
| **Problem** | Compound **Drought–Flood Abrupt Alternation (DFAA)** — "monsoon-whiplash" — over Bangladesh |
| **Deliverable** | **Calibrated probabilistic early warning** (predictive distribution + lead-time skill), not point forecasts |
| **Framing** | **Climate-adaptive** early warning (motivation/eval lens), with a concrete EO + AI/physics method as the gradeable contribution |
| **Data** | `Bangladesh Meterological data.csv` — 13 stations, monthly, 2000–2022; all EO-derived (provenance to be confirmed) |
| **Physics role** | **Ablation only — never load-bearing.** Realised as **physics-guided input features + a water-balance bucket baseline; NO physics term in any loss → this is *not* a PINN.** Headline holds even though physics was neutral (`RESULTS_LOG.md` S6-B) |
| **Method family (LOCKED)** | **NOT a PINN.** Monotone-quantile **GBQ** + compact **QR-LSTM** → **split-conformal (CQR)** calibration. Quantile-regression + conformal, physics-*guided* not physics-*informed* |
| **Foundation-model role** | **Upside ablation only** (LoRA/light decoder; small data ⇒ probe/transfer, not main claim) |
| **Primary InGARSS track** | **Natural Hazards / Disaster Management** (secondary tag: AI/ML for EO) |

**RESOLVED & RUN — experiments S1–S6 complete** (these were designed → approved → run per § 1; final
definitions in `EXPERIMENT_DESIGN.md` "Locked decisions", numbers in `RESULTS_LOG.md`. Kept here for history):
- The exact **DFAA index** definition (multivariate composite over rainfall + soil moisture + SPEI; Eq. 2,
  α=1.8, equal weights) and its **DFAA event labels** (θ_D = 80th pct of |DFAA| on train) — **decided & run**.
- The **forecast horizons** h∈{1,2,3} months and the **split** (train 2000–14 / val 2015–17 / test 2018–22,
  + leave-one-station-out; all thresholds/scalers/index params fit on **train only** — see § 5) — **decided & run**.
- The **model stack** — **LOCKED & run** (below).

**LOCKED model stack (run; results in `RESULTS_LOG.md`):** baselines (climatology, seasonal climatology,
persistence, SARIMA, **water-balance bucket**) → core **gradient-boosting quantile (GBQ)** + a compact
**QR-LSTM** deep reference, both using a **monotone non-crossing quantile head** → **split-conformal (CQR)**
calibration wrapper. **NOT a PINN — there is no physics term in any loss.** Physics enters only as
**physics-guided input features** (Hargreaves PET, climatic water balance, API, ΔSM) and the **bucket
baseline**. TFT, ST-GNN, the foundation-model probe, and a physics-residual/PINN variant were scoped as
*upside only* and **not pursued** for the 5-page paper.
**Metrics:** CRPS, pinball/quantile loss, reliability (PICP/coverage diagrams), Brier/ROC for DFAA-event
detection, **lead-time vs. skill**, and a **cost–loss / decision-threshold** analysis (operational value
for the AIMSIR/services angle).

This project **reuses only the monotone non-crossing quantile head** (Cannon 2018; cumsum-of-softplus →
non-crossing CDF) from the `Multiple pollutant combine impact  study in Bangladesh/` QR-PINN project —
**not** its physics residual. Dropping that residual is exactly what makes our model a **QR-LSTM, not a
QR-PINN**. Mine that repo for the head's code only; do not import its physics loss or rebuild its pollution study.

---

## 4. De-risking decisions (already agreed — keep them)

- **Physics is an ablation with a graceful-degradation ladder:** hard residual → soft/annealed residual
  (tunable weight) → physics-guided *features* → physics as a post-hoc *consistency diagnostic/baseline*.
  The paper's headline (DFAA novelty + calibration) does **not** depend on which rung is reached.
  Reporting paths: physics helps → headline includes it; physics neutral → one honest ablation; physics
  hurts → a *finding* with the sensitivity curve + mechanism (monthly aggregation + EO closure error).
  **Realised rung: physics-guided *features* (no residual, no PINN); result was neutral (`RESULTS_LOG.md`
  S6-B) → reported as one honest ablation. The headline does not depend on it.**
- **Foundation-model fine-tune is upside, not load-bearing** — small dataset (~3.5k rows) means favour
  parameter-efficient fine-tuning / feature-extraction and report overfit honestly if it occurs.

---

## 5. Reproducibility & no-leakage norms

- **No leakage, ever.** All scalers, percentile thresholds, the DFAA index parameters, KMeans/regime
  defs (if any), and event labels are fit on the **training split only** and applied unchanged to
  val/test. Rows are autocorrelated → **never random-shuffle**; use time-ordered + leave-one-station-out.
- **Determinism.** Fix and log all seeds (numpy/torch). Persist index definitions, scalers, and
  thresholds as artefacts so labels/targets are reproducible.
- **Machine-readable outputs.** Each notebook writes a JSON/CSV of its metrics consumed verbatim by the
  report; the paper never hand-copies numbers.
- **Raw data is never edited.** `Bangladesh Meterological data.csv` is read-only; cleaning happens in
  code (drop the NaN row, etc.), outputs go to new files.
- Record environment (key package versions) alongside results.

---

## 6. Notebook workflow (Rule 9)

- Deliverables are **`.ipynb`** that run unchanged on **Colab GPU/CPU or local**:
  - First cell: guarded `pip install`, imports, **device auto-detect** (CUDA → GPU else CPU), seed set.
  - Data load auto-detects Colab (mount Drive / file upload) vs. local path; fail loudly if data missing.
  - Self-contained: a reviewer can *Run All* top-to-bottom.
- Mirror the AeroHealth discipline if helpful: prototype/smoke-test logic as plain `.py`, then assemble
  the shipped notebook — but the **shipped artefact is the notebook**.
- Every notebook ends by writing its metrics JSON (for `RESULTS_LOG.md`) and a short printed summary
  (for the Rule 8 overview).

---

## 7. Repository map (target layout — create as work proceeds)

```
Bangladesh Flood/
├── CLAUDE.md                       # this file — governance
├── LITERATURE_REVIEW.md            # cited, DOI-verified review (DFAA, calibrated forecasting, EO drought)
├── EXPERIMENT_DESIGN.md            # architecture + plans + equations (approve before running)
├── RESULTS_LOG.md                  # created on first run — source of truth for the paper
├── Bangladesh Meterological data.csv   # raw data (DO NOT EDIT)
├── notebooks/                      # shipped .ipynb (Colab/local) — the deliverables
├── docs/                           # dated docs/YYYY-MM-DD_*.md per completed step (incl. negatives)
├── artefacts/                      # index defs, scalers, thresholds, metrics JSON
└── paper/                          # main.tex (IEEEtran, 5 pages) + figures
```

---

## 8. Interaction style expected here

- Literature-review → explain → **wait for approval** → execute → **report for correction** (§ 1).
- Recommend a default; don't dump every option.
- Be precise about what is **measured** vs **planned** — the credibility of a 5-page IEEE paper and of
  the AIMSIR application depends on it.
- Keep `LITERATURE_REVIEW.md`, `EXPERIMENT_DESIGN.md`, and `RESULTS_LOG.md` current as work proceeds.
```
