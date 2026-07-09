"""Step 5 - split-conformal quantile regression (CQR, Eq. 7-8) + reliability/PIT + event probs.

Calibration set C = the VAL block (2015-2017), disjoint from train (model fit) and test (eval).
For each symmetric quantile pair we compute a conformal radius E-hat on C and widen the test
quantiles, giving distribution-free marginal coverage (exchangeability approx; the 2018-22 shift
is itself reported). Rebuilds a calibrated CDF -> recomputes event Brier/AUC + PIT/ECE.

Writes artefacts/conformal_metrics.json + appends RESULTS_LOG.
"""
import json
import numpy as np
from common import ART, LEADS, TAUS, SEED
import evalkit as ek

np.random.seed(SEED)
print("=" * 70); print("STEP 5 - conformal calibration (CQR) + reliability"); print("=" * 70)

pp = np.load(ART / "model_preds.npz", allow_pickle=True)
meta = json.loads((ART / "dfaa_meta.json").read_text())
THETA = {h: meta["theta_D"][str(h)] for h in LEADS}

# symmetric (lo_idx, hi_idx, alpha) for the 7-tau grid {.05,.10,.25,.50,.75,.90,.95}
PAIRS = [(0, 6, 0.10), (1, 5, 0.20), (2, 4, 0.50)]   # 90%, 80%, 50% central intervals


def cqr_radius(Qcal, ycal, lo_i, hi_i, alpha):
    E = np.maximum(Qcal[:, lo_i] - ycal, ycal - Qcal[:, hi_i])
    n = len(E)
    k = min(int(np.ceil((n + 1) * (1 - alpha))), n)
    return float(np.sort(E)[k - 1])


def apply_cqr(Qcal, ycal, Qtest):
    """Return calibrated test quantiles (widen each symmetric pair by its conformal radius)."""
    Qc = Qtest.copy()
    radii = {}
    for lo_i, hi_i, a in PAIRS:
        e = cqr_radius(Qcal, ycal, lo_i, hi_i, a)
        radii[(lo_i, hi_i)] = e
        Qc[:, lo_i] = Qtest[:, lo_i] - e
        Qc[:, hi_i] = Qtest[:, hi_i] + e
    Qc = np.sort(Qc, axis=1)   # keep monotone
    return Qc, radii


def pit_ece(Q, y, grid=np.linspace(0.05, 0.95, 19)):
    pit = ek.cdf_at(Q, y)                       # PIT values, should be ~U(0,1)
    emp = np.array([(pit <= p).mean() for p in grid])
    return float(np.mean(np.abs(emp - grid))), pit


results = {}
for model in ["gbq", "lstm"]:
    results[model] = {}
    for h in LEADS:
        Qcal = pp[f"{model}_{h}_val_Q"]; ycal = pp[f"{model}_{h}_val_y"]
        Qte = pp[f"{model}_{h}_test_Q"]; yte = pp[f"{model}_{h}_test_y"]
        # before
        p80b, w80b = ek.coverage(Qte, yte, 1, 5)
        p90b, w90b = ek.coverage(Qte, yte, 0, 6)
        ece_b, _ = pit_ece(Qte, yte)
        evb = ek.event_metrics(Qte, yte, THETA[h])
        # after CQR
        Qc, radii = apply_cqr(Qcal, ycal, Qte)
        p80a, w80a = ek.coverage(Qc, yte, 1, 5)
        p90a, w90a = ek.coverage(Qc, yte, 0, 6)
        ece_a, _ = pit_ece(Qc, yte)
        eva = ek.event_metrics(Qc, yte, THETA[h])
        crps_a = ek.crps(Qc, yte)
        results[model][h] = {
            "before": {"picp80": p80b, "width80": w80b, "picp90": p90b, "width90": w90b,
                       "ece": ece_b, "event": evb},
            "after": {"picp80": p80a, "width80": w80a, "picp90": p90a, "width90": w90a,
                      "ece": ece_a, "event": eva, "crps": crps_a},
            "cqr_radius90": radii[(0, 6)], "cqr_radius80": radii[(1, 5)],
        }

(ART / "conformal_metrics.json").write_text(json.dumps(results, indent=2, default=float))

print("\nPICP (test) BEFORE -> AFTER CQR  [target 0.80 / 0.90], width, ECE:")
for model in ["gbq", "lstm"]:
    print(f"\n=== {model.upper()} ===")
    print(f"{'h':>2} {'PICP80_b':>9} {'PICP80_a':>9} {'PICP90_b':>9} {'PICP90_a':>9} "
          f"{'w90_b':>7} {'w90_a':>7} {'ECE_b':>7} {'ECE_a':>7}")
    for h in LEADS:
        b = results[model][h]["before"]; a = results[model][h]["after"]
        print(f"{h:>2} {b['picp80']:>9.3f} {a['picp80']:>9.3f} {b['picp90']:>9.3f} {a['picp90']:>9.3f} "
              f"{b['width90']:>7.3f} {a['width90']:>7.3f} {b['ece']:>7.3f} {a['ece']:>7.3f}")

print("\nEvent detection (test) AFTER CQR - DTF/FTD Brier & AUC (AUC rank-invariant to widening):")
for model in ["gbq", "lstm"]:
    print(f"\n=== {model.upper()} ===  {'h':>1} {'DTF-Brier':>10} {'DTF-AUC':>8} {'FTD-Brier':>10} {'FTD-AUC':>8}")
    for h in LEADS:
        ev = results[model][h]["after"]["event"]
        print(f"{'':>20} {h:>1} {ev['DTF']['brier']:>10.4f} {ev['DTF']['auc']:>8.3f} "
              f"{ev['FTD']['brier']:>10.4f} {ev['FTD']['auc']:>8.3f}")
print(f"\nWrote {ART/'conformal_metrics.json'}")
print("\nSTEP 5 OK.")
