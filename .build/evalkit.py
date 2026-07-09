"""Shared probabilistic-forecast evaluation (reused by Steps 3-6).

All metrics take predictive quantiles Q[N, nT] (monotone non-decreasing in tau) and targets y[N].
CRPS uses the quantile estimator CRPS ~= 2*mean_tau(pinball_tau) (EXPERIMENT_DESIGN Step 7);
coarse but identical across methods, so CRPSS comparisons are fair.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from common import TAUS


def pinball_per_tau(Q, y, taus=TAUS):
    err = y[:, None] - Q                       # [N,nT]
    pb = np.maximum(taus[None, :] * err, (taus[None, :] - 1) * err)
    return pb.mean(axis=0)                      # [nT]


def pinball(Q, y, taus=TAUS):
    return float(pinball_per_tau(Q, y, taus).mean())


def crps(Q, y, taus=TAUS):
    return float(2.0 * pinball_per_tau(Q, y, taus).mean())


def coverage(Q, y, lo_i, hi_i):
    lo, hi = Q[:, lo_i], Q[:, hi_i]
    inside = (y >= lo) & (y <= hi)
    return float(inside.mean()), float((hi - lo).mean())


def cdf_at(Q, thr, taus=TAUS):
    """Predictive CDF at threshold thr via linear interpolation across the quantile grid."""
    Q = np.asarray(Q, float)
    N, m = Q.shape
    thr_arr = np.full(N, thr) if np.isscalar(thr) else np.asarray(thr, float)
    k = np.sum(Q < thr_arr[:, None], axis=1)
    lo = np.clip(k - 1, 0, m - 1); hi = np.clip(k, 0, m - 1)
    ar = np.arange(N)
    qlo, qhi = Q[ar, lo], Q[ar, hi]
    tlo, thi = taus[lo], taus[hi]
    gap = qhi - qlo
    frac = np.where(gap > 1e-9, (thr_arr - qlo) / np.where(gap > 1e-9, gap, 1.0), 0.0)
    cdf = tlo + frac * (thi - tlo)
    cdf = np.where(k == 0, taus[0], cdf)
    cdf = np.where(k == m, taus[-1], cdf)
    return np.clip(cdf, 0.0, 1.0)


def event_metrics(Q, y, theta, taus=TAUS):
    """DTF (y>=+theta) and FTD (y<=-theta) detection from the predictive CDF."""
    out = {}
    p_dtf = 1.0 - cdf_at(Q, theta, taus)
    p_ftd = cdf_at(Q, -theta, taus)
    for name, p, ind in [("DTF", p_dtf, (y >= theta).astype(int)),
                          ("FTD", p_ftd, (y <= -theta).astype(int))]:
        d = {"base_rate": float(ind.mean()), "brier": float(brier_score_loss(ind, np.clip(p, 0, 1)))}
        if ind.sum() > 0 and ind.sum() < len(ind):
            d["auc"] = float(roc_auc_score(ind, p))
            d["pr_auc"] = float(average_precision_score(ind, p))
        else:
            d["auc"] = float("nan"); d["pr_auc"] = float("nan")
        out[name] = d
    return out


def all_metrics(Q, y, theta, crps_ref=None, taus=TAUS):
    """Bundle of probabilistic + calibration + event metrics for one method/lead/split."""
    pin = pinball(Q, y, taus)
    cr = crps(Q, y, taus)
    p80, w80 = coverage(Q, y, 1, 5)   # taus index 1=0.10, 5=0.90 -> 80% PI
    p90, w90 = coverage(Q, y, 0, 6)   # taus index 0=0.05, 6=0.95 -> 90% PI
    m = {"n": int(len(y)), "pinball": pin, "crps": cr,
         "picp80": p80, "width80": w80, "picp90": p90, "width90": w90}
    if crps_ref is not None and crps_ref > 0:
        m["crpss"] = float(1.0 - cr / crps_ref)
    m["event"] = event_metrics(Q, y, theta, taus)
    return m


def residual_quantiles(resid_train, taus=TAUS):
    """Empirical quantiles of training residuals -> additive spread for a point forecast."""
    r = resid_train[np.isfinite(resid_train)]
    return np.quantile(r, taus)


def point_to_quantiles(point, resid_q):
    """Q[N,nT] = point[:,None] + resid_q[None,:] (homoscedastic residual probabilization)."""
    Q = point[:, None] + resid_q[None, :]
    return np.maximum.accumulate(Q, axis=1)   # enforce monotone (sorted resid_q already monotone)
