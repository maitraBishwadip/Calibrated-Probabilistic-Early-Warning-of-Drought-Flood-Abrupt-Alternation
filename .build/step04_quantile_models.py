"""Step 4 - quantile forecasting models for DFAA(s,t,h).

  * GBQ     : sklearn GradientBoostingRegressor(loss='quantile') per (lead, tau);
              predicted quantiles sorted to enforce non-crossing.
  * QR-LSTM : LSTM encoder over an L=12-month feature window + station embedding ->
              monotone non-crossing quantile head (cumsum of softplus increments, Cannon 2018),
              multi-lead output, masked pinball loss (Koenker-Bassett).

Same valid-DFAA origin set as the baselines (missing features mean-filled with TRAIN means).
Writes artefacts/model_metrics.json (+ raw test quantiles for Step 5 conformal) and appends RESULTS_LOG.
"""
import json
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingRegressor
from common import ART, load_clean, train_mask_t, LEADS, TAUS, SEED
import evalkit as ek

np.random.seed(SEED); torch.manual_seed(SEED)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("=" * 70); print(f"STEP 4 - quantile models (GBQ + QR-LSTM)  device={DEV}"); print("=" * 70)

df, stations, sid, _ = load_clean()
S, T = len(stations), 276
train_t = train_mask_t()
yr = 2000 + np.arange(T) // 12

fz = np.load(ART / "features.npz", allow_pickle=True)
feat = fz["feat"].astype(np.float64)            # (S,T,F)
fnames = list(fz["names"])
F = feat.shape[-1]
dz = np.load(ART / "dfaa.npz", allow_pickle=True)
DFAA = {h: dz[f"DFAA_h{h}"] for h in LEADS}
WE = {h: dz[f"WE_h{h}"] for h in LEADS}
meta = json.loads((ART / "dfaa_meta.json").read_text())
THETA = {h: meta["theta_D"][str(h)] for h in LEADS}

# ---- per-feature train mean/std (leak-free), for mean-fill + LSTM scaling ----
fmu = np.zeros(F); fsd = np.ones(F)
for j in range(F):
    v = feat[:, train_t, j]
    v = v[np.isfinite(v)]
    fmu[j] = v.mean(); fsd[j] = v.std() + 1e-8
feat_fill = np.where(np.isfinite(feat), feat, fmu[None, None, :])   # mean-filled (raw scale)
feat_z = (feat_fill - fmu[None, None, :]) / fsd[None, None, :]      # standardized (for LSTM)

we_mu = {h: float(np.nanmean(WE[h][:, train_t])) for h in LEADS}


def split_of(t):
    y = yr[t]
    return "train" if y <= 2014 else ("val" if y <= 2017 else "test")


# origins with valid target per lead
origins = {h: [(s, t) for s in range(S) for t in range(T) if np.isfinite(DFAA[h][s, t])] for h in LEADS}
spl = {h: {sp: [(s, t) for (s, t) in origins[h] if split_of(t) == sp] for sp in ["train", "val", "test"]}
       for h in LEADS}

# ================= GBQ =================
print("\n[GBQ] training GradientBoostingRegressor(loss='quantile') per (lead, tau)...")
t0 = time.time()
gbq_metrics = {}
gbq_Q = {h: {} for h in LEADS}   # store test/val raw quantiles for conformal
for h in LEADS:
    def Xy(idx):
        X = np.array([np.concatenate([feat_fill[s, t],
                      [WE[h][s, t] if np.isfinite(WE[h][s, t]) else we_mu[h]]]) for s, t in idx])
        y = np.array([DFAA[h][s, t] for s, t in idx])
        return X, y
    Xtr, ytr = Xy(spl[h]["train"])
    preds = {}
    for sp in ["val", "test"]:
        Xs, ys = Xy(spl[h][sp]); preds[sp] = (Xs, ys)
    Qcols = {sp: [] for sp in ["val", "test"]}
    for tau in TAUS:
        gb = GradientBoostingRegressor(loss="quantile", alpha=float(tau), n_estimators=300,
                                       max_depth=3, learning_rate=0.05, subsample=0.8,
                                       random_state=SEED)
        gb.fit(Xtr, ytr)
        for sp in ["val", "test"]:
            Qcols[sp].append(gb.predict(preds[sp][0]))
    gbq_metrics[h] = {}
    for sp in ["val", "test"]:
        Q = np.sort(np.stack(Qcols[sp], axis=1), axis=1)   # enforce non-crossing
        y = preds[sp][1]
        ref = ek.crps(np.tile(np.quantile(ytr, TAUS), (len(y), 1)), y)
        gbq_metrics[h][sp] = ek.all_metrics(Q, y, THETA[h], crps_ref=ref)
        gbq_Q[h][sp] = (Q, y)
print(f"  GBQ done in {time.time()-t0:.0f}s")

# ================= QR-LSTM =================
L = 12
nT = len(TAUS)
nL = len(LEADS)
ystd = {h: float(np.std([DFAA[h][s, t] for s, t in spl[h]["train"]])) for h in LEADS}  # target scale


def make_window_set(split):
    """Common origins (valid for ALL leads) in a split -> (Xwin, sid, Y[nL], M[nL])."""
    valid = [(s, t) for s in range(S) for t in range(T)
             if split_of(t) == split and t - L + 1 >= 0
             and all(np.isfinite(DFAA[h][s, t]) for h in LEADS)]
    Xw = np.array([feat_z[s, t - L + 1:t + 1] for s, t in valid], dtype=np.float32)  # (N,L,F)
    si = np.array([s for s, t in valid], dtype=np.int64)
    Y = np.array([[DFAA[h][s, t] / ystd[h] for h in LEADS] for s, t in valid], dtype=np.float32)
    idx = np.array(valid)
    return Xw, si, Y, idx


Xtr, sitr, Ytr, _ = make_window_set("train")
Xva, siva, Yva, idxva = make_window_set("val")
Xte, site, Yte, idxte = make_window_set("test")
print(f"\n[QR-LSTM] window L={L}  train {len(Xtr)} / val {len(Xva)} / test {len(Xte)} origins")


class QRLSTM(nn.Module):
    def __init__(self, F, nstat, hidden=48, emb=6):
        super().__init__()
        self.emb = nn.Embedding(nstat, emb)
        self.lstm = nn.LSTM(F, hidden, batch_first=True)
        trunk = hidden + emb
        self.base = nn.Linear(trunk, nL)
        self.inc = nn.Linear(trunk, nL * (nT - 1))

    def forward(self, x, si):
        out, (hn, cn) = self.lstm(x)
        z = torch.cat([hn[-1], self.emb(si)], dim=1)
        base = self.base(z)                                  # [B,nL]
        inc = torch.nn.functional.softplus(self.inc(z)).view(-1, nL, nT - 1)
        q = torch.cat([base.unsqueeze(-1), base.unsqueeze(-1) + torch.cumsum(inc, -1)], -1)
        return q                                              # [B,nL,nT]


def pinball_loss(q, y, taus):
    err = y.unsqueeze(-1) - q
    l = torch.maximum(taus.view(1, 1, nT) * err, (taus.view(1, 1, nT) - 1) * err)
    return l.mean()


model = QRLSTM(F, S).to(DEV)
opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
taus_t = torch.tensor(TAUS, dtype=torch.float32, device=DEV)
Xtr_t = torch.tensor(Xtr, device=DEV); sitr_t = torch.tensor(sitr, device=DEV); Ytr_t = torch.tensor(Ytr, device=DEV)
Xva_t = torch.tensor(Xva, device=DEV); siva_t = torch.tensor(siva, device=DEV)
EPOCHS = 250; BATCH = 128
best_val = np.inf; best_state = None
n = len(Xtr_t)
for ep in range(EPOCHS):
    model.train(); perm = torch.randperm(n)
    for i in range(0, n, BATCH):
        b = perm[i:i + BATCH]
        opt.zero_grad()
        q = model(Xtr_t[b], sitr_t[b])
        loss = pinball_loss(q, Ytr_t[b], taus_t)
        loss.backward(); opt.step()
    if (ep + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            qv = model(Xva_t, siva_t)
            vl = pinball_loss(qv, torch.tensor(Yva, device=DEV), taus_t).item()
        if vl < best_val:
            best_val = vl; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
model.load_state_dict(best_state)
print(f"  QR-LSTM best val pinball (scaled) = {best_val:.4f}")

# evaluate QR-LSTM per lead (unscale)
model.eval()
lstm_metrics = {}; lstm_Q = {h: {} for h in LEADS}
with torch.no_grad():
    for sp, Xs, sis, idxs in [("val", Xva, siva, idxva), ("test", Xte, site, idxte)]:
        q = model(torch.tensor(Xs, device=DEV), torch.tensor(sis, device=DEV)).cpu().numpy()  # [N,nL,nT]
        for hi, h in enumerate(LEADS):
            Q = q[:, hi, :] * ystd[h]
            y = np.array([DFAA[h][s, t] for s, t in idxs])
            ytr_h = np.array([DFAA[h][s, t] for s, t in spl[h]["train"]])
            ref = ek.crps(np.tile(np.quantile(ytr_h, TAUS), (len(y), 1)), y)
            lstm_metrics.setdefault(h, {})[sp] = ek.all_metrics(Q, y, THETA[h], crps_ref=ref)
            lstm_Q[h][sp] = (Q, y, idxs)

# save raw test/val quantiles for the conformal step (Step 5)
np.savez_compressed(ART / "model_preds.npz",
                    **{f"gbq_{h}_{sp}_Q": gbq_Q[h][sp][0] for h in LEADS for sp in ["val", "test"]},
                    **{f"gbq_{h}_{sp}_y": gbq_Q[h][sp][1] for h in LEADS for sp in ["val", "test"]},
                    **{f"lstm_{h}_{sp}_Q": lstm_Q[h][sp][0] for h in LEADS for sp in ["val", "test"]},
                    **{f"lstm_{h}_{sp}_y": lstm_Q[h][sp][1] for h in LEADS for sp in ["val", "test"]},
                    **{f"lstm_{h}_{sp}_idx": lstm_Q[h][sp][2] for h in LEADS for sp in ["val", "test"]})

out = {"gbq": gbq_metrics, "lstm": lstm_metrics}
(ART / "model_metrics.json").write_text(json.dumps(out, indent=2, default=float))

print("\n" + "=" * 70); print("MODEL SUMMARY (TEST) - CRPS / CRPSS / PICP90 / DTF-AUC / FTD-AUC"); print("=" * 70)
for h in LEADS:
    print(f"\n-- lead h={h} --")
    print(f"{'model':10s} {'CRPS':>7s} {'CRPSS':>7s} {'PICP80':>7s} {'PICP90':>7s} {'DTF-AUC':>8s} {'FTD-AUC':>8s}")
    for name, mm in [("GBQ", gbq_metrics), ("QR-LSTM", lstm_metrics)]:
        m = mm[h]["test"]; ev = m["event"]
        print(f"{name:10s} {m['crps']:7.4f} {m.get('crpss', float('nan')):7.3f} {m['picp80']:7.3f} "
              f"{m['picp90']:7.3f} {ev['DTF']['auc']:8.3f} {ev['FTD']['auc']:8.3f}")
print(f"\nWrote {ART/'model_metrics.json'} and {ART/'model_preds.npz'}")
print("\nSTEP 4 OK.")
