"""Publication-grade manuscript figures (CLAUDE.md Rule 11).
Reads ONLY verified artefacts (no model re-runs); every value traces to RESULTS_LOG.
Outputs paper/figs/*.{png,pdf,tiff} at 400 dpi, Okabe-Ito colorblind-safe, serif typography.
All on-data labels carry a white halo (path_effects) and live in cleared headroom -> no overlaps.
"""
import json
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from common import ART, ROOT, LEADS, TAUS, STATION_LATLON
import evalkit as ek
from figstyle import set_style, save_fig, panel_tag, C, OI

set_style()
HALO = [pe.withStroke(linewidth=2.4, foreground="white")]
FIG = ROOT / "paper" / "figs"; FIG.mkdir(parents=True, exist_ok=True)

base = json.loads((ART / "baseline_metrics.json").read_text())
modl = json.loads((ART / "model_metrics.json").read_text())
conf = json.loads((ART / "conformal_metrics.json").read_text())
intp = json.loads((ART / "interpret_metrics.json").read_text())
meta = json.loads((ART / "dfaa_meta.json").read_text())
dz = np.load(ART / "dfaa.npz", allow_pickle=True)
pp = np.load(ART / "model_preds.npz", allow_pickle=True)
stations = [str(x) for x in dz["stations"]]
W = dz["W"]; D2 = dz["DFAA_h2"]; LAB2 = dz["lab_h2"]; WE2 = dz["WE_h2"]; WL2 = dz["WL_h2"]
T = W.shape[1]; tvec = 2000 + np.arange(T) / 12.0
yrT = (2000 + np.arange(T) // 12)
TH = {int(h): meta["theta_D"][h] for h in meta["theta_D"]}
ALPHA = meta["alpha"]
hs = list(LEADS)
LEADC = {1: OI["blue"], 2: OI["vermillion"], 3: OI["green"]}


# ============================ FIG 1 — mechanism + intensification ============================
def fig1():
    fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.9))
    d = D2.copy(); d[~np.isfinite(d)] = -np.inf
    s_, t_ = np.unravel_index(np.argmax(d), d.shape)
    lo, hi = max(t_ - 14, 0), min(t_ + 14, T - 1)
    tt = tvec[lo:hi + 1]; ww = W[s_, lo:hi + 1]
    a = ax[0]
    a.axhline(0, color="#444", lw=0.6, zorder=1)
    a.fill_between(tt, 0, ww, where=ww >= 0, interpolate=True, color=C["wet"], alpha=0.5, lw=0, label="wet (W>0)")
    a.fill_between(tt, 0, ww, where=ww < 0, interpolate=True, color=C["dry"], alpha=0.5, lw=0, label="dry (W<0)")
    a.plot(tt, ww, color="#222", lw=1.3, zorder=3)
    we = float(np.mean(W[s_, t_ - 1:t_ + 1])); wl = float(np.mean(W[s_, t_ + 1:t_ + 3]))
    a.axvspan(tvec[t_ - 1], tvec[t_], color=C["dry"], alpha=0.16, zorder=0)
    a.axvspan(tvec[t_ + 1], tvec[t_ + 2], color=C["wet"], alpha=0.16, zorder=0)
    a.annotate("", xy=(tvec[t_ + 1] + 0.04, wl), xytext=(tvec[t_] - 0.04, we),
               arrowprops=dict(arrowstyle="-|>", color=OI["purple"], lw=2.0))
    wabs = np.nanmax(np.abs(ww))
    a.set_ylim(-wabs * 1.30, wabs * 2.05)                      # headroom for annotation + legend
    ylo, yhi = a.get_ylim()
    a.text(tt[1], yhi * 0.97, f"DFAA = {D2[s_, t_]:.2f}\ndrought $\\to$ flood flip", color=OI["purple"],
           fontsize=7.4, va="top", ha="left", fontweight="bold", path_effects=HALO)
    a.text((tvec[t_ - 1] + tvec[t_]) / 2, ylo * 0.78, "E", color=C["dry"], fontweight="bold",
           ha="center", fontsize=8.5, path_effects=HALO)
    a.text((tvec[t_ + 1] + tvec[t_ + 2]) / 2, wl + wabs * 0.22, "L", color=C["wet"], fontweight="bold",
           ha="center", fontsize=8.5, path_effects=HALO)
    a.set_xlabel("year"); a.set_ylabel("wetness anomaly $W$")
    a.set_title(f"DFAA mechanism — {stations[s_]}")
    a.legend(loc="upper right", ncol=1, fontsize=6.6, borderpad=0.4)
    panel_tag(a, "(a)")

    b = ax[1]
    tr = np.abs(D2[:, (yrT >= 2000) & (yrT <= 2014)]); tr = tr[np.isfinite(tr)]
    te = np.abs(D2[:, (yrT >= 2018) & (yrT <= 2022)]); te = te[np.isfinite(te)]
    th = TH[2]
    rate_tr = 100 * (tr >= th).mean(); rate_te = 100 * (te >= th).mean()
    bins = np.linspace(0, 2.2, 34)
    b.hist(tr, bins=bins, density=True, color=OI["sky"], alpha=0.45, label=f"train 2000–14  ({rate_tr:.0f}% events)")
    b.hist(te, bins=bins, density=True, color=OI["vermillion"], alpha=0.40, label=f"test 2018–22  ({rate_te:.0f}% events)")
    xx = np.linspace(0, 2.2, 200)
    b.plot(xx, gaussian_kde(tr)(xx), color=OI["blue"], lw=1.6)
    b.plot(xx, gaussian_kde(te)(xx), color=OI["vermillion"], lw=1.6)
    b.axvline(th, color="#333", ls="--", lw=1.0)
    b.text(th + 0.04, b.get_ylim()[1] * 0.96, r"$\theta_D$", fontsize=8.5, path_effects=HALO)
    b.set_xlabel("|DFAA|  (h=2)"); b.set_ylabel("density")
    b.set_title("Whiplash intensifies in recent years")
    b.legend(loc="upper right", fontsize=6.8)
    panel_tag(b, "(b)")
    fig.tight_layout(w_pad=2.4)
    save_fig(fig, FIG / "fig1_mechanism_intensification")


# ============================ FIG 2 — spatial hotspot + transfer maps ============================
# manual label placement (dx pt, dy pt, ha, va) to avoid the central cluster overlaps
LBL = {
    "Rangpur": (0, 9, "center", "bottom"), "Bogra": (0, 9, "center", "bottom"),
    "Rajshahi": (-8, 0, "right", "center"), "Mymensingh": (0, 9, "center", "bottom"),
    "Sylhet": (0, 9, "center", "bottom"), "Dhaka": (8, 5, "left", "bottom"),
    "Faridpur": (-8, 2, "right", "bottom"), "Comilla": (8, -1, "left", "center"),
    "Jessore": (-8, -2, "right", "top"), "Khulna": (-3, -11, "center", "top"),
    "Barisal": (8, 0, "left", "center"), "Chittagong(Air-port)": (-8, 1, "right", "bottom"),
    "Cox's Bazar": (-8, -2, "right", "top"),
}


def fig2():
    lats = np.array([STATION_LATLON[s][0] for s in stations])
    lons = np.array([STATION_LATLON[s][1] for s in stations])
    freq = np.array([100 * np.nansum(np.abs(LAB2[i]) == 1) / max(np.isfinite(LAB2[i]).sum(), 1)
                     for i in range(len(stations))])
    loso = np.array([intp["loso_h2"].get(s, {}).get("crpss", np.nan) for s in stations])
    asp = 1.0 / np.cos(np.radians(lats.mean()))
    fig, ax = plt.subplots(1, 2, figsize=(7.16, 3.7))

    def mapax(a, vals, cmap, title, label):
        a.set_facecolor("#EFF3F6")
        sc = a.scatter(lons, lats, c=vals, s=150, cmap=cmap, edgecolor="#222", lw=0.8, zorder=3)
        for i, s in enumerate(stations):
            dx, dy, ha, va = LBL[s]
            short = s.replace("(Air-port)", "")
            a.annotate(short, (lons[i], lats[i]), textcoords="offset points", xytext=(dx, dy),
                       ha=ha, va=va, fontsize=5.8, color="#111", path_effects=HALO, zorder=4)
        a.set_aspect(asp)
        a.set_xlim(lons.min() - 1.05, lons.max() + 0.85); a.set_ylim(lats.min() - 0.85, lats.max() + 1.0)
        a.set_xlabel("longitude (°E)"); a.set_ylabel("latitude (°N)"); a.set_title(title)
        a.grid(True, alpha=0.3)
        cb = fig.colorbar(sc, ax=a, fraction=0.046, pad=0.03)
        cb.set_label(label, fontsize=7.5); cb.ax.tick_params(labelsize=7)
    mapax(ax[0], freq, "YlOrRd", "DFAA-event frequency (h=2)", "events (% of months)")
    panel_tag(ax[0], "(a)")
    mapax(ax[1], loso, "viridis", "LOSO transfer skill (h=2)", "CRPSS at held-out station")
    ax[1].text(0.5, -0.30, f"all 13 stations CRPSS>0   (mean +{np.nanmean(loso):.3f})",
               transform=ax[1].transAxes, ha="center", fontsize=7.4, style="italic", color="#0B6E4F")
    panel_tag(ax[1], "(b)")
    fig.tight_layout(w_pad=3.0)
    save_fig(fig, FIG / "fig2_spatial_maps")


# ============================ FIG 3 — lead-time skill + event AUC ============================
def fig3():
    fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.9))
    a = ax[0]
    gbq = [intp["lead_time"]["gbq_crpss"][str(h)] for h in hs]
    lstm = [intp["lead_time"]["lstm_crpss"][str(h)] for h in hs]
    buck = [intp["lead_time"]["bucket_crpss"][str(h)] for h in hs]
    a.axhspan(0, 0.22, color=OI["green"], alpha=0.06)
    a.axhline(0, color=C["clim"], lw=1.1)
    a.plot(hs, gbq, "o-", color=C["gbq"], label="GBQ", zorder=5)
    a.plot(hs, lstm, "s-", color=C["lstm"], label="QR-LSTM")
    a.plot(hs, buck, "^--", color=C["bucket"], label="water-balance bucket")
    for h, v in zip(hs, gbq):
        a.annotate(f"+{v:.3f}", (h, v), textcoords="offset points", xytext=(0, 8),
                   fontsize=6.8, color=C["gbq"], ha="center", fontweight="bold", path_effects=HALO)
    a.text(hs[-1] + 0.03, -0.018, "climatology (skill 0)", fontsize=6.6, color=C["clim"],
           ha="right", va="top", path_effects=HALO)
    a.set_xticks(hs); a.set_xlabel("lead time h (months)"); a.set_ylabel("CRPSS vs climatology")
    a.set_title("Probabilistic skill grows with lead"); a.legend(loc="lower right")
    a.set_ylim(-0.34, 0.24); a.set_xlim(0.8, 3.2)
    panel_tag(a, "(a)")

    b = ax[1]
    dtf = [modl["gbq"][str(h)]["test"]["event"]["DTF"]["auc"] for h in hs]
    ftd = [modl["gbq"][str(h)]["test"]["event"]["FTD"]["auc"] for h in hs]
    bdtf = [base[str(h)]["test"]["bucket"]["event"]["DTF"]["auc"] for h in hs]
    b.axhline(0.5, color="#333", ls=":", lw=0.9)
    b.text(hs[0] - 0.02, 0.515, "no skill", fontsize=6.6, ha="left", color="#666", path_effects=HALO)
    b.plot(hs, dtf, "o-", color=C["dtf"], label="GBQ — DTF")
    b.plot(hs, ftd, "s-", color=C["ftd"], label="GBQ — FTD")
    b.plot(hs, bdtf, "^--", color=C["bucket"], label="bucket — DTF")
    b.set_xticks(hs); b.set_ylim(0.45, 0.9); b.set_xlim(0.8, 3.2); b.set_xlabel("lead time h (months)")
    b.set_ylabel("event ROC-AUC"); b.set_title("Event detection (DTF / FTD)"); b.legend(loc="lower right")
    panel_tag(b, "(b)")
    fig.tight_layout(w_pad=2.4)
    save_fig(fig, FIG / "fig3_skill_leadtime")


# ============================ FIG 4 — calibration ============================
def cqr_apply(Qcal, ycal, Qte):
    Qc = Qte.copy()
    for lo_i, hi_i, al in [(0, 6, 0.10), (1, 5, 0.20), (2, 4, 0.50)]:
        E = np.maximum(Qcal[:, lo_i] - ycal, ycal - Qcal[:, hi_i])
        k = min(int(np.ceil((len(E) + 1) * (1 - al))), len(E)); e = np.sort(E)[k - 1]
        Qc[:, lo_i] -= e; Qc[:, hi_i] += e
    return np.sort(Qc, 1)


def fig4():
    fig, ax = plt.subplots(1, 2, figsize=(7.16, 3.0))
    a = ax[0]
    grid = np.linspace(0.02, 0.98, 33)
    a.plot([0, 1], [0, 1], color="#333", lw=1.0, zorder=1)
    a.text(0.66, 0.60, "perfect", rotation=40, fontsize=6.8, color="#333", path_effects=HALO,
           rotation_mode="anchor", va="bottom", ha="center")
    for h in hs:
        Qte = pp[f"gbq_{h}_test_Q"]; yte = pp[f"gbq_{h}_test_y"]
        Qcal = pp[f"gbq_{h}_val_Q"]; ycal = pp[f"gbq_{h}_val_y"]
        pit_raw = ek.cdf_at(Qte, yte); pit_cqr = ek.cdf_at(cqr_apply(Qcal, ycal, Qte), yte)
        a.plot(grid, [np.mean(pit_raw <= p) for p in grid], "--", color=LEADC[h], lw=1.0, alpha=0.6)
        a.plot(grid, [np.mean(pit_cqr <= p) for p in grid], "-", color=LEADC[h], lw=1.8, label=f"h={h}")
    a.set_xlabel("nominal probability"); a.set_ylabel("observed frequency")
    a.set_title("Reliability:  raw (--) $\\to$ CQR (—)")
    a.set_xlim(0, 1); a.set_ylim(0, 1)
    a.legend(loc="lower right", title="after CQR", title_fontsize=7)
    a.text(0.04, 0.93, "raw curves sag below\ndiagonal (under-coverage)", fontsize=6.4,
           ha="left", va="top", color="#555", style="italic", path_effects=HALO)
    panel_tag(a, "(a)")

    b = ax[1]
    x = np.arange(len(hs)); w = 0.18
    p90b = [conf["gbq"][str(h)]["before"]["picp90"] for h in hs]
    p90a = [conf["gbq"][str(h)]["after"]["picp90"] for h in hs]
    p80b = [conf["gbq"][str(h)]["before"]["picp80"] for h in hs]
    p80a = [conf["gbq"][str(h)]["after"]["picp80"] for h in hs]
    b.axhline(0.9, color=OI["vermillion"], lw=0.9, ls="--"); b.axhline(0.8, color=OI["blue"], lw=0.9, ls="--")
    b.text(len(hs) - 0.55, 0.908, "90% target", fontsize=6.2, color=OI["vermillion"], ha="right", path_effects=HALO)
    b.text(len(hs) - 0.55, 0.808, "80% target", fontsize=6.2, color=OI["blue"], ha="right", path_effects=HALO)
    b.bar(x - 1.5 * w, p80b, w, color=OI["blue"], alpha=0.4, label="80% raw")
    b.bar(x - 0.5 * w, p80a, w, color=OI["blue"], label="80% CQR")
    b.bar(x + 0.5 * w, p90b, w, color=OI["vermillion"], alpha=0.4, label="90% raw")
    b.bar(x + 1.5 * w, p90a, w, color=OI["vermillion"], label="90% CQR")
    b.set_xticks(x); b.set_xticklabels([f"h={h}" for h in hs]); b.set_ylim(0.4, 1.12)
    b.set_ylabel("PICP (coverage)"); b.set_title("CQR restores coverage")
    b.legend(loc="upper center", ncol=4, fontsize=6.0, columnspacing=0.8, handlelength=1.1, handletextpad=0.4)
    panel_tag(b, "(b)")
    fig.tight_layout(w_pad=2.4)
    save_fig(fig, FIG / "fig4_calibration")


# ============================ FIG 5 — drivers + cost-loss value ============================
def fig5():
    fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.9))
    a = ax[0]
    PHYS = {"PET", "D", "aridity", "API", "dSM", "dSPEI", "roll3", "roll6"}
    imp = intp["perm_importance_h2"]
    items = sorted(imp.items(), key=lambda kv: kv[1])[-10:]
    names = [k for k, v in items]; vals = [v for k, v in items]
    cap = 0.0088
    disp = [min(v, cap) for v in vals]
    cols = [C["phys"] if n in PHYS else (C["accent"] if n == "WE" else C["base"]) for n in names]
    a.barh(np.arange(len(names)), disp, color=cols, edgecolor="#333", lw=0.4)
    for i, (n, v) in enumerate(zip(names, vals)):
        if v > cap:
            a.text(cap * 0.97, i, f"{v:.3f} »", va="center", ha="right", fontsize=7.0,
                   color="white", fontweight="bold")
    a.set_yticks(np.arange(len(names))); a.set_yticklabels([n.replace("WE", "$W_E$") for n in names], fontsize=7.2)
    a.set_xlabel(r"$\Delta$ pinball when permuted"); a.set_xlim(0, cap * 1.05)
    a.set_title("Drivers (GBQ, h=2)")
    from matplotlib.patches import Patch
    a.legend(handles=[Patch(color=C["accent"], label="known state $W_E$"),
                      Patch(color=C["base"], label="EO / calendar"),
                      Patch(color=C["phys"], label="physics-guided")], loc="lower right", fontsize=6.6)
    panel_tag(a, "(a)")

    b = ax[1]
    for nm, col in [("DTF", C["dtf"]), ("FTD", C["ftd"])]:
        cl = intp["cost_loss_h2"][nm]; r = np.array(cl["r"]); v = np.array(cl["value"])
        b.plot(r, v, color=col, label=f"{nm}  ($V_{{max}}$={cl['vmax']:.2f})")
        b.fill_between(r, 0, v, where=v > 0, color=col, alpha=0.12)
        b.scatter([cl["r_at_vmax"]], [cl["vmax"]], color=col, s=30, zorder=5, edgecolor="white", lw=0.7)
    b.axhline(0, color="#333", lw=0.8)
    b.set_xlabel("cost / loss ratio  $C/L$"); b.set_ylabel("decision value $V$")
    b.set_xlim(0, 1); b.set_ylim(-0.05, 0.56); b.set_title("Operational value (GBQ+CQR, h=2)")
    b.legend(loc="upper right")
    panel_tag(b, "(b)")
    fig.tight_layout(w_pad=2.4)
    save_fig(fig, FIG / "fig5_drivers_costloss")


# ============================ FIG 6 — DFAA index surface (CONTOUR) ============================
def fig6():
    th = TH[2]
    g = np.linspace(-2.6, 2.6, 400)
    GE, GL = np.meshgrid(g, g)
    Z = (GL - GE) * (np.abs(GE) + np.abs(GL)) * ALPHA ** (-np.abs(GE + GL))   # Eq. 2
    lim = float(np.nanpercentile(np.abs(Z), 99))
    levels = np.linspace(-lim, lim, 21)
    fig, ax = plt.subplots(1, 2, figsize=(7.16, 3.35))

    a = ax[0]
    cf = a.contourf(GE, GL, Z, levels=levels, cmap="RdBu_r", extend="both")
    cs = a.contour(GE, GL, Z, levels=[-th, 0, th], colors="k", linewidths=[1.3, 0.6, 1.3],
                   linestyles=["--", "-", "--"])
    a.clabel(cs, fmt={-th: r"$-\theta_D$", 0.0: "0", th: r"$+\theta_D$"}, fontsize=6.5, inline=True)
    a.plot([-2.6, 2.6], [-2.6, 2.6], color="k", lw=0.7, alpha=0.45)
    a.text(-1.7, 2.05, "Drought$\\to$Flood\n(DTF, +)", ha="center", va="center", fontsize=7.2,
           fontweight="bold", color="#7A0177", path_effects=HALO)
    a.text(1.7, -2.05, "Flood$\\to$Drought\n(FTD, −)", ha="center", va="center", fontsize=7.2,
           fontweight="bold", color="#08519C", path_effects=HALO)
    a.text(1.55, 1.95, "persistent\n(suppressed)", ha="center", va="center", fontsize=6.4,
           style="italic", color="#222", rotation=45, path_effects=HALO)
    a.set_aspect("equal"); a.set_xlim(-2.6, 2.6); a.set_ylim(-2.6, 2.6)
    a.set_xlabel("early-window wetness  $W_E$"); a.set_ylabel("late-window wetness  $W_L$")
    a.set_title("DFAA index surface (Eq. 2)")
    cb = fig.colorbar(cf, ax=a, fraction=0.046, pad=0.03); cb.set_label("DFAA", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    panel_tag(a, "(a)")

    b = ax[1]
    m = np.isfinite(WE2) & np.isfinite(WL2) & np.isfinite(LAB2)
    we, wl, lab = WE2[m], WL2[m], LAB2[m]
    for nm, val, col, z in [("no event", 0, "#B9B9B9", 2), ("DTF", 1, OI["vermillion"], 4),
                            ("FTD", -1, OI["sky"], 3)]:
        sel = lab == val
        b.scatter(we[sel], wl[sel], s=7, color=col, alpha=0.55, lw=0, zorder=z,
                  label=f"{nm}  (n={int(sel.sum())})")
    b.contour(GE, GL, Z, levels=[-th, th], colors="k", linewidths=1.1, linestyles="--", zorder=5)
    b.plot([-2.6, 2.6], [-2.6, 2.6], color="k", lw=0.7, alpha=0.45)
    b.set_aspect("equal"); b.set_xlim(-2.6, 2.6); b.set_ylim(-2.6, 2.6)
    b.set_xlabel("early-window wetness  $W_E$"); b.set_ylabel("late-window wetness  $W_L$")
    b.set_title("Observed station-months (h=2)")
    leg = b.legend(loc="upper left", fontsize=6.6, markerscale=1.6, handletextpad=0.3)
    leg.set_zorder(6)
    panel_tag(b, "(b)")
    fig.tight_layout(w_pad=2.6)
    save_fig(fig, FIG / "fig6_dfaa_surface_contour")


for fn in (fig1, fig2, fig3, fig4, fig5, fig6):
    fn(); print("done", fn.__name__, flush=True)
print("\nFigures written to", FIG)
for p in sorted(FIG.glob("*.png")):
    print("  ", p.stem)
