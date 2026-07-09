# LITERATURE_REVIEW.md — cited, DOI-verified review

**Status:** seed review for the Bangladesh DFAA early-warning paper (2026-06-18). Per `CLAUDE.md` Rule 3,
**every DOI must resolve via doi.org before it is cited in any `.md` writeup or the `.tex`.** Verification
key below; the ⚠/● rows still need a doi.org resolution check (logged in the table at the bottom).

Verification key: ✓ = web-confirmed this session · ● = canonical, high-confidence, resolve at write-time ·
⚠ = confirm article id/DOI.

---

## 1A. DFAA — definition & detection (the gap we fill)
| Ref | DOI / id | Status | Contributes |
|---|---|---|---|
| Wu, Li, He et al. (2006) *Large-scale singularities & summer long-cycle drought–flood abrupt alternation, Yangtze*, Sci. Bull. | 10.1007/s11434-006-2060-x | ✓ | **The seminal LDFAI index** (Eq. 2 backbone) |
| Bai, Wang, Wu, Zhang, Zhang (2024) *A novel multivariate multiscale index for DFAA: precipitation, evapotranspiration, soil moisture*, J. Hydrology 643:132039 | 10.1016/j.jhydrol.2024.132039 | ✓ Crossref (art. 132039 confirmed) | **Multivariate MSDFI** — multi-variable, copula-optional extension |
| Bai, Zhao, Tang et al. (2023) *Identification, physical mechanisms and impacts of DFAA: a review*, Front. Earth Sci. 11:1203603 | 10.3389/feart.2023.1203603 | ✓ Crossref | Mechanisms, DTF vs FTD asymmetry, prior index survey. **NB: first authors are Bai & Zhao, not "Shan" — corrected.** |
| (2025) *A systematic review of methods for identifying DFAA*, Front. Environ. Sci. | 10.3389/fenvs.2025.1590613 | ✓ | Method taxonomy; positions our approach |

## 1B. Drought indices (target construction)
| Ref | DOI / id | Status | Contributes |
|---|---|---|---|
| Vicente-Serrano, Beguería, López-Moreno (2010) *SPEI*, J. Climate 23(7):1696–1718 | 10.1175/2009JCLI2909.1 | ✓ | SPEI definition (P−PET, log-logistic) — `SPEI_3` provenance |
| McKee, Doesken, Kleist (1993) *SPI*, 8th Conf. Applied Climatology (AMS) | no DOI (proceedings) | ● | Standardization concept behind our z-anomalies |

## 1C. Calibrated / conformal probabilistic forecasting (our novelty vs BD literature)
| Ref | DOI / id | Status | Contributes |
|---|---|---|---|
| Romano, Patterson, Candès (2019) *Conformalized Quantile Regression*, NeurIPS 32:3538–3548 | arXiv:1905.03222 | ✓ | **CQR** — distribution-free coverage (Eq. 7–8) |
| Xu & Xie (2021) *Conformal prediction for dynamic time-series* (EnbPI), ICML | arXiv:2010.09107 | ● | Time-series-valid conformal variant |
| Kuleshov, Fenner, Ermon (2018) *Accurate uncertainties via calibrated regression*, ICML | arXiv:1807.00263 | ● | Recalibration baseline / reliability |

## 1D. Scoring rules, quantile loss, monotone heads
| Ref | DOI / id | Status | Contributes |
|---|---|---|---|
| Gneiting & Raftery (2007) *Strictly proper scoring rules*, JASA 102(477):359–378 | 10.1198/016214506000001437 | ● | CRPS as headline probabilistic score |
| Koenker & Bassett (1978) *Regression quantiles*, Econometrica 46(1):33–50 | 10.2307/1913643 | ● | Pinball loss foundation (Eq. 6) |
| Cannon (2018) *Non-crossing monotone composite QRNN*, SERRA 32:3207–3225 | 10.1007/s00477-018-1573-6 | ● | **Monotone non-crossing quantile head** (Eq. 5) |

## 1E. Probabilistic forecasting models (our model stack)
| Ref | DOI / id | Status | Contributes |
|---|---|---|---|
| Lim, Arık, Loeff, Pfister (2021) *Temporal Fusion Transformer*, IJF 37(4):1748–1764 | 10.1016/j.ijforecast.2021.03.012 | ● | Interpretable quantile deep model (optional) |
| Duan et al. (2020) *NGBoost*, ICML | arXiv:1910.03225 | ● | Probabilistic gradient boosting (small-data core) |
| Kratzert et al. (2018) *Rainfall-runoff LSTM*, HESS 22:6005–6022 | 10.5194/hess-22-6005-2018 | ● | Precedent for LSTM hydrology / regional modeling |

## 1F. Physics-guided (features + baseline, NOT a PINN)
| Ref | DOI / id | Status | Contributes |
|---|---|---|---|
| Hargreaves & Samani (1985) *Reference crop ET from temperature*, Appl. Eng. Agric. 1(2):96–99 | 10.13031/2013.26773 | ● | **PET from Tmax/Tmin** (Eq. 3) → features + bucket model |
| Raissi, Perdikaris, Karniadakis (2019) *PINNs*, J. Comput. Phys. 378:686–707 | 10.1016/j.jcp.2018.10.045 | ✓ Crossref (**was .561 — WRONG/404, corrected to .045**) | The family we **deliberately avoid**; justifies guided-feature choice |

## 1G. Compound extremes & Bangladesh context (motivation / baselines)
| Ref | DOI / id | Status | Contributes |
|---|---|---|---|
| Zscheischler et al. (2018) *Future climate risk from compound events*, Nat. Clim. Change 8:469–477 | 10.1038/s41558-018-0156-3 | ● | Compound-event framing |
| (2023/24) *Influential weather parameters & seasonal drought prediction in Bangladesh ML*, Sci. Rep. | 10.1038/s41598-023-51111-2 | ✓ | BD drought-ML baseline (point forecast → our calibration gap) |
| (2025) *Ensemble ML for agricultural droughts, Ganges Delta*, Earth Syst. Environ. | 10.1007/s41748-025-00943-1 | ✓ | BD agricultural-drought baseline |
| (2023) *SPI vs SPEI ML in Rangpur, Bangladesh*, Geol. Ecol. Landscapes | 10.1080/24749508.2023.2254003 | ✓ | BD SPEI-ML baseline |

**Optional (stretch extensions, cite only if used):** Prithvi-WxC `arXiv:2409.13598`, ClimaX `arXiv:2301.10343`
(foundation-model probe); Wu et al. Graph WaveNet `arXiv:1906.00121` (spatial GNN).

---

## Synthesis / gap statement (for the paper Intro)

DFAA indices exist (LDFAI, MSDFI) but are **China-centric and used descriptively**; Bangladesh drought-ML
exists but is **point-forecast and single-hazard**. **No work delivers a calibrated, probabilistic,
physics-guided DFAA early-warning for Bangladesh.** That gap is our contribution: (i) first DFAA index +
event climatology for Bangladesh from satellite EO; (ii) a calibrated probabilistic forecast (quantile head +
conformal) turning the index distribution into operational event probabilities; (iii) honest physics-guided
ablation and a water-balance baseline; (iv) a climate-adaptation/cost–loss evaluation lens.

## DOI verification log (Rule 3 — all citations in `paper/main.tex` verified 2026-06-19)
Method: doi.org resolution + Crossref `api.crossref.org/works/{DOI}` title match; arXiv via web. All 18
references in the `.tex` resolve and titles match. One bad DOI found & fixed (Raissi). One authorship
correction (the 2023 review is first-authored by Bai & Zhao, not "Shan").

| DOI / id | Resolved? | Title matches? | Checked | Notes |
|---|---|---|---|---|
| 10.1007/s11434-006-2060-x | Y | Y | 2026-06-19 | Wu & Li, Yangtze long-cycle DFAA singularities |
| 10.1016/j.jhydrol.2024.132039 | Y | Y | 2026-06-19 | Bai et al., MSDFI, J. Hydrol. 643, art. 132039 |
| 10.3389/feart.2023.1203603 | Y | Y | 2026-06-19 | DFAA review; first authors Bai, Zhao (not Shan) |
| 10.1038/s41598-023-51111-2 | Y | Y | 2026-06-19 | Al Mamun et al., BD drought ML; pub. year **2024** |
| 10.1007/s41748-025-00943-1 | Y | Y | 2026-06-19 | Mondol et al., BD agri drought, Ganges delta |
| 10.1080/24749508.2023.2254003 | Y | Y | 2026-06-19 | Akter et al., Rangpur SPI/SPEI ML |
| 10.1038/s41558-018-0156-3 | Y | Y | 2026-06-19 | Zscheischler & Westra, compound events |
| 10.1175/2009JCLI2909.1 | Y | Y | 2026-06-19 | Vicente-Serrano et al., SPEI, J. Climate 23:1696–1718 |
| 10.13031/2013.26773 | Y | Y | 2026-06-19 | Hargreaves & Samani, PET from temperature |
| 10.1007/s00477-018-1573-6 | Y | Y | 2026-06-19 | Cannon, non-crossing monotone QRNN |
| 10.2307/1913643 | Y | Y | 2026-06-19 | Koenker & Bassett, Regression Quantiles, Econometrica 46 |
| 10.1198/016214506000001437 | Y | Y | 2026-06-19 | Gneiting & Raftery, proper scoring rules / CRPS |
| 10.1016/j.jcp.2018.10.045 | Y | Y | 2026-06-19 | Raissi et al., PINNs (**corrected from .561 which 404s**) |
| 10.1016/j.ijforecast.2021.03.012 | Y | Y | 2026-06-19 | Lim et al., Temporal Fusion Transformers |
| 10.5194/hess-22-6005-2018 | Y | Y | 2026-06-19 | Kratzert et al., rainfall–runoff LSTM |
| arXiv:1905.03222 | Y | Y | 2026-06-19 | Romano, Patterson, Candès, CQR, NeurIPS 2019 pp.3538–3548 |
| arXiv:1910.03225 | Y | Y | 2026-06-19 | Duan et al., NGBoost, ICML 2020 |
| arXiv:2010.09107 | Y | Y | 2026-06-19 | Xu & Xie, EnbPI (adaptive conformal), ICML 2021 |
