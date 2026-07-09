"""Assemble shipped notebooks/*.ipynb from the reviewed .build/*.py sources (Rule 9).

Each notebook is self-contained and runs top-to-bottom on Colab (GPU/CPU) or local:
  cell 1  : guarded pip + device detect + data-path auto-detect -> sets env DFAA_ROOT
  cell 2  : %%writefile common.py   (+ evalkit.py where needed)
  cell 3+ : the step logic (imports the just-written modules)
"""
import json
from pathlib import Path

BUILD = Path(__file__).parent
NB = BUILD.parent / "notebooks"
NB.mkdir(exist_ok=True)

SETUP = r"""# === Colab/local auto-setup (device + data path) ===
import sys, os, subprocess
from pathlib import Path
def _pip(*pkgs):
    for p in pkgs:
        mod = p.split('==')[0].replace('-', '_').replace('scikit_learn', 'sklearn')
        try:
            __import__(mod)
        except Exception:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', p])
_pip('numpy', 'pandas', 'scipy', 'scikit-learn', 'statsmodels', 'torch', 'matplotlib')
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', DEVICE)
CSV = 'Bangladesh Meterological data.csv'
cands = [Path.cwd()/CSV, Path('/content')/CSV, Path(r'd:\BUET RESEARCH WORK\Bangladesh Flood')/CSV]
root = next((c.parent for c in cands if c.exists()), None)
if root is None:
    try:
        from google.colab import files
        files.upload(); root = Path.cwd()
    except Exception:
        raise FileNotFoundError('Upload "%s" next to this notebook.' % CSV)
os.environ['DFAA_ROOT'] = str(root)
print('DFAA_ROOT =', root)
"""


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.splitlines(keepends=True)}


def strip_module_imports(src, mods=("common", "evalkit")):
    """Keep the step code intact (it imports the writefile'd modules)."""
    return src


def writefile_cell(fname):
    body = (BUILD / fname).read_text(encoding="utf-8")
    return code(f"%%writefile {fname}\n" + body)


STEPS = [
    ("01_explore_clean.ipynb", "step01_explore_clean.py", ["common.py"],
     "# 01 — Explore, clean, train-only climatology\nProfiles the 13-station monthly record, reports gaps, "
     "fits leak-free climatology. Writes `artefacts/clima.json`. (Data prep — not an experiment.)"),
    ("02_build_dfaa_index.ipynb", "step02_build_dfaa_index.py", ["common.py"],
     "# 02 — Composite wetness W + DFAA index + event labels\nBuilds the forecast target (Eq. 2) and "
     "train-fit event thresholds. Writes `artefacts/dfaa.npz`."),
    ("03_features_baselines.ipynb", "step03_features_baselines.py", ["common.py", "evalkit.py"],
     "# 03 — PET, physics-guided features, probabilistic baselines\nClimatology / seasonal / persistence / "
     "water-balance bucket / SARIMA. Writes `artefacts/features.npz`, `baseline_metrics.json`."),
    ("04_quantile_models.ipynb", "step04_quantile_models.py", ["common.py", "evalkit.py"],
     "# 04 — Quantile models (GBQ + QR-LSTM)\nGradient-boosting quantile regression and a compact "
     "monotone-head QR-LSTM. Writes `model_metrics.json`, `model_preds.npz`."),
    ("05_conformal_calibration.ipynb", "step05_conformal.py", ["common.py", "evalkit.py"],
     "# 05 — Split-conformal calibration (CQR) + reliability\nRestores coverage on test; PIT/ECE; "
     "calibrated event probabilities. Writes `conformal_metrics.json`."),
    ("06_interpret_eval.ipynb", "step06_interpret_eval.py", ["common.py", "evalkit.py"],
     "# 06 — Interpretation & evaluation\nLead-time skill, physics-feature ablation, permutation importance, "
     "cost–loss value, LOSO transfer. Writes `interpret_metrics.json`."),
    ("make_paper_figs.ipynb", "make_paper_figs.py", ["common.py"],
     "# Figures — regenerate manuscript figures from logged artefacts\nReads only `artefacts/*` — no model "
     "re-runs. Writes `paper/figs/*.png`."),
]

for nbname, stepfile, mods, title in STEPS:
    cells = [md(title), code(SETUP)]
    for mfile in mods:
        cells.append(writefile_cell(mfile))
    step_src = (BUILD / stepfile).read_text(encoding="utf-8")
    cells.append(code(step_src))
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": "3.x"}},
          "nbformat": 4, "nbformat_minor": 5}
    (NB / nbname).write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote", NB / nbname)
print("All notebooks assembled.")
