"""Publication-grade matplotlib style + multi-format export (CLAUDE.md Rule 11).
Okabe-Ito colorblind-safe palette; serif typography; PNG + PDF + TIFF(LZW) at 400 dpi.
"""
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito colorblind-safe palette
OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
      "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7",
      "grey": "#7F7F7F"}
# semantic roles
C = {"gbq": OI["blue"], "lstm": OI["orange"], "bucket": OI["green"], "clim": OI["grey"],
     "dtf": OI["vermillion"], "ftd": OI["sky"], "wet": "#2C7FB8", "dry": "#B35806",
     "phys": OI["orange"], "base": OI["blue"], "accent": OI["purple"]}


def set_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 9, "axes.titlesize": 9.5, "axes.titleweight": "bold",
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 7.6, "legend.frameon": False, "legend.handlelength": 1.6,
        "axes.linewidth": 0.7, "axes.edgecolor": "#333333",
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "axes.grid": True, "grid.color": "#B0B0B0", "grid.alpha": 0.30, "grid.linewidth": 0.45,
        "figure.dpi": 150, "savefig.dpi": 400, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
        "lines.linewidth": 1.6, "lines.markersize": 5, "patch.linewidth": 0.6,
    })


def panel_tag(ax, tag, x=-0.15, y=1.10):
    ax.text(x, y, tag, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom", ha="left")


def save_fig(fig, path_stem):
    """Save PNG + PDF + TIFF(LZW). path_stem is a pathlib.Path without extension."""
    fig.savefig(str(path_stem) + ".png", dpi=400)
    fig.savefig(str(path_stem) + ".pdf")
    try:
        fig.savefig(str(path_stem) + ".tiff", dpi=400, pil_kwargs={"compression": "tiff_lzw"})
    except Exception:
        try:
            from PIL import Image
            Image.open(str(path_stem) + ".png").save(str(path_stem) + ".tiff", compression="tiff_lzw")
        except Exception as e:
            warnings.warn(f"TIFF export failed for {path_stem.name}: {e}")
    plt.close(fig)
