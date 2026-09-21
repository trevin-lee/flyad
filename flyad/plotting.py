"""Shared figure styling for both studies.

Colour does the work of the argument: the fly owns the accent hue, the AXOL1TL
variants form two families, and generic baselines recede to grey, so "fly versus
everything else" reads before anyone parses a legend. The palette was checked for
colourblind separation and contrast against the chart surface.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FLY = "#8f6414"       # the fly's own chitin ochre
AXO_CMS = "#00856f"   # AXOL1TL as deployed by CMS
AXO_RE = "#7038c4"    # AXOL1TL retrained on this background
GREY_D = "#5d686e"
GREY_L = "#98a1a6"
INK = "#1c2226"
RULE = "#c9c4b8"
WARN = "#c0392e"

STYLE = {
    "fly":              dict(color=FLY, lw=2.4, marker="o", ms=6, zorder=5, label="Fly olfactory circuit"),
    "gaussian":         dict(color=GREY_L, lw=1.6, marker="s", ms=4, ls="-",  label="Gaussian"),
    "autoencoder":      dict(color=GREY_D, lw=1.6, marker="^", ms=4, ls="-",  label="Autoencoder"),
    "axol1tl_v3":       dict(color=AXO_CMS, lw=1.8, marker="D", ms=4, ls="-",  label="AXOL1TL v3 (CMS)"),
    "axol1tl_v5":       dict(color=AXO_CMS, lw=1.8, marker="D", ms=4, ls="--", label="AXOL1TL v5 (CMS)"),
    "axo_v3_retrained": dict(color=AXO_RE, lw=1.8, marker="v", ms=4, ls="-",  label="AXOL1TL v3 (retrained)"),
    "axo_v5_retrained": dict(color=AXO_RE, lw=1.8, marker="v", ms=4, ls="--", label="AXOL1TL v5 (retrained)"),
}
ORDER = ["fly", "gaussian", "autoencoder",
         "axo_v3_retrained", "axo_v5_retrained", "axol1tl_v3", "axol1tl_v5"]
EPS = [1e-2, 1e-3, 1e-4, 1e-5]


def setup() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "font.family": "sans-serif", "font.size": 9,
        "axes.edgecolor": RULE, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK,
        "axes.grid": True, "grid.color": "#e8e4da", "grid.linewidth": 0.8,
        "axes.axisbelow": True, "legend.frameon": False, "savefig.bbox": "tight",
    })


def save(fig, name: str, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"{name}.{ext}", dpi=200)
    plt.close(fig)
    print(f"  wrote {outdir.name}/{name}.png|pdf", flush=True)
