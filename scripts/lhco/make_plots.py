"""Comparison figures for the LHC Olympics dijet study.

Run:  python scripts/lhco/make_plots.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402

from flyad.plotting import (  # noqa: E402
    AXO_CMS, AXO_RE, EPS, FLY, GREY_D, GREY_L, INK, ORDER, RULE, STYLE, WARN,
    save as _save, setup,
)

OUT = ROOT / "outputs" / "lhco" / "figures"
RESULTS = ROOT / "outputs" / "lhco"


def save(fig, name):
    _save(fig, name, OUT)


def fig_lhco(res: dict) -> None:
    """Full ROC and SIC curves, where every detector's scores were kept."""
    from sklearn.metrics import roc_curve

    d = np.load(RESULTS / "roc.npz")
    y = d["y_test"]
    pairs = [("fly", "score_fly"), ("gaussian", "score_gaussian"),
             ("autoencoder", "score_ae")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    grid = np.logspace(-4, 0, 260)
    for key, skey in pairs:
        if skey not in d:
            continue
        fpr, tpr, _ = roc_curve(y, d[skey], drop_intermediate=False)
        t = np.interp(grid, fpr, tpr)
        t = np.where(t <= 0, np.nan, t)
        st = {k: v for k, v in STYLE[key].items() if k != "marker" and k != "ms"}
        a1.plot(grid, t, **st)
        a2.plot(grid, t / np.sqrt(grid), **st)
    a1.plot(grid, grid, color=GREY_D, lw=1.1, ls=":", label="random")
    a2.axhline(1.0, color=WARN, lw=1.2, ls=":")
    for ax in (a1, a2):
        ax.set_xscale("log"); ax.invert_xaxis()
        ax.set_xlabel(r"background efficiency  $\varepsilon_B$")
    a1.set_yscale("log"); a1.set_ylabel(r"signal efficiency  $\varepsilon_S$")
    a1.set_title("ROC (LHC Olympics dijets)", fontsize=10)
    a2.set_yscale("log"); a2.set_ylabel(r"SIC")
    a2.set_title("Significance improvement", fontsize=10)
    a2.text(1e-4, 1.1, "cut helps above this", fontsize=7.5, color=WARN)
    a1.legend(fontsize=8, loc="lower left")
    fig.suptitle("Dijet benchmark: all three fall below random in the tail",
                 fontsize=11, y=1.0)
    save(fig, "01_roc_and_sic")


def fig_sweep(res: dict) -> None:
    sw = res["kc_sweep"]
    n = [p["n_kc"] for p in sw]; a = [p["auc"] for p in sw]
    peak = max(sw, key=lambda p: p["auc"])
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(n, a, color=FLY, lw=2.2, marker="o", ms=5)
    ax.axvline(2000, color=FLY, lw=1.1, ls=":")
    ax.annotate("a real fly\nhas ~2,000", (2000, min(a)), textcoords="offset points",
                xytext=(8, 6), fontsize=8, color=FLY)
    ax.scatter([peak["n_kc"]], [peak["auc"]], s=130, facecolor="none",
               edgecolor=FLY, linewidth=1.8, zorder=6)
    ax.annotate(f"peak {peak['auc']:.4f}", (peak["n_kc"], peak["auc"]),
                textcoords="offset points", xytext=(6, 8), fontsize=8, color=FLY)
    ax.set_xscale("log")
    ax.set_xlabel("Kenyon cells (expansion width)"); ax.set_ylabel("AUC")
    ax.set_title("The expansion does real work, up to a point", fontsize=11)
    save(fig, "02_kenyon_cell_sweep")


def _mjj(idx):
    """Dijet invariant mass for the held-out test events."""
    import pandas as pd
    a = pd.read_hdf(ROOT / "data" / "lhco" / "events_anomalydetection_v2.features.h5").to_numpy()
    px1, py1, pz1, m1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    px2, py2, pz2, m2 = a[:, 7], a[:, 8], a[:, 9], a[:, 10]
    e1 = np.sqrt(px1**2 + py1**2 + pz1**2 + m1**2)
    e2 = np.sqrt(px2**2 + py2**2 + pz2**2 + m2**2)
    m = (e1 + e2) ** 2 - ((px1 + px2) ** 2 + (py1 + py2) ** 2 + (pz1 + pz2) ** 2)
    return np.sqrt(np.maximum(m, 0.0))[idx]


def fig_sculpting() -> None:
    """The check CMS actually publishes: does the cut distort the mass spectrum?

    A tagger correlated with mass sculpts a fake bump into the background and
    destroys the sideband fit a bump hunt depends on. This is the plot that
    matters far more than AUC for anything downstream.
    """
    d = np.load(RESULTS / "roc.npz")
    y = d["y_test"]
    mjj = _mjj(d["test_idx"])
    bkg = y == 0
    base = np.median(mjj[bkg])
    bins = np.linspace(2000, 5000, 70)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (q, lab) in zip(axes, ((0.99, "top 1% of scores"),
                                   (0.999, "top 0.1% of scores"))):
        ax.hist(mjj[bkg], bins=bins, density=True, histtype="stepfilled",
                color="#e3ded2", edgecolor=GREY_D, lw=1.0,
                label="all QCD background")
        for key, skey in (("fly", "score_fly"), ("gaussian", "score_gaussian"),
                          ("autoencoder", "score_ae")):
            if skey not in d:
                continue
            thr = np.quantile(d[skey][bkg], q)
            sel = bkg & (d[skey] >= thr)
            st = STYLE[key]
            med = np.median(mjj[sel])
            ax.hist(mjj[sel], bins=bins, density=True, histtype="step",
                    color=st["color"], lw=2.0,
                    label=f"{st['label']}  ({100*(med-base)/base:+.1f}%)")
        ax.axvline(base, color=GREY_D, lw=1.1, ls=":")
        ax.set_xlabel(r"$m_{JJ}$  [GeV]")
        ax.set_title(lab, fontsize=10)
    axes[0].set_ylabel("normalised")
    axes[0].legend(fontsize=7.8, loc="upper right")
    fig.suptitle("Mass sculpting: does the cut move the background spectrum?",
                 fontsize=11, y=1.0)
    save(fig, "03_mass_sculpting")


def main() -> None:
    setup()
    res = json.loads((RESULTS / "results.json").read_text())
    print("building LHC Olympics figures...", flush=True)
    fig_lhco(res)
    fig_sweep(res)
    fig_sculpting()
    print(f"\nfigures in {OUT}", flush=True)


if __name__ == "__main__":
    main()
