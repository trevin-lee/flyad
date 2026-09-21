"""Comparison figures for the ADC2021 / AXOL1TL trigger study.

Run:  python scripts/adc2021/make_plots.py
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

OUT = ROOT / "outputs" / "adc2021" / "figures"
RESULTS = ROOT / "outputs" / "adc2021"


def save(fig, name):
    _save(fig, name, OUT)


def fig_sic(trig: dict) -> None:
    """The plot that decides usefulness: SIC = eps_S / sqrt(eps_B)."""
    sigs = list(next(iter(trig["detectors"].values()))["signals"])
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.5), sharey=True)
    for ax, sig in zip(axes, sigs):
        for key in ORDER:
            e = trig["detectors"].get(key)
            if not e:
                continue
            y = np.array([e["signals"][sig][f"eps_s_at_{p:g}"] / np.sqrt(p)
                          for p in EPS], dtype=float)
            # eps_S == 0 means no signal survived at all. On a log axis that is
            # -inf; drawn naively it becomes a vertical plunge to the frame edge
            # that reads like a measured value. Break the line instead, and mark
            # the point so "nothing survived" stays visible.
            dead = y <= 0
            ax.plot(EPS, np.where(dead, np.nan, y), **STYLE[key])
            if dead.any():
                ax.scatter(np.asarray(EPS)[dead],
                           np.full(dead.sum(), ax.get_ylim()[0] if False else 1e-4),
                           marker="x", s=22, color=STYLE[key]["color"],
                           linewidth=1.2, zorder=4)
        ax.axhline(1.0, color=WARN, lw=1.2, ls=":", zorder=1)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"background efficiency  $\varepsilon_B$")
        ax.set_title(sig, fontsize=10)
        ax.invert_xaxis()
    axes[0].set_ylabel(r"SIC  $\varepsilon_S/\sqrt{\varepsilon_B}$")
    axes[0].text(EPS[0], 1.15, "above this line the cut helps",
                 fontsize=7.5, color=WARN, va="bottom")
    axes[0].text(EPS[0], 1.1e-4, "x = no signal survived", fontsize=7,
                 color=GREY_D, va="bottom")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=7, fontsize=8,
               bbox_to_anchor=(0.5, -0.13))
    fig.suptitle("Does the cut actually improve the search?  (ADC2021 trigger objects)",
                 fontsize=11, y=1.02)
    fig.text(0.5, -0.20, "(CMS) = real emulator run OFF-DISTRIBUTION (trained on Zero Bias, evaluated on Delphes/lepton-filtered ADC2021).  (retrained) = same topology refit here.", ha="center", fontsize=7, color=GREY_D)
    save(fig, "01_significance_improvement")


def fig_auc(trig: dict) -> None:
    sigs = list(next(iter(trig["detectors"].values()))["signals"])
    keys = [k for k in ORDER if k in trig["detectors"]]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    n = len(keys)
    h = 0.8 / n
    ypos = np.arange(len(sigs))
    # colour carries the family, hatching separates v3 from v5 within it, so
    # the two AXOL1TL variants are not rendered identically
    for i, key in enumerate(keys):
        e = trig["detectors"][key]
        vals = [e["signals"][s]["auc"] for s in sigs]
        st = STYLE[key]
        ax.barh(ypos + (i - n / 2 + 0.5) * h, vals, height=h * 0.92,
                color=st["color"], alpha=1.0 if key == "fly" else 0.82,
                edgecolor="white", linewidth=0.6, label=st["label"],
                hatch="///" if key.endswith("v5") or key.endswith("v5_retrained")
                else None)
    ax.axvline(0.5, color=GREY_D, lw=1.1, ls=":")
    ax.text(0.502, len(sigs) - 0.45, "coin flip", fontsize=7.5, color=GREY_D)
    ax.set_yticks(ypos); ax.set_yticklabels(sigs)
    ax.set_xlabel("AUC"); ax.set_xlim(0.2, 1.0)
    ax.invert_yaxis()
    ax.set_xlim(0.2, 1.0)
    h_, l_ = ax.get_legend_handles_labels()
    ax.legend(h_, l_, fontsize=8, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.14))
    ax.set_title("Ranking performance by signal (ADC2021)", fontsize=11)
    ax.text(0.5, -0.30, "(CMS) = real emulator run OFF-DISTRIBUTION (trained on Zero Bias, evaluated on Delphes/lepton-filtered ADC2021).  (retrained) = same topology refit here.", transform=ax.transAxes,
            ha="center", fontsize=7, color=GREY_D)
    save(fig, "02_auc_by_signal")


def fig_params(trig: dict) -> None:
    """Cost against benefit: fitted parameters versus best achieved SIC.

    The fly has zero fitted parameters, which has no place on a log axis, so it
    is drawn in its own panel to the left of an explicit axis break rather than
    parked at a fictitious 0.6.
    """
    sigs = list(next(iter(trig["detectors"].values()))["signals"])

    def best_sic(e):
        return max(e["signals"][s][f"eps_s_at_{p:g}"] / np.sqrt(p)
                   for s in sigs for p in EPS)

    fig, (axf, ax) = plt.subplots(
        1, 2, figsize=(8.2, 4.6), sharey=True,
        gridspec_kw={"width_ratios": [1, 7], "wspace": 0.06})

    for key in ORDER:
        e = trig["detectors"].get(key)
        if not e:
            continue
        st = STYLE[key]
        target = axf if key == "fly" else ax
        x = 0 if key == "fly" else e["trainable_parameters"]
        target.scatter(x, best_sic(e), s=170 if key == "fly" else 85,
                       color=st["color"], marker=st["marker"],
                       zorder=5, edgecolor="white", linewidth=1.2,
                       label=st["label"])

    for a in (axf, ax):
        a.axhline(1.0, color=WARN, lw=1.2, ls=":")
        a.set_yscale("log")
    axf.set_xlim(-0.55, 0.55); axf.set_xticks([0]); axf.set_xticklabels(["0"])
    axf.set_ylabel("best SIC achieved")
    axf.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(labelleft=False)
    ax.set_xscale("log"); ax.set_xlim(700, 12000)

    # axis-break marks
    for a, xpos in ((axf, 1.0), (ax, 0.0)):
        a.plot([xpos, xpos], [0.02, 0.06], transform=a.transAxes, color=RULE,
               lw=1.2, clip_on=False)
    fig.text(0.5, 0.015, "parameters fitted by optimisation",
             ha="center", fontsize=9)
    ax.text(760, 1.1, "below this line the cut makes the search worse",
            fontsize=7.5, color=WARN)
    # the fly lives in the left panel, so gather handles from both
    hf, lf = axf.get_legend_handles_labels()
    hr, lr = ax.get_legend_handles_labels()
    ax.legend(hf + hr, lf + lr, fontsize=8, loc="center right", ncol=1)
    fig.suptitle("What the fitting buys you", fontsize=11, y=0.97)
    fig.text(0.5, -0.02, "(CMS) = real emulator run OFF-DISTRIBUTION (trained on Zero Bias, evaluated on Delphes/lepton-filtered ADC2021).  (retrained) = same topology refit here.", ha="center", fontsize=7, color=GREY_D)
    save(fig, "03_parameters_vs_benefit")


def fig_rate(trig: dict) -> None:
    """The trigger idiom: signal efficiency against output rate, not eps_B."""
    sigs = list(next(iter(trig["detectors"].values()))["signals"])
    rate_khz = [40e6 * e / 1e3 for e in EPS]
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.5), sharey=True)
    for ax, sig in zip(axes, sigs):
        for key in ORDER:
            e = trig["detectors"].get(key)
            if not e:
                continue
            eff = np.array([e["signals"][sig][f"eps_s_at_{p:g}"] for p in EPS])
            ax.plot(rate_khz, np.where(eff <= 0, np.nan, eff), **STYLE[key])
        ax.axvline(1.0, color=WARN, lw=1.2, ls=":")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("rate [kHz]")
        ax.set_title(sig, fontsize=10)
    axes[0].set_ylabel(r"signal efficiency  $\varepsilon_S$")
    axes[0].text(1.15, axes[0].get_ylim()[0] * 2, "1 kHz\n(AXOL1TL working point)",
                 fontsize=7.5, color=WARN)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=7, fontsize=8,
               bbox_to_anchor=(0.5, -0.13))
    fig.suptitle("Signal efficiency against trigger rate  "
                 "(rate is within the lepton-filtered ADC2021 stream, "
                 "not minimum bias)", fontsize=10.5, y=1.02)
    fig.text(0.5, -0.20, "(CMS) = real emulator run OFF-DISTRIBUTION (trained on Zero Bias, evaluated on Delphes/lepton-filtered ADC2021).  (retrained) = same topology refit here.", ha="center", fontsize=7, color=GREY_D)
    save(fig, "04_efficiency_vs_rate")


def main() -> None:
    setup()
    trig = json.loads((RESULTS / "trigger_results.json").read_text())
    print("building ADC2021 figures...", flush=True)
    fig_sic(trig)
    fig_auc(trig)
    fig_params(trig)
    fig_rate(trig)
    print(f"\nfigures in {OUT}", flush=True)


if __name__ == "__main__":
    main()
