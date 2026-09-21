"""Assemble the published report page from the real analysis outputs.

Reads outputs/{results.json, roc.npz, reel.mp4, reel_manifest.json} and writes
outputs/report.html with the video embedded and the charts driven by the actual
measured arrays.

Run:  python scripts/build_report.py
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def thin_roc(fpr: np.ndarray, tpr: np.ndarray, n: int = 220) -> list[list[float]]:
    """Downsample an ROC to n points on a log-spaced fpr grid.

    Log spacing keeps resolution in the low-background-efficiency region, which
    is the part anyone actually reads.
    """
    grid = np.unique(np.concatenate([[0.0], np.logspace(-4, 0, n)]))
    grid = grid[grid <= 1.0]
    interp = np.interp(grid, fpr, tpr)
    return [[round(float(a), 5), round(float(b), 5)] for a, b in zip(grid, interp)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", type=Path, default=ROOT / "outputs" / "lhco")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "lhco" / "report.html")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    from sklearn.metrics import roc_curve

    res = json.loads((args.outputs / "results.json").read_text())
    d = np.load(args.outputs / "roc.npz")
    y = d["y_test"]

    series = []
    for key, tok in (("fly", "s1"), ("gaussian", "s2"), ("autoencoder", "s3")):
        skey = {"fly": "score_fly", "gaussian": "score_gaussian",
                "autoencoder": "score_ae"}[key]
        if skey not in d:
            continue
        fpr, tpr, _ = roc_curve(y, d[skey], drop_intermediate=False)
        series.append({
            "key": key,
            "name": res[key]["name"],
            "token": tok,
            "auc": res[key]["auc"],
            "params": res[key]["trainable_parameters"],
            "points": thin_roc(fpr, tpr),
        })

    sweep = res.get("kc_sweep", [])

    # score distributions for the fly, as histograms
    s_fly = d["score_fly"]
    lo, hi = float(np.quantile(s_fly, 0.001)), float(np.quantile(s_fly, 0.999))
    edges = np.linspace(lo, hi, 61)
    hist = {
        "edges": [round(float(e), 5) for e in edges],
        "qcd": np.histogram(s_fly[y == 0], bins=edges, density=True)[0].round(4).tolist(),
        "signal": np.histogram(s_fly[y == 1], bins=edges, density=True)[0].round(4).tolist(),
    }

    # working points: signal efficiency at fixed background efficiency, against
    # the random-guessing line. This is where all three detectors fail.
    wp = []
    for eb in (0.5, 0.2, 0.1, 0.05, 0.01, 0.001):
        row = {"eps_b": eb}
        for key, skey in (("fly", "score_fly"), ("gaussian", "score_gaussian"),
                          ("autoencoder", "score_ae")):
            if skey not in d:
                continue
            sc = d[skey]
            thr = np.quantile(sc[y == 0], 1 - eb)
            row[key] = round(float((sc[y == 1] >= thr).mean()), 5)
        wp.append(row)

    manifest = []
    mpath = args.outputs / "reel_manifest.json"
    if mpath.exists():
        manifest = json.loads(mpath.read_text())

    video_uri = ""
    vpath = args.outputs / "reel.mp4"
    if vpath.exists() and not args.no_video:
        raw = vpath.read_bytes()
        video_uri = "data:video/mp4;base64," + base64.b64encode(raw).decode()
        print(f"embedded video: {len(raw)/1e6:.1f} MB raw -> "
              f"{len(video_uri)/1e6:.1f} MB base64", flush=True)

    payload = {
        "series": series,
        "sweep": sweep,
        "hist": hist,
        "working_points": wp,
        "results": {k: v for k, v in res.items() if k != "kc_sweep"},
        "manifest": manifest,
        "diagnostics": res["fly"].get("diagnostics", {}),
    }

    tpl = (ROOT / "scripts" / "lhco" / "report_template.html").read_text()
    html = tpl.replace("__DATA__", json.dumps(payload, separators=(",", ":")))
    html = html.replace("__VIDEO__", video_uri)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html)
    print(f"wrote {args.out}  ({args.out.stat().st_size/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
