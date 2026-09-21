"""Does the measured wiring beat a random one?

Substitutes the real FlyWire PN -> Kenyon cell matrix for the sampled one and
compares, with everything else held fixed:

  1. measured        the FlyWire v783 wiring, one hemisphere
  2. degree-matched  same claw count and weights per KC, glomeruli reassigned
  3. sampled         the original FlyHash: 6 glomeruli per KC, uniform

Caron et al. (2013) report that PN -> KC wiring is random with respect to
glomerular identity. If that holds here, (1) and (2) should be indistinguishable.

Run:  python scripts/lhco/connectome_test.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from sklearn.metrics import roc_auc_score  # noqa: E402

from flyad.connectome import degree_matched_shuffle, load_wiring  # noqa: E402
from flyad.flyhash import FlyHash, WiredFlyHash  # noqa: E402
from flyad.mushroom_body import MushroomBody  # noqa: E402


def evaluate(hash_obj, Xtr, Xte, yte, *, batch=50_000) -> float:
    mb = MushroomBody(hash_obj.n_kc if hasattr(hash_obj, "n_kc") else hash_obj.n_kc)
    for i in range(0, len(Xtr), batch):
        mb.experience(hash_obj.tag(Xtr[i : i + batch]))
    scores = []
    for i in range(0, len(Xte), batch):
        scores.append(mb.novelty(hash_obj.tag(Xte[i : i + batch])))
    return float(roc_auc_score(yte, np.concatenate(scores)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glomeruli", type=Path,
                    default=ROOT / "outputs" / "lhco" / "glomeruli.npz")
    ap.add_argument("--dataset", default="malecns",
                    choices=["malecns", "flywire"],
                    help="malecns is the Sept 2026 Janelia+Google release")
    ap.add_argument("--side", default="right")
    ap.add_argument("--n-train", type=int, default=400_000)
    ap.add_argument("--n-test-bkg", type=int, default=100_000)
    ap.add_argument("--n-test-sig", type=int, default=100_000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "outputs" / "lhco" / "connectome_test.json")
    args = ap.parse_args()

    d = np.load(args.glomeruli)
    X, y = d["X"].astype(np.float64), d["y"]
    n_in = X.shape[1]

    print(f"loading measured wiring: {args.dataset} "
          f"({args.side} hemisphere)...", flush=True)
    w = load_wiring(args.dataset, side=args.side, n_glomeruli=n_in)
    print(json.dumps(w.summary(), indent=2), flush=True)
    print(f"glomeruli kept: {', '.join(w.glomeruli)}", flush=True)

    # Kenyon cells with no olfactory input cannot contribute; drop them so the
    # winner-take-all is not choosing among dead units.
    live = w.claws_per_kc() > 0
    W = w.weights[live]
    print(f"\nlive Kenyon cells: {live.sum():,} of {w.n_kc:,}", flush=True)

    rng = np.random.default_rng(0)
    bkg, sig = np.where(y == 0)[0], np.where(y == 1)[0]
    rng.shuffle(bkg); rng.shuffle(sig)
    tr = bkg[: args.n_train]
    te = np.concatenate([bkg[args.n_train : args.n_train + args.n_test_bkg],
                         sig[: args.n_test_sig]])
    Xtr, Xte, yte = X[tr], X[te], y[te]
    print(f"train {len(Xtr):,} background / test {len(Xte):,}\n", flush=True)

    # the measured claw count, not the textbook 6 FlyHash assumes
    measured_claws = float(w.claws_per_kc()[w.claws_per_kc() > 0].mean())
    print(f"measured claws per live KC: {measured_claws:.2f} "
          f"(FlyHash assumes {6})", flush=True)

    results: dict = {"wiring": w.summary(), "glomeruli": w.glomeruli,
                     "measured_claws_per_live_kc": measured_claws,
                     "n_live_kc": int(live.sum()), "runs": {}}

    t0 = time.time()
    auc = evaluate(WiredFlyHash(W), Xtr, Xte, yte)
    results["runs"]["measured"] = {"auc": auc, "seconds": time.time() - t0}
    print(f"  measured wiring          AUC = {auc:.4f}", flush=True)

    shuf = []
    for s in range(args.seeds):
        a = evaluate(WiredFlyHash(degree_matched_shuffle(W, seed=s)), Xtr, Xte, yte)
        shuf.append(a)
        print(f"  degree-matched (seed {s})  AUC = {a:.4f}", flush=True)
    results["runs"]["degree_matched"] = {"aucs": shuf,
                                         "mean": float(np.mean(shuf)),
                                         "std": float(np.std(shuf))}

    samp = []
    for s in range(args.seeds):
        a = evaluate(FlyHash(n_in, n_kc=int(live.sum()),
                             pn_per_kc=int(round(measured_claws)), seed=s),
                     Xtr, Xte, yte)
        samp.append(a)
        print(f"  sampled FlyHash (seed {s}) AUC = {a:.4f}", flush=True)
    results["runs"]["sampled"] = {"aucs": samp, "mean": float(np.mean(samp)),
                                  "std": float(np.std(samp))}

    m = results["runs"]["measured"]["auc"]
    dm = results["runs"]["degree_matched"]
    print(f"\nmeasured {m:.4f}  vs  degree-matched {dm['mean']:.4f} "
          f"+/- {dm['std']:.4f}", flush=True)
    z = (m - dm["mean"]) / dm["std"] if dm["std"] > 0 else float("nan")
    print(f"difference = {m - dm['mean']:+.4f}  ({z:+.1f} sd of the shuffles)",
          flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
