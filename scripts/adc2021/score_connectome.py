"""Score ADC2021 events through the measured male CNS wiring.

Produces the novelty scores the reel renders from, so the fly in the video is
driven by the real Janelia + Google connectome rather than a sampled matrix.

Run:  python scripts/adc2021/score_connectome.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from flyad import divisive_normalisation  # noqa: E402
from flyad.adc2021.lobe import N_TRIGGER_GLOMERULI, encode_trigger_batch  # noqa: E402
from flyad.connectome import load_wiring  # noqa: E402
from flyad.flyhash import WiredFlyHash  # noqa: E402
from flyad.mushroom_body import MushroomBody  # noqa: E402

DATA = ROOT / "data" / "adc2021"
SIGNALS = {
    "LQ -> b tau": "leptoquark.h5",
    "A -> 4 leptons": "Ato4l.h5",
    "h0 -> tau tau": "hToTauTau.h5",
    "h+ -> tau nu": "hChToTauNu.h5",
}


def encode(path: Path, limit=None, offset=0, chunk=200_000) -> np.ndarray:
    with h5py.File(path, "r") as f:
        ds = f["Particles"]
        n = (len(ds) - offset) if limit is None else min(limit, len(ds) - offset)
        out = np.zeros((n, N_TRIGGER_GLOMERULI), dtype=np.float32)
        for i in range(0, n, chunk):
            j = min(i + chunk, n)
            out[i:j] = encode_trigger_batch(ds[offset + i : offset + j].astype(np.float64))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="malecns", choices=["malecns", "flywire"])
    ap.add_argument("--side", default="right")
    ap.add_argument("--n-train", type=int, default=1_000_000)
    ap.add_argument("--n-test-bkg", type=int, default=200_000)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "outputs" / "adc2021" / "trigger_scores_connectome.npz")
    args = ap.parse_args()

    w = load_wiring(args.dataset, side=args.side, n_glomeruli=N_TRIGGER_GLOMERULI)
    live = w.claws_per_kc() > 0
    W = w.weights[live]
    print(f"{args.dataset}: {W.shape[0]:,} live Kenyon cells x "
          f"{W.shape[1]} glomeruli, {int(w.weights.sum()):,} synapses", flush=True)
    print(f"glomeruli: {', '.join(w.glomeruli)}", flush=True)

    hash_ = WiredFlyHash(W)
    mb = MushroomBody(hash_.n_kc)

    t0 = time.time()
    print(f"\nencoding {args.n_train:,} background events for training...", flush=True)
    Xtr = divisive_normalisation(encode(DATA / "background.h5", limit=args.n_train))
    for i in range(0, len(Xtr), 50_000):
        mb.experience(hash_.tag(Xtr[i : i + 50_000]))
    print(f"  trained in {time.time()-t0:.0f}s", flush=True)
    del Xtr

    def score(X):
        out = []
        for i in range(0, len(X), 50_000):
            out.append(mb.novelty(hash_.tag(X[i : i + 50_000])))
        return np.concatenate(out)

    Xb = divisive_normalisation(encode(DATA / "background.h5",
                                       limit=args.n_test_bkg, offset=args.n_train))
    parts_s, parts_y, parts_id = [score(Xb)], [np.zeros(len(Xb))], [np.zeros(len(Xb))]
    del Xb
    names = []
    for k, (name, fn) in enumerate(SIGNALS.items(), start=1):
        if not (DATA / fn).exists():
            continue
        Xs = divisive_normalisation(encode(DATA / fn))
        s = score(Xs)
        parts_s.append(s); parts_y.append(np.ones(len(s)))
        parts_id.append(np.full(len(s), k))
        names.append(name)
        print(f"  scored {name:16s} {len(s):>8,} events", flush=True)
        del Xs

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        score_fly=np.concatenate(parts_s),
        y_test=np.concatenate(parts_y).astype(np.int8),
        signal_id=np.concatenate(parts_id).astype(np.int8),
        signal_names=np.array(names, dtype=object),
        glomeruli=np.array(w.glomeruli, dtype=object),
        n_kc=W.shape[0], dataset=args.dataset,
    )
    from sklearn.metrics import roc_auc_score
    y = np.concatenate(parts_y); s = np.concatenate(parts_s)
    print(f"\noverall AUC (all signals pooled): {roc_auc_score(y, s):.4f}", flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
