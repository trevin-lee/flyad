"""Cluster every LHCO event and encode it as 50 glomeruli of projection-neuron
activity. Writes outputs/glomeruli.npz.

Run:  python scripts/encode.py [--limit N] [--workers K]
"""

from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_H5 = ROOT / "data" / "lhco" / "events_anomalydetection_v2.h5"
OUT = ROOT / "outputs" / "lhco" / "glomeruli.npz"


def _work(job):
    path, start, stop = job
    from flyad.lhco.data import encode_chunk, read_chunk

    hadrons, labels = read_chunk(path, start, stop)
    return start, encode_chunk(hadrons), labels.astype(np.int8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", type=Path, default=DEFAULT_H5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--chunk-size", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    from flyad.lhco.data import n_events

    total = n_events(args.h5)
    if args.limit:
        total = min(total, args.limit)
    print(f"encoding {total:,} events from {args.h5.name}", flush=True)

    jobs = [
        (str(args.h5), s, min(s + args.chunk_size, total))
        for s in range(0, total, args.chunk_size)
    ]

    X = np.zeros((total, 50), dtype=np.float32)
    y = np.zeros(total, dtype=np.int8)

    t0 = time.time()
    done = 0
    with Pool(args.workers) as pool:
        for start, px, ly in pool.imap_unordered(_work, jobs):
            X[start : start + len(px)] = px
            y[start : start + len(ly)] = ly
            done += len(px)
            el = time.time() - t0
            pct = 100 * done / total
            print(
                f"\r  {done:,}/{total:,} ({pct:5.1f}%)  {done/el:7.0f} ev/s  "
                f"eta {(total-done)/(done/el)/60:5.1f} min",
                end="", flush=True,
            )
    print(f"\ndone in {(time.time()-t0)/60:.1f} min", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, X=X, y=y)
    print(f"wrote {args.out}  X={X.shape}  signal={int((y==1).sum()):,}", flush=True)


if __name__ == "__main__":
    main()
