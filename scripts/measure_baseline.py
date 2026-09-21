"""Measure the treadmill's intrinsic turning bias, for the current mount.

Real tethered experiments measure each animal's baseline turning under a
symmetric drive and subtract it. So do we -- and the constant has to be
re-measured whenever the rig changes, which centring the ball mount did.

Run:  MUJOCO_GL=egl python scripts/measure_baseline.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--duration", type=float, default=1.2)
    ap.add_argument("--stock-mount", action="store_true",
                    help="use flygym's off-axis ball_pos instead of the centred one")
    args = ap.parse_args()

    from flyad.embodied_tethered import TetheredRunner

    yaws, asyms = [], []
    print(f"symmetric drive (novelty == threshold), "
          f"{'stock' if args.stock_mount else 'centred'} mount, "
          f"{args.duration}s trials\n", flush=True)
    print(f"{'seed':>5s}{'ball yaw':>12s}{'leg L-R':>12s}")
    for seed in range(args.seeds):
        r = TetheredRunner(duration_s=args.duration, seed=seed,
                           centred_ball=not args.stock_mount)
        v = r.run(novelty=0.5, threshold=0.5)      # zero commanded turn
        deg = np.degrees(v.yaw_rad)
        yaws.append(deg); asyms.append(v.leg_asymmetry)
        print(f"{seed:>5d}{deg:>+12.1f}{v.leg_asymmetry:>+12.4f}", flush=True)

    y = np.array(yaws)
    print(f"\nball yaw   mean {y.mean():+.1f} deg  sd {y.std():.1f}  "
          f"({int((y > 0).sum())} left / {int((y < 0).sum())} right)")
    print(f"leg L-R    mean {np.mean(asyms):+.4f}  sd {np.std(asyms):.4f}")
    print(f"\nBASELINE_YAW_RATE = np.radians({y.mean():.1f}) / {args.duration}"
          f"   # rad/s")


if __name__ == "__main__":
    main()
