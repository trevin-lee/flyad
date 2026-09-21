"""Light up the real Kenyon cells while the circuit classifies events.

Everything on screen is measured except the stimulus:

  * soma positions  -- FlyWire v783, 2,597 right-hemisphere Kenyon cells
  * connectivity    -- the measured PN -> KC matrix from the same release
  * which cells fire -- computed by the circuit, top 5% after winner-take-all
  * MBON weight     -- the Bloom filter state, depressed by background

Run:  python scripts/render_brain_activity.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from flyad.connectome import load_annotations, load_wiring, FLYWIRE  # noqa: E402
from flyad.flyhash import WiredFlyHash  # noqa: E402
from flyad.mushroom_body import MushroomBody  # noqa: E402
from flyad.plotting import FLY, GREY_D, INK  # noqa: E402

ANOM = "#d8483f"
NORM = "#3d86d8"
BG = "#0c0d0f"
# FlyWire voxels are 4 x 4 x 40 nm, so z needs scaling to be physically square
VOXEL = np.array([4.0, 4.0, 40.0])


def kc_positions(side: str = "right") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ann = load_annotations(FLYWIRE)
    kc = ann[(ann["cell_class"] == "Kenyon_Cell") & (ann["side"] == side)]
    xyz = kc[["soma_x", "soma_y", "soma_z"]].apply(
        lambda c: c.astype(float)).to_numpy() * VOXEL
    xyz -= xyz.mean(axis=0)
    return kc["root_id"].to_numpy(), xyz / 1000.0, kc["cell_type"].to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glomeruli", type=Path,
                    default=ROOT / "outputs" / "lhco" / "glomeruli.npz")
    ap.add_argument("--n-train", type=int, default=300_000)
    ap.add_argument("--n-events", type=int, default=8, help="events to show")
    ap.add_argument("--frames-per-event", type=int, default=26)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "outputs" / "lhco" / "brain_activity.mp4")
    args = ap.parse_args()

    import imageio.v2 as imageio

    print("loading measured wiring and soma positions...", flush=True)
    w = load_wiring("flywire", side="right", n_glomeruli=50)
    ids, xyz, types = kc_positions("right")
    # keep the cells that both have olfactory input and a known position
    order = {r: i for i, r in enumerate(ids)}
    keep = np.array([order[r] for r in w.kc_ids if r in order])
    live = w.claws_per_kc()[[list(w.kc_ids).index(ids[k]) for k in keep]] > 0
    idx_w = np.array([list(w.kc_ids).index(ids[k]) for k in keep])[live]
    xyz = xyz[keep][live]
    W = w.weights[idx_w]
    print(f"  {len(W):,} Kenyon cells with wiring and position", flush=True)

    hash_ = WiredFlyHash(W)
    mb = MushroomBody(hash_.n_kc)

    d = np.load(args.glomeruli)
    X, y = d["X"].astype(np.float64), d["y"]
    rng = np.random.default_rng(0)
    bkg, sig = np.where(y == 0)[0], np.where(y == 1)[0]
    rng.shuffle(bkg); rng.shuffle(sig)
    Xtr = X[bkg[: args.n_train]]
    print("training the mushroom body on background...", flush=True)
    for i in range(0, len(Xtr), 50_000):
        mb.experience(hash_.tag(Xtr[i : i + 50_000]))

    pool = np.concatenate([bkg[args.n_train : args.n_train + 40_000], sig[:40_000]])
    scores = mb.novelty(hash_.tag(X[pool]))
    thr = float(np.quantile(scores[y[pool] == 0], 0.90))
    # alternate signal and background so the display changes character
    picks, s_i, b_i = [], 0, 0
    sp = pool[y[pool] == 1][np.argsort(-scores[y[pool] == 1])]
    bp = pool[y[pool] == 0][np.argsort(scores[y[pool] == 0])]
    for k in range(args.n_events):
        if k % 2 == 0:
            picks.append(sp[s_i]); s_i += 1
        else:
            picks.append(bp[b_i]); b_i += 1

    print(f"rendering {len(picks)} events...", flush=True)
    frames = []
    for n, ev in enumerate(picks, 1):
        pn = X[ev][None, :]
        tag = hash_.tag(pn)[0]
        nov = float(mb.novelty(hash_.tag(pn))[0])
        anom = nov >= thr
        active = np.zeros(hash_.n_kc, dtype=bool)
        active[tag] = True
        print(f"  [{n}/{len(picks)}] {'signal' if y[ev] else 'background':10s} "
              f"novelty {nov:.4f}  {int(active.sum())} cells firing", flush=True)

        for f in range(args.frames_per_event):
            az = -60 + 360 * (n - 1 + f / args.frames_per_event) / len(picks)
            fig = plt.figure(figsize=(7.04, 5.44), facecolor=BG)
            ax = fig.add_subplot(111, projection="3d", facecolor=BG)
            ax.scatter(*xyz[~active].T, s=2.0, c=GREY_D, alpha=0.30,
                       linewidths=0, depthshade=False)
            ax.scatter(*xyz[active].T, s=22.0, c=ANOM if anom else NORM,
                       alpha=0.95, linewidths=0, depthshade=False)
            ax.view_init(elev=18, azim=az)
            ax.set_axis_off()
            ax.set_box_aspect(np.ptp(xyz, axis=0))
            # matplotlib's 3D axes reserve a wide margin for tick labels we are
            # not drawing, so overscan the axes to let the soma cloud fill the
            # frame instead of sitting in a corner of it
            ax.set_position([-0.18, -0.20, 1.36, 1.34])
            lo, hi = xyz.min(axis=0), xyz.max(axis=0)
            pad = 0.02 * (hi - lo)
            ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
            ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
            ax.set_zlim(lo[2] - pad[2], hi[2] + pad[2])
            fig.text(0.03, 0.95, f"EVENT {int(ev):,}", color="#f3f3f0",
                     fontsize=11, fontweight="600", family="monospace")
            fig.text(0.03, 0.915, "FlyWire v783 - right mushroom body, "
                     f"{hash_.n_kc:,} Kenyon cells", color="#8b969c",
                     fontsize=7.5, family="monospace")
            fig.text(0.03, 0.885,
                     f"{int(active.sum())} firing  ({100*active.mean():.0f}%)",
                     color="#8b969c", fontsize=7.5, family="monospace")
            fig.text(0.97, 0.95, "ANOMALY" if anom else "NOT ANOMALY",
                     color=ANOM if anom else NORM, fontsize=15,
                     fontweight="600", ha="right", family="monospace")
            fig.text(0.97, 0.915,
                     f"MBON novelty {nov:.4f}   threshold {thr:.4f}",
                     color="#8b969c", fontsize=7.5, ha="right", family="monospace")
            fig.text(0.97, 0.885,
                     f"truth: {'signal' if y[ev] else 'background'}",
                     color="#48b868" if bool(y[ev]) == anom else ANOM,
                     fontsize=7.5, ha="right", family="monospace")
            fig.canvas.draw()
            frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
            plt.close(fig)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(args.out, frames, fps=26, quality=7, macro_block_size=1,
                     codec="libx264", output_params=["-pix_fmt", "yuv420p"])
    print(f"\nwrote {args.out}  ({len(frames)} frames, "
          f"{args.out.stat().st_size/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
