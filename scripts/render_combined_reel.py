"""One video: the real Kenyon cells firing, beside the fly deciding.

Left panel  -- the measured right mushroom body, ~2,400 Kenyon cells at their
               FlyWire soma coordinates, with the 5% the circuit fires lit up.
               The pattern is computed once per event (one forward pass) and
               the view rotates while the fly walks.
Right panel -- NeuroMechFly on the spherical treadmill, stepping under the
               descending drive that same score produced.

Run:  MUJOCO_GL=egl python scripts/render_combined_reel.py --study lhco
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from render_tethered_reel import annotate  # noqa: E402

BG = (12, 13, 15)
ANOM = "#d8483f"
NORM = "#3d86d8"
GREY = "#5d686e"
BRAIN_W, PANEL_H = 560, 544
N_BRAIN_FRAMES = 40          # unique brain renders per clip, tiled to the clip


def build_brain():
    """Measured wiring + soma positions for the right mushroom body."""
    from flyad.connectome import FLYWIRE, load_annotations, load_wiring

    w = load_wiring("flywire", side="right", n_glomeruli=50)
    ann = load_annotations(FLYWIRE)
    kc = ann[(ann["cell_class"] == "Kenyon_Cell") & (ann["side"] == "right")]
    pos = {r: p for r, p in zip(
        kc["root_id"].to_numpy(),
        kc[["soma_x", "soma_y", "soma_z"]].astype(float).to_numpy()
        * np.array([4.0, 4.0, 40.0]) / 1000.0)}

    rows = [i for i, r in enumerate(w.kc_ids) if r in pos]
    live = w.claws_per_kc()[rows] > 0
    rows = np.array(rows)[live]
    xyz = np.stack([pos[w.kc_ids[i]] for i in rows])
    xyz -= xyz.mean(axis=0)
    return w.weights[rows], xyz


def brain_frame(xyz, active, anomalous, azim):
    fig = plt.figure(figsize=(BRAIN_W / 100, PANEL_H / 100), dpi=100,
                     facecolor="#0c0d0f")
    ax = fig.add_subplot(111, projection="3d", facecolor="#0c0d0f")
    ax.scatter(*xyz[~active].T, s=1.8, c=GREY, alpha=0.28,
               linewidths=0, depthshade=False)
    ax.scatter(*xyz[active].T, s=20.0, c=ANOM if anomalous else NORM,
               alpha=0.95, linewidths=0, depthshade=False)
    ax.view_init(elev=18, azim=azim)
    ax.set_axis_off()
    ax.set_box_aspect(np.ptp(xyz, axis=0))
    ax.set_position([-0.20, -0.22, 1.40, 1.40])
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    for setter, a, b in ((ax.set_xlim, lo[0], hi[0]), (ax.set_ylim, lo[1], hi[1]),
                         (ax.set_zlim, lo[2], hi[2])):
        setter(a, b)
    fig.canvas.draw()
    out = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", choices=["lhco", "adc2021"], default="lhco")
    ap.add_argument("--n-per-class", type=int, default=1)
    ap.add_argument("--duration", type=float, default=1.2)
    ap.add_argument("--n-train", type=int, default=300_000)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import imageio.v2 as imageio

    from flyad.embodied_tethered import TetheredRunner
    from flyad.flyhash import WiredFlyHash
    from flyad.mushroom_body import MushroomBody

    DEF = {"lhco": (ROOT / "outputs" / "lhco" / "roc.npz",
                    ROOT / "outputs" / "lhco" / "glomeruli.npz",
                    "LHC Olympics 2020 / dijet constituents"),
           "adc2021": (ROOT / "outputs" / "adc2021" / "trigger_scores.npz",
                       None, "ADC2021 / L1 trigger objects")}
    scores_f, glom_f, label = DEF[args.study]
    if args.out is None:
        args.out = ROOT / "outputs" / args.study / "reel_combined.mp4"

    print("loading measured wiring + soma positions...", flush=True)
    W, xyz = build_brain()
    print(f"  {len(W):,} Kenyon cells", flush=True)
    hash_ = WiredFlyHash(W)
    mb = MushroomBody(hash_.n_kc)

    # the brain panel needs its own trained Bloom filter on the same encoding
    if glom_f is None:
        raise SystemExit("combined reel currently needs LHCO glomeruli "
                         "(the 51-channel trigger encoding is too sparse for "
                         "the measured wiring -- see the README)")
    g = np.load(glom_f)
    Xg, yg = g["X"].astype(np.float64), g["y"]
    rng = np.random.default_rng(0)
    bkg = np.where(yg == 0)[0]; rng.shuffle(bkg)
    for i in range(0, args.n_train, 50_000):
        mb.experience(hash_.tag(Xg[bkg[i : i + 50_000]]))

    d = np.load(scores_f, allow_pickle=True)
    s, y, idx = d["score_fly"], d["y_test"], d["test_idx"]
    thr = float(np.quantile(s[y == 0], 0.90))
    called = s >= thr
    groups = {"true positive": np.where((y == 1) & called)[0],
              "true negative": np.where((y == 0) & ~called)[0],
              "false positive": np.where((y == 0) & called)[0],
              "false negative": np.where((y == 1) & ~called)[0]}
    picks = []
    for lab, ii in groups.items():
        if len(ii) == 0:
            continue
        ch = (ii[np.argsort(-s[ii])[: args.n_per_class]] if lab == "true positive"
              else rng.choice(ii, size=min(args.n_per_class, len(ii)), replace=False))
        picks += [(lab, int(k)) for k in ch]

    runner = TetheredRunner(duration_s=args.duration,
                            window_size=(704, PANEL_H))
    frames, manifest = [], []
    for n, (lab, k) in enumerate(picks, 1):
        nov, truth, event_id = float(s[k]), int(y[k]), int(idx[k])
        print(f"  [{n}/{len(picks)}] {lab:15s} novelty={nov:.4f}", flush=True)

        tmp = args.out.parent / f"_comb_{n:02d}.mp4"
        v = runner.run(novelty=nov, threshold=thr, event_index=event_id,
                       truth=truth, video_path=tmp)
        fly = imageio.mimread(tmp, memtest=False)
        tmp.unlink(missing_ok=True)

        # one forward pass gives the firing pattern for the whole clip
        tag = hash_.tag(Xg[event_id][None, :])[0]
        active = np.zeros(hash_.n_kc, dtype=bool); active[tag] = True
        brains = [brain_frame(xyz, active, v.called_anomaly,
                              -60 + 360 * i / N_BRAIN_FRAMES)
                  for i in range(N_BRAIN_FRAMES)]

        bmap = np.linspace(0, N_BRAIN_FRAMES - 1, len(fly)).astype(int)
        ymap = np.linspace(0, len(v.yaw_corrected_trace) - 1, len(fly)).astype(int)
        for f, bi, yi in zip(fly, bmap, ymap):
            combo = np.concatenate([brains[bi], f], axis=1)
            frames.append(annotate(
                combo, event_id=event_id, novelty=nov, threshold=thr,
                yaw_deg=float(np.degrees(v.yaw_corrected_trace[yi])),
                final_anomaly=v.called_anomaly, truth=truth, drive=v.drive,
                dataset_label=label, leg_asym=v.leg_asymmetry))

        manifest.append({"clip": n, "class": lab, "event_id": event_id,
                         "novelty": nov, "truth": truth,
                         "called_anomaly": v.called_anomaly,
                         "kc_firing": int(active.sum()),
                         "kc_total": int(hash_.n_kc),
                         "leg_asymmetry": v.leg_asymmetry})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(args.out, frames, fps=30, quality=7, macro_block_size=1,
                     codec="libx264", output_params=["-pix_fmt", "yuv420p"])
    print(f"\nwrote {args.out}  ({len(frames)} frames, "
          f"{args.out.stat().st_size/1e6:.1f} MB)", flush=True)
    (args.out.parent / "reel_combined_manifest.json").write_text(
        json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
