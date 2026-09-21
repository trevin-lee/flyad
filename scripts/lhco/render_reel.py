"""Render the highlight reel: the fly physically calling a selection of events.

Picks true positives, true negatives and the mistakes, runs each through
MuJoCo, annotates the frames, and writes one concatenated reel.

Run:  MUJOCO_GL=egl python scripts/render_reel.py [--n-per-class 2]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

FONT_DIR = Path("/usr/share/fonts/Adwaita")
RED = (216, 38, 38)
BLUE = (38, 115, 216)
WHITE = (245, 245, 245)
DIM = (165, 165, 165)
GREEN = (60, 180, 90)


def _font(size: int, bold: bool = False):
    name = "AdwaitaMono-Bold.ttf" if bold else "AdwaitaMono-Regular.ttf"
    try:
        return ImageFont.truetype(str(FONT_DIR / name), size)
    except OSError:
        return ImageFont.load_default()


def annotate(frame: np.ndarray, *, event_id: int, novelty: float, threshold: float,
             called_anomaly: bool, truth: int) -> np.ndarray:
    """Draw the HUD onto one rendered frame."""
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img, "RGBA")
    W, _ = img.size

    f_small, f_mid, f_big = _font(15), _font(17, True), _font(27, True)

    d.rectangle([0, 0, W, 74], fill=(15, 15, 18, 205))
    d.text((14, 9), f"EVENT {event_id:,}", font=f_mid, fill=WHITE)
    d.text((14, 32), "LHC Olympics 2020 R&D", font=f_small, fill=DIM)
    d.text((14, 51), f"MBON novelty {novelty:.4f}   threshold {threshold:.4f}",
           font=f_small, fill=DIM)

    verdict = "ANOMALY" if called_anomaly else "NOT ANOMALY"
    colour = RED if called_anomaly else BLUE
    tw = d.textlength(verdict, font=f_big)
    d.text((W - tw - 14, 14), verdict, font=f_big, fill=colour)

    truth_txt = "Z' -> XY" if truth == 1 else "QCD dijet"
    correct = (truth == 1) == called_anomaly
    mark = "correct" if correct else "WRONG"
    truth_line = f"truth: {truth_txt}  [{mark}]"
    d.text((W - d.textlength(truth_line, font=f_small) - 14, 50), truth_line,
           font=f_small, fill=GREEN if correct else RED)

    # legend, so the buttons read without narration
    H = img.size[1]
    d.rectangle([0, H - 30, W, H], fill=(15, 15, 18, 205))
    d.ellipse([14, H - 22, 28, H - 8], fill=RED)
    d.text((34, H - 23), "ANOMALY", font=f_small, fill=DIM)
    d.ellipse([148, H - 22, 162, H - 8], fill=BLUE)
    d.text((168, H - 23), "NOT ANOMALY", font=f_small, fill=DIM)
    return np.asarray(img)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roc", type=Path, default=ROOT / "outputs" / "lhco" / "roc.npz")
    ap.add_argument("--n-per-class", type=int, default=2)
    ap.add_argument("--fpr-target", type=float, default=0.10,
                    help="background efficiency that sets the fly's threshold")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "lhco" / "reel.mp4")
    args = ap.parse_args()

    import imageio.v2 as imageio

    from flyad.embodied import ButtonPressRunner

    d = np.load(args.roc)
    scores, y, test_idx = d["score_fly"], d["y_test"], d["test_idx"]

    # threshold at the requested background efficiency
    threshold = float(np.quantile(scores[y == 0], 1 - args.fpr_target))
    called = scores >= threshold
    print(f"threshold {threshold:.4f} at eps_B={args.fpr_target:.0%}", flush=True)

    rng = np.random.default_rng(0)
    groups = {
        "true positive": np.where((y == 1) & called)[0],
        "true negative": np.where((y == 0) & ~called)[0],
        "false positive": np.where((y == 0) & called)[0],
        "false negative": np.where((y == 1) & ~called)[0],
    }

    picks = []
    for label, idx in groups.items():
        if len(idx) == 0:
            print(f"  (no {label} events)", flush=True)
            continue
        # for true positives take the most confident; otherwise sample
        chosen = (idx[np.argsort(-scores[idx])[: args.n_per_class]]
                  if label == "true positive"
                  else rng.choice(idx, size=min(args.n_per_class, len(idx)), replace=False))
        for i in chosen:
            picks.append((label, int(i)))

    print(f"rendering {len(picks)} events", flush=True)

    runner = ButtonPressRunner(max_time_s=3.0, play_speed=0.5)
    all_frames: list[np.ndarray] = []
    manifest = []

    for n, (label, i) in enumerate(picks, 1):
        event_id = int(test_idx[i])
        nov, truth = float(scores[i]), int(y[i])
        print(f"  [{n}/{len(picks)}] {label:15s} event {event_id:,} "
              f"novelty={nov:.4f}", flush=True)

        tmp = ROOT / "outputs" / "lhco" / f"_clip_{n:02d}.mp4"
        v = runner.run(novelty=nov, threshold=threshold, event_index=event_id,
                       truth=truth, video_path=tmp)

        frames = [
            annotate(f, event_id=event_id, novelty=nov, threshold=threshold,
                     called_anomaly=v.called_anomaly, truth=truth)
            for f in imageio.mimread(tmp, memtest=False)
        ]
        all_frames.extend(frames)
        tmp.unlink(missing_ok=True)

        manifest.append({
            "clip": n, "class": label, "event_id": event_id,
            "novelty": nov, "threshold": threshold,
            "called_anomaly": v.called_anomaly, "truth": truth,
            "reached": v.reached, "walk_time_s": v.walk_time_s,
            "n_frames": len(frames),
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(args.out, all_frames, fps=30, quality=7,
                     macro_block_size=1, codec="libx264",
                     output_params=["-pix_fmt", "yuv420p"])
    size_mb = args.out.stat().st_size / 1e6
    print(f"\nwrote {args.out}  ({len(all_frames)} frames, {size_mb:.1f} MB)", flush=True)

    mpath = ROOT / "outputs" / "lhco" / "reel_manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {mpath}", flush=True)


if __name__ == "__main__":
    main()
