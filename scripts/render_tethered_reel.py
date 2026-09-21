"""Highlight reel from the tethered preparation.

The fly stays on its spherical treadmill. The MBON's novelty score sets a
descending drive, the CPG steps the legs, the legs turn the ball, and the
verdict is read from the ball's measured yaw. The HUD shows that yaw
accumulating in real time, because it is a measurement rather than a replay of
the command.

Run:  MUJOCO_GL=egl python scripts/render_tethered_reel.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FONT_DIR = Path("/usr/share/fonts/Adwaita")
RED = (216, 54, 46)
BLUE = (54, 118, 216)
WHITE = (243, 243, 240)
DIM = (150, 156, 160)
GREEN = (72, 184, 104)
DECIDE_DEG = 25.0   # |yaw| past which the turn is called


def _font(size: int, bold: bool = False):
    name = "AdwaitaMono-Bold.ttf" if bold else "AdwaitaMono-Regular.ttf"
    try:
        return ImageFont.truetype(str(FONT_DIR / name), size)
    except OSError:
        return ImageFont.load_default()


def annotate(frame, *, event_id, novelty, threshold, yaw_deg, final_anomaly,
             truth, drive, dataset_label="", leg_asym=None):
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img, "RGBA")
    W, H = img.size
    f_s, f_m, f_b = _font(14), _font(16, True), _font(25, True)

    # ---- header ----
    d.rectangle([0, 0, W, 76], fill=(12, 13, 15, 215))
    d.text((14, 8), f"EVENT {event_id:,}", font=f_m, fill=WHITE)
    d.text((14, 30), dataset_label, font=f_s, fill=DIM)
    d.text((14, 50), f"MBON novelty {novelty:.4f}   threshold {threshold:.4f}",
           font=f_s, fill=DIM)

    # The legs' call -- see flyad/embodied_tethered.py for why the ball is
    # shown but does not vote.
    verdict, colour = ("ANOMALY", RED) if final_anomaly else ("NOT ANOMALY", BLUE)
    decided = True
    tw = d.textlength(verdict, font=f_b)
    d.text((W - tw - 14, 12), verdict, font=f_b, fill=colour)

    if decided:
        ok = (truth == 1) == final_anomaly
        line = (f"truth: {'signal' if truth == 1 else 'background'}  "
                f"[{'correct' if ok else 'WRONG'}]")
        d.text((W - d.textlength(line, font=f_s) - 14, 48), line,
               font=f_s, fill=GREEN if ok else RED)

    # ---- descending drive, the circuit's actual output ----
    d.text((14, H - 78), f"descending drive  L {drive[0]:.2f}   R {drive[1]:.2f}",
           font=f_s, fill=DIM)

    # ---- the decision: leg stepping asymmetry ----
    # Displayed as R-L rather than L-R so the sign agrees with the picture:
    # positive means the right legs sweep wider, the fly turns toward its own
    # left, and on a face-on camera that is the viewer's right -- which is where
    # the red ANOMALY panel is. Bar fills right in red, number is positive, and
    # the scene, the meter and the verdict all point the same way.
    bx0, bx1, by = 14, W - 14, H - 40
    mid = (bx0 + bx1) / 2
    d.rectangle([0, H - 58, W, H], fill=(12, 13, 15, 222))
    nl = "NOT ANOMALY"
    d.text((bx0, H - 56), nl, font=_font(11), fill=BLUE)
    d.text((bx1 - d.textlength("ANOMALY", font=_font(11)), H - 56), "ANOMALY",
           font=_font(11), fill=RED)
    d.line([bx0, by, bx1, by], fill=(70, 74, 78), width=2)
    d.line([mid, by - 9, mid, by + 9], fill=(110, 115, 120), width=2)

    turn = None if leg_asym is None else -leg_asym      # R - L
    if turn is not None:
        frac = float(np.clip(turn / 0.25, -1.0, 1.0))
        d.line([mid, by, mid + frac * (bx1 - bx0) / 2, by],
               fill=RED if frac > 0 else BLUE, width=9)

    lead = ("leg stepping asymmetry  R-L "
            f"{turn:+.4f}") if turn is not None else "leg stepping asymmetry"
    d.text((bx0, H - 20), lead, font=f_s, fill=WHITE)
    ball = f"ball yaw {yaw_deg:+.0f} deg"
    d.text((bx1 - d.textlength(ball, font=_font(11)), H - 19), ball,
           font=_font(11), fill=DIM)
    return np.asarray(img)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", choices=["lhco", "adc2021"], default="adc2021")
    ap.add_argument("--results", type=Path, default=None,
                    help="score file; defaults to the study's own")
    ap.add_argument("--n-per-class", type=int, default=1)
    ap.add_argument("--duration", type=float, default=1.2)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    DEFAULTS = {
        "lhco": (ROOT / "outputs" / "lhco" / "roc.npz",
                 "LHC Olympics 2020 / dijet constituents"),
        "adc2021": (ROOT / "outputs" / "adc2021" / "trigger_scores.npz",
                    "ADC2021 / L1 trigger objects"),
    }
    default_scores, dataset_label = DEFAULTS[args.study]
    if args.results is None:
        args.results = default_scores
    if args.out is None:
        args.out = ROOT / "outputs" / args.study / "reel_tethered.mp4"

    import imageio.v2 as imageio

    from flyad.embodied_tethered import TetheredRunner, descending_drive

    d = np.load(args.results)
    scores, y = d["score_fly"], d["y_test"]
    threshold = float(np.quantile(scores[y == 0], 1 - 0.10))
    called = scores >= threshold
    print(f"threshold {threshold:.4f} at eps_B=10%", flush=True)

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
            continue
        chosen = (idx[np.argsort(-scores[idx])[: args.n_per_class]]
                  if label == "true positive"
                  else rng.choice(idx, size=min(args.n_per_class, len(idx)), replace=False))
        picks += [(label, int(i)) for i in chosen]

    runner = TetheredRunner(duration_s=args.duration)
    frames, manifest = [], []
    for n, (label, i) in enumerate(picks, 1):
        nov, truth = float(scores[i]), int(y[i])
        print(f"  [{n}/{len(picks)}] {label:15s} novelty={nov:.4f}", flush=True)
        tmp = ROOT / "outputs" / "adc2021" / f"_teth_{n:02d}.mp4"
        v = runner.run(novelty=nov, threshold=threshold, event_index=i,
                       truth=truth, video_path=tmp)

        raw = imageio.mimread(tmp, memtest=False)
        # map each rendered frame onto its point in the measured yaw trace
        idx_map = np.linspace(0, len(v.yaw_trace) - 1, len(raw)).astype(int)
        for f, k in zip(raw, idx_map):
            frames.append(annotate(
                f, event_id=v.event_index, novelty=nov, threshold=threshold,
                yaw_deg=float(np.degrees(v.yaw_corrected_trace[k])),
                final_anomaly=v.called_anomaly, truth=truth, drive=v.drive,
                dataset_label=dataset_label, leg_asym=v.leg_asymmetry))
        tmp.unlink(missing_ok=True)
        manifest.append({
            "clip": n, "class": label, "novelty": nov, "truth": truth,
            "called_anomaly": v.called_anomaly,
            "yaw_deg": float(np.degrees(v.yaw_rad)),
            "yaw_corrected_deg": float(np.degrees(v.yaw_corrected_rad)),
            "leg_asymmetry": v.leg_asymmetry,
            "drive": list(v.drive), "n_frames": len(raw),
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(args.out, frames, fps=30, quality=7, macro_block_size=1,
                     codec="libx264", output_params=["-pix_fmt", "yuv420p"])
    print(f"\nwrote {args.out}  ({len(frames)} frames, "
          f"{args.out.stat().st_size/1e6:.1f} MB)", flush=True)
    # write beside the reel, not to a hardcoded study directory
    (args.out.parent / "reel_tethered_manifest.json").write_text(
        json.dumps(manifest, indent=2))
    print(f"wrote {args.out.parent / 'reel_tethered_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
