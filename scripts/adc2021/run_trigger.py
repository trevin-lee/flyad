"""Run the fly against the real AXOL1TL on the CMS trigger representation.

Trains one-class on the ADC2021 Standard Model background, then evaluates every
detector against each benchmark signal separately.

Detectors:
  * the fly olfactory circuit          -- 0 parameters, trained here
  * Gaussian / autoencoder / VAE       -- baselines, trained here
  * AXOL1TL v3 and v5                  -- the real CMS emulator, pretrained

Run:  python scripts/run_trigger.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from sklearn.metrics import roc_auc_score  # noqa: E402

from flyad import FlyAnomalyDetector  # noqa: E402
from flyad import divisive_normalisation  # noqa: E402
from flyad.adc2021.axol1tl_retrain import (  # noqa: E402
    Axol1tlV3Retrained,
    Axol1tlV5Retrained,
)
from flyad.baselines import AutoencoderBaseline, GaussianBaseline  # noqa: E402
from flyad.adc2021.lobe import N_TRIGGER_GLOMERULI, encode_trigger_batch  # noqa: E402

DATA = ROOT / "data" / "adc2021"
SIGNALS = {
    "LQ -> b tau": "leptoquark.h5",
    "A -> 4 leptons": "Ato4l.h5",
    "h0 -> tau tau": "hToTauTau.h5",
    "h+ -> tau nu": "hChToTauNu.h5",
}
# the L1 budget is ~100 kHz out of 40 MHz; an anomaly seed gets a few kHz
WORKING_POINTS = (1e-2, 1e-3, 1e-4, 1e-5)


def load(path: Path, *, limit: int | None = None, offset: int = 0,
         need_hw: bool = False, chunk: int = 200_000
         ) -> tuple[np.ndarray, np.ndarray | None]:
    """Stream a contiguous block of an ADC2021 file into (glomeruli, hardware).

    The hardware conversion is only computed when a pretrained emulator is
    going to see the events -- it is the memory-hungry half.
    """
    from flyad.adc2021.axol1tl import to_hardware

    with h5py.File(path, "r") as f:
        ds = f["Particles"]
        avail = len(ds) - offset
        n = avail if limit is None else min(limit, avail)
        glom = np.zeros((n, N_TRIGGER_GLOMERULI), dtype=np.float32)
        hw = np.zeros((n, 57), dtype=np.float64) if need_hw else None
        for i in range(0, n, chunk):
            j = min(i + chunk, n)
            raw = ds[offset + i : offset + j].astype(np.float64)
            glom[i:j] = encode_trigger_batch(raw)
            if need_hw:
                hw[i:j] = to_hardware(raw)
    return glom, hw


def eps_s_at(bkg: np.ndarray, sig: np.ndarray, eps_b: float) -> float:
    return float((sig >= np.quantile(bkg, 1 - eps_b)).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=3_000_000)
    ap.add_argument("--n-test-bkg", type=int, default=1_000_000)
    ap.add_argument("--n-kc", type=int, default=2000)
    ap.add_argument("--ae-epochs", type=int, default=12)
    ap.add_argument("--axo-versions", default="v3,v5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "adc2021" / "trigger_results.json")
    ap.add_argument("--save-scores", type=Path, default=None,
                    help="also write per-event fly scores, for the reel renderer")
    args = ap.parse_args()

    # The ADC2021 background file is already thoroughly shuffled -- per-block
    # means of MET, jet, muon and e/gamma multiplicity are flat across it -- so
    # contiguous train/test blocks are unbiased, and avoid the very slow
    # scattered HDF5 reads a random permutation would need.
    t0 = time.time()
    print(f"loading {args.n_train:,} background events for training...", flush=True)
    tr_g, tr_hw = load(DATA / "background.h5", limit=args.n_train, need_hw=True)
    Xtr = divisive_normalisation(tr_g)
    del tr_g
    print(f"  encoded in {time.time()-t0:.0f}s", flush=True)

    t0 = time.time()
    print(f"loading {args.n_test_bkg:,} held-out background events...", flush=True)
    te_g, bkg_hw = load(DATA / "background.h5", limit=args.n_test_bkg,
                        offset=args.n_train, need_hw=True)
    Xte_b = divisive_normalisation(te_g)
    del te_g
    print(f"  encoded in {time.time()-t0:.0f}s", flush=True)
    print(f"  train {len(Xtr):,} / test-background {len(Xte_b):,}", flush=True)

    sigs_g, sigs_hw = {}, {}
    for name, fn in SIGNALS.items():
        p = DATA / fn
        if not p.exists():
            print(f"  (missing {fn})", flush=True)
            continue
        g, hw = load(p, need_hw=True)
        sigs_g[name] = divisive_normalisation(g)
        sigs_hw[name] = hw
        print(f"  signal {name:16s} {len(g):>8,} events", flush=True)

    detectors: dict = {}

    # ---- the fly -------------------------------------------------------
    print("\n[fly] experiencing background...", flush=True)
    t0 = time.time()
    fly = FlyAnomalyDetector(n_inputs=N_TRIGGER_GLOMERULI, n_kc=args.n_kc, seed=args.seed)
    for i in range(0, len(Xtr), 100_000):
        fly.experience(Xtr[i : i + 100_000])
    detectors["fly"] = {
        "name": "Fly olfactory circuit", "params": 0, "train_s": time.time() - t0,
        "score_g": fly.score, "diagnostics": fly.diagnostics(), "trained_here": True,
        "state_values": fly.n_state_values,
    }
    print(f"  {time.time()-t0:.1f}s, 0 parameters", flush=True)

    print("[gaussian] fitting...", flush=True)
    t0 = time.time(); g = GaussianBaseline().fit(Xtr)
    detectors["gaussian"] = {"name": g.name, "params": g.n_trainable_parameters,
                             "train_s": time.time() - t0, "score_g": g.score,
                             "trained_here": True}

    print("[autoencoder] training...", flush=True)
    t0 = time.time()
    ae = AutoencoderBaseline(N_TRIGGER_GLOMERULI, hidden=(32, 16), latent=3,
                             epochs=args.ae_epochs, seed=args.seed).fit(Xtr, verbose=True)
    detectors["autoencoder"] = {"name": ae.name, "params": ae.n_trainable_parameters,
                                "train_s": time.time() - t0, "score_g": ae.score,
                                "trained_here": True}

    # The AXOL1TL topologies, exactly as deployed, but fitted to THIS background
    # so the comparison is not confounded by the training distribution.
    for key, cls in (("axo_v3_retrained", Axol1tlV3Retrained),
                     ("axo_v5_retrained", Axol1tlV5Retrained)):
        print(f"[{cls.name}] training...", flush=True)
        t0 = time.time()
        m = cls(epochs=args.ae_epochs, seed=args.seed).fit(tr_hw, verbose=True)
        entry = {"name": cls.name, "params": m.n_trainable_parameters,
                 "train_s": time.time() - t0, "score_hw": m.score_hardware,
                 "trained_here": True}
        if hasattr(m, "n_inference_parameters"):
            entry["inference_parameters"] = m.n_inference_parameters
        detectors[key] = entry

    # ---- the real thing ------------------------------------------------
    from flyad.adc2021.axol1tl import Axol1tl

    axos = []
    for v in [x for x in args.axo_versions.split(",") if x]:
        try:
            a = Axol1tl(v)
            axos.append(a)
            detectors[f"axol1tl_{v}"] = {
                "name": a.name, "params": a.n_trainable_parameters, "train_s": 0.0,
                "score_hw": a.score_hardware, "trained_here": False,
            }
            print(f"[AXOL1TL {v}] loaded (pretrained CMS emulator)", flush=True)
        except Exception as e:
            print(f"[AXOL1TL {v}] unavailable: {type(e).__name__}: {e}", flush=True)

    # ---- evaluate ------------------------------------------------------
    results: dict = {
        "n_train": int(len(Xtr)), "n_test_bkg": int(len(Xte_b)),
        "n_glomeruli": N_TRIGGER_GLOMERULI, "detectors": {},
    }

    for key, d in detectors.items():
        t0 = time.time()
        if "score_g" in d:
            sb = d["score_g"](Xte_b)
            ss = {n: d["score_g"](x) for n, x in sigs_g.items()}
        else:
            sb = d["score_hw"](bkg_hw)
            ss = {n: d["score_hw"](sigs_hw[n]) for n in sigs_g}
        entry = {"name": d["name"], "trainable_parameters": d["params"],
                 "train_seconds": d["train_s"], "trained_here": d["trained_here"],
                 "score_seconds": time.time() - t0, "signals": {}}
        for k in ("diagnostics", "inference_parameters", "state_values"):
            if k in d:
                entry[k] = d[k]
        for name, s in ss.items():
            y = np.concatenate([np.zeros(len(sb)), np.ones(len(s))])
            entry["signals"][name] = {
                "auc": float(roc_auc_score(y, np.concatenate([sb, s]))),
                "n_signal": int(len(s)),
                **{f"eps_s_at_{e:g}": eps_s_at(sb, s, e) for e in WORKING_POINTS},
            }
        results["detectors"][key] = entry
        print(f"  scored {d['name']}", flush=True)

    # ---- report --------------------------------------------------------
    names = list(sigs_g)
    print(f"\n{'AUC':32s}" + "".join(f"{n:>17s}" for n in names))
    print("-" * (32 + 17 * len(names)))
    for e in results["detectors"].values():
        mark = "" if e["trained_here"] else "  *"
        print(f"{e['name'] + mark:32s}" + "".join(
            f"{e['signals'][n]['auc']:>17.4f}" for n in names))
    print("\n  * pretrained on CMS Zero Bias data, evaluated here off-distribution")

    for eps in WORKING_POINTS:
        print(f"\nsignal efficiency at eps_B = {eps:g}   (random gives {eps:g})")
        for e in results["detectors"].values():
            print(f"  {e['name']:30s}" + "".join(
                f"{e['signals'][n][f'eps_s_at_{eps:g}']:>17.5f}" for n in names))

    if args.save_scores:
        fly_d = detectors["fly"]
        sb = fly_d["score_g"](Xte_b)
        parts_s, parts_y, parts_src = [sb], [np.zeros(len(sb))], [np.zeros(len(sb))]
        for k, (name, xs) in enumerate(sigs_g.items(), start=1):
            ss = fly_d["score_g"](xs)
            parts_s.append(ss); parts_y.append(np.ones(len(ss)))
            parts_src.append(np.full(len(ss), k))
        args.save_scores.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.save_scores,
            score_fly=np.concatenate(parts_s),
            y_test=np.concatenate(parts_y).astype(np.int8),
            signal_id=np.concatenate(parts_src).astype(np.int8),
            signal_names=np.array(list(sigs_g), dtype=object),
        )
        print(f"wrote {args.save_scores}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwrote {args.out}", flush=True)
    for a in axos:
        a.close()


if __name__ == "__main__":
    main()
