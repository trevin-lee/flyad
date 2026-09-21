"""Train the fly on background only, then compare it against real baselines.

Run:  python scripts/run_analysis.py
Writes outputs/results.json and outputs/roc.npz
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

from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

from flyad import FlyAnomalyDetector  # noqa: E402
from flyad.baselines import AutoencoderBaseline, GaussianBaseline  # noqa: E402


def hep_metrics(y_true: np.ndarray, score: np.ndarray) -> dict:
    """AUC plus the significance-improvement numbers HEP actually reports."""
    # drop_intermediate=True prunes points sklearn considers collinear, which on
    # a well-separated ROC deletes the entire low-eps_B region that SIC is read
    # from. Keep every point.
    fpr, tpr, _ = roc_curve(y_true, score, drop_intermediate=False)
    auc = roc_auc_score(y_true, score)

    # SIC = eps_S / sqrt(eps_B), the factor by which S/sqrt(B) improves.
    ok = fpr > 1e-6
    sic = np.zeros_like(tpr)
    sic[ok] = tpr[ok] / np.sqrt(fpr[ok])
    # ignore the noisy far tail where almost no background survives
    stable = fpr > 1e-4
    max_sic = float(sic[stable].max()) if stable.any() else float("nan")

    def tpr_at(target_fpr: float) -> float:
        return float(np.interp(target_fpr, fpr, tpr))

    return {
        "auc": float(auc),
        "max_sic": max_sic,
        "eps_s_at_eps_b_1e-2": tpr_at(1e-2),
        "eps_s_at_eps_b_1e-3": tpr_at(1e-3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glomeruli", type=Path, default=ROOT / "outputs" / "lhco" / "glomeruli.npz")
    ap.add_argument("--n-train", type=int, default=600_000)
    ap.add_argument("--n-test-bkg", type=int, default=100_000)
    ap.add_argument("--n-test-sig", type=int, default=100_000)
    ap.add_argument("--n-kc", type=int, default=2000)
    ap.add_argument("--ae-epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-sweep", action="store_true",
                    help="reuse the kc_sweep already in results.json")
    args = ap.parse_args()

    d = np.load(args.glomeruli)
    X, y = d["X"].astype(np.float64), d["y"]
    print(f"loaded {X.shape[0]:,} events  ({int((y==1).sum()):,} signal)", flush=True)

    rng = np.random.default_rng(args.seed)
    bkg = np.where(y == 0)[0]
    sig = np.where(y == 1)[0]
    rng.shuffle(bkg)
    rng.shuffle(sig)

    n_train = min(args.n_train, len(bkg) - args.n_test_bkg)
    train_idx = bkg[:n_train]
    test_idx = np.concatenate(
        [bkg[n_train : n_train + args.n_test_bkg], sig[: args.n_test_sig]]
    )
    rng.shuffle(test_idx)

    Xtr, Xte, yte = X[train_idx], X[test_idx], y[test_idx]
    print(
        f"train: {len(Xtr):,} background events (no labels used)\n"
        f"test:  {len(Xte):,} events, {int(yte.sum()):,} signal",
        flush=True,
    )

    results: dict[str, dict] = {}

    # ---- the fly -------------------------------------------------------
    print("\n[fly] experiencing background...", flush=True)
    t0 = time.time()
    fly = FlyAnomalyDetector(n_inputs=X.shape[1], n_kc=args.n_kc, seed=args.seed)
    for i in range(0, len(Xtr), 50_000):
        fly.experience(Xtr[i : i + 50_000])
    train_s = time.time() - t0
    s_fly = fly.score(Xte)
    results["fly"] = {
        "name": "Fly olfactory circuit",
        "trainable_parameters": fly.n_trainable_parameters,
        "train_seconds": train_s,
        "diagnostics": fly.diagnostics(),
        **hep_metrics(yte, s_fly),
    }
    print(f"  trained in {train_s:.1f}s  AUC={results['fly']['auc']:.4f}", flush=True)

    # ---- Gaussian ------------------------------------------------------
    print("[gaussian] fitting...", flush=True)
    t0 = time.time()
    g = GaussianBaseline().fit(Xtr)
    s_gauss = g.score(Xte)
    results["gaussian"] = {
        "name": g.name,
        "trainable_parameters": g.n_trainable_parameters,
        "train_seconds": time.time() - t0,
        **hep_metrics(yte, s_gauss),
    }
    print(f"  AUC={results['gaussian']['auc']:.4f}", flush=True)

    # ---- autoencoder ---------------------------------------------------
    print("[autoencoder] training...", flush=True)
    t0 = time.time()
    ae = AutoencoderBaseline(X.shape[1], epochs=args.ae_epochs, seed=args.seed)
    ae.fit(Xtr, verbose=True)
    s_ae = ae.score(Xte)
    results["autoencoder"] = {
        "name": ae.name,
        "trainable_parameters": ae.n_trainable_parameters,
        "train_seconds": time.time() - t0,
        **hep_metrics(yte, s_ae),
    }
    print(f"  AUC={results['autoencoder']['auc']:.4f}", flush=True)

    # ---- report --------------------------------------------------------
    print(f"\n{'detector':26s} {'params':>10s} {'train':>8s} {'AUC':>7s} {'maxSIC':>7s}")
    print("-" * 64)
    for k, r in results.items():
        if k == "kc_sweep":
            continue
        print(
            f"{r['name']:26s} {r['trainable_parameters']:>10,} "
            f"{r['train_seconds']:>7.1f}s {r['auc']:>7.4f} {r['max_sic']:>7.2f}"
        )

    # ---- does the expansion actually help? ----------------------------
    # The fly expands 50 -> 2000. Sweep the Kenyon cell count to see whether
    # that matters or whether any random projection would do.
    prior = ROOT / "outputs" / "lhco" / "results.json"
    if args.skip_sweep and prior.exists():
        results["kc_sweep"] = json.loads(prior.read_text()).get("kc_sweep", [])
        print("\n[sweep] reusing existing kc_sweep", flush=True)
        sweep = None
    else:
        print("\n[sweep] AUC vs number of Kenyon cells", flush=True)
    # Subsampled: the sweep is about the trend, and the widest expansions are
    # expensive to score. The headline fly above uses the full split.
        sw_tr, sw_te = Xtr[:200_000], Xte[:100_000]
        sw_yte = yte[:100_000]
        sweep = []
        for n_kc in (50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000):
            f = FlyAnomalyDetector(n_inputs=X.shape[1], n_kc=n_kc, seed=args.seed)
            for i in range(0, len(sw_tr), 50_000):
                f.experience(sw_tr[i : i + 50_000])
            a = roc_auc_score(sw_yte, f.score(sw_te))
            sweep.append({"n_kc": n_kc, "auc": float(a)})
            flag = "  <- fly" if n_kc == 2000 else ""
            print(f"  n_kc={n_kc:6,}  AUC={a:.4f}{flag}", flush=True)
        results["kc_sweep"] = sweep

    out = ROOT / "outputs" / "lhco" / "results.json"
    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwrote {out}", flush=True)

    # ROC curves + the fly's scores, for plotting and for picking highlights
    fpr_f, tpr_f, _ = roc_curve(yte, s_fly, drop_intermediate=False)
    np.savez_compressed(
        ROOT / "outputs" / "lhco" / "roc.npz",
        fpr_fly=fpr_f, tpr_fly=tpr_f,
        score_fly=s_fly, score_gaussian=s_gauss, score_ae=s_ae,
        y_test=yte, test_idx=test_idx,
    )
    print(f"wrote {ROOT / 'outputs' / 'roc.npz'}", flush=True)


if __name__ == "__main__":
    main()
