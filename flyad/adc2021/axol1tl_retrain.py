"""The AXOL1TL architectures, retrained on ADC2021 background.

The deployed model in `flyad.axol1tl` carries CMS's own weights, trained on
real Zero Bias data. Evaluating it on ADC2021 is therefore off-distribution and
its score there is not its CMS performance. These classes close that gap: the
*same topology*, the *same input scaling*, and the *same scoring rule*, but
fitted to the ADC2021 background everything else in this repository trains on.

Topologies are read from the emulator source, not guessed:

  v3 (the version described in arXiv:2411.19506)
      57 -> 32 -> 16 -> 8 latent; only the encoder runs at inference and the
      anomaly score is the sum of squared latent means, sum_i mu_i^2.

  v5 (the version validated bit-exactly here)
      57 -> 29 -> 10 -> 9 -> 6 -> 4 -> 6 -> 9 -> 10, and -- unusually -- the
      decoder reconstructs the 10-dimensional post-ReLU activation rather than
      the 57 inputs. The score is the squared distance between the two. Doing
      the reconstruction in feature space keeps the error computation 10-wide
      instead of 57-wide, which is the sort of thing you do for an FPGA.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
AXO_DIR = ROOT / "external" / "AXOL1TL"


def load_scales(version: str = "v5") -> tuple[np.ndarray, np.ndarray]:
    """Read (offsets, shifts) out of the emulator's own scales.h."""
    text = (AXO_DIR / f"AXOL1TL_{version}" / "scales.h").read_text()

    def grab(name: str) -> np.ndarray:
        m = re.search(rf"{name}\[[^\]]*\]\s*=\s*\{{([^}}]*)\}}", text)
        if not m:
            raise ValueError(f"{name} not found in {version}/scales.h")
        vals = [float(v.strip().rstrip(".")) for v in m.group(1).split(",") if v.strip()]
        return np.asarray(vals, dtype=np.float64)

    return grab("ad_offsets"), grab("ad_shift")


def scale_inputs(hw: np.ndarray, version: str = "v5") -> np.ndarray:
    """Apply the trigger's own integer scaling: (raw - offset) >> shift."""
    off, sh = load_scales(version)
    return (np.asarray(hw, dtype=np.float64) - off) / np.power(2.0, sh)


class _Retrained:
    """Shared training loop for the AXOL1TL topologies."""

    version = "v5"

    def __init__(self, *, epochs: int = 12, batch_size: int = 1024, lr: float = 1e-3,
                 beta: float = 1e-3, seed: int = 0, device: str = "cpu") -> None:
        import torch

        torch.manual_seed(seed)
        self.torch = torch
        self.epochs, self.batch_size, self.lr, self.beta = epochs, batch_size, lr, beta
        self.device = device
        self._build()

    def _build(self) -> None:
        raise NotImplementedError

    def _modules(self):
        raise NotImplementedError

    def _loss(self, batch):
        raise NotImplementedError

    @property
    def n_trainable_parameters(self) -> int:
        return sum(p.numel() for m in self._modules() for p in m.parameters()
                   if p.requires_grad)

    def _prep(self, hw: np.ndarray):
        return self.torch.tensor(
            scale_inputs(hw, self.version), dtype=self.torch.float32, device=self.device
        )

    def fit(self, hw: np.ndarray, *, verbose: bool = False):
        torch = self.torch
        x = self._prep(hw)
        params = [p for m in self._modules() for p in m.parameters()]
        opt = torch.optim.Adam(params, lr=self.lr)
        n = len(x)
        for epoch in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            tot = 0.0
            for i in range(0, n, self.batch_size):
                b = x[perm[i : i + self.batch_size]]
                loss = self._loss(b)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item() * len(b)
            if verbose:
                print(f"  epoch {epoch + 1:3d}/{self.epochs}  loss {tot / n:.5f}", flush=True)
        return self

    def score_hardware(self, hw: np.ndarray) -> np.ndarray:
        torch = self.torch
        x = self._prep(hw)
        out = []
        with torch.no_grad():
            for i in range(0, len(x), 8192):
                out.append(self._score(x[i : i + 8192]).cpu().numpy())
        return np.concatenate(out)


class Axol1tlV3Retrained(_Retrained):
    """57 -> 32 -> 16 -> 8 VAE; score = sum of squared latent means."""

    version = "v3"
    name = "AXOL1TL v3 architecture, retrained here"

    def _build(self) -> None:
        nn = self.torch.nn
        self.e1 = nn.Linear(57, 32).to(self.device)
        self.e2 = nn.Linear(32, 16).to(self.device)
        self.mu = nn.Linear(16, 8).to(self.device)
        self.lv = nn.Linear(16, 8).to(self.device)
        self.d1 = nn.Linear(8, 16).to(self.device)
        self.d2 = nn.Linear(16, 32).to(self.device)
        self.d3 = nn.Linear(32, 57).to(self.device)

    def _modules(self):
        return (self.e1, self.e2, self.mu, self.lv, self.d1, self.d2, self.d3)

    def _encode(self, x):
        F = self.torch.nn.functional
        h = F.relu(self.e2(F.relu(self.e1(x))))
        return self.mu(h), self.lv(h)

    def _loss(self, b):
        torch = self.torch
        F = torch.nn.functional
        mu, lv = self._encode(b)
        z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)
        rec = self.d3(F.relu(self.d2(F.relu(self.d1(z)))))
        mse = ((rec - b) ** 2).mean()
        kl = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
        return mse + self.beta * kl

    def _score(self, x):
        mu, _ = self._encode(x)
        return (mu ** 2).sum(dim=1)

    @property
    def n_inference_parameters(self) -> int:
        return sum(p.numel() for m in (self.e1, self.e2, self.mu)
                   for p in m.parameters() if p.requires_grad)


class Axol1tlV5Retrained(_Retrained):
    """Exact v5 topology: a feature-space autoencoder on the 10-wide activation."""

    version = "v5"
    name = "AXOL1TL v5 architecture, retrained here"

    def _build(self) -> None:
        nn = self.torch.nn
        dims_enc = [(57, 29), (29, 10), (10, 9), (9, 6), (6, 4)]
        dims_dec = [(4, 6), (6, 9), (9, 10)]
        self.enc = nn.ModuleList([nn.Linear(a, b) for a, b in dims_enc]).to(self.device)
        self.dec = nn.ModuleList([nn.Linear(a, b) for a, b in dims_dec]).to(self.device)

    def _modules(self):
        return (self.enc, self.dec)

    def _forward(self, x):
        F = self.torch.nn.functional
        h1 = F.relu(self.enc[0](x))
        h2 = F.relu(self.enc[1](h1))      # the reconstruction target
        h = F.relu(self.enc[2](h2))
        h = F.relu(self.enc[3](h))
        z = self.enc[4](h)                # 4-wide latent, no activation
        d = F.relu(self.dec[0](z))
        d = F.relu(self.dec[1](d))
        rec = self.dec[2](d)
        return rec, h2

    def _loss(self, b):
        rec, h2 = self._forward(b)
        # detach the target so the trivial solution (collapse h2 to zero) is not
        # directly rewarded; the encoder still trains through the latent path
        return ((rec - h2.detach()) ** 2).mean()

    def _score(self, x):
        rec, h2 = self._forward(x)
        return ((rec - h2) ** 2).sum(dim=1)
