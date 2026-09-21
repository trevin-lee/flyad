"""Baselines the fly is measured against.

The point of the comparison is not to make the fly look good. Both baselines
see exactly the same 50 glomeruli, trained one-class on exactly the same
background events, so the only thing that differs is the detector.
"""

from __future__ import annotations

import numpy as np


class GaussianBaseline:
    """Mahalanobis distance under a Gaussian fitted to the background.

    Cheap, strong, and the thing any sensible person tries first.
    """

    name = "Gaussian (Mahalanobis)"

    def __init__(self, *, shrinkage: float = 1e-6) -> None:
        self.shrinkage = shrinkage
        self.mean_: np.ndarray | None = None
        self.prec_: np.ndarray | None = None

    @property
    def n_trainable_parameters(self) -> int:
        if self.mean_ is None:
            return 0
        d = len(self.mean_)
        return d + d * (d + 1) // 2

    def fit(self, x: np.ndarray) -> "GaussianBaseline":
        x = np.asarray(x, dtype=np.float64)
        self.mean_ = x.mean(axis=0)
        cov = np.cov(x, rowvar=False)
        cov += self.shrinkage * np.eye(len(cov)) * np.trace(cov) / len(cov)
        self.prec_ = np.linalg.pinv(cov)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        d = np.asarray(x, dtype=np.float64) - self.mean_
        return np.einsum("ij,jk,ik->i", d, self.prec_, d)


class AutoencoderBaseline:
    """A conventional bottleneck autoencoder scored by reconstruction error.

    This is the standard approach in the LHC anomaly detection literature and
    the thing the fly circuit inverts: it compresses where the fly expands, and
    it is fitted by gradient descent where the fly is not fitted at all.
    """

    name = "Autoencoder"

    def __init__(
        self,
        n_inputs: int,
        *,
        hidden: tuple[int, ...] = (32, 16),
        latent: int = 6,
        epochs: int = 20,
        batch_size: int = 1024,
        lr: float = 1e-3,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        import torch
        import torch.nn as nn

        torch.manual_seed(seed)
        self.torch, self.nn = torch, nn
        self.epochs, self.batch_size, self.lr = epochs, batch_size, lr
        self.device = device

        enc_dims = [n_inputs, *hidden, latent]
        dec_dims = [latent, *reversed(hidden), n_inputs]

        def stack(dims, final_activation):
            layers = []
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                if i < len(dims) - 2 or final_activation:
                    layers.append(nn.ReLU())
            return layers

        self.model = nn.Sequential(
            *stack(enc_dims, False), *stack(dec_dims, False)
        ).to(device)
        self.mu_: np.ndarray | None = None
        self.sd_: np.ndarray | None = None

    @property
    def n_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _standardise(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mu_) / self.sd_

    def fit(self, x: np.ndarray, *, verbose: bool = False) -> "AutoencoderBaseline":
        torch, nn = self.torch, self.nn
        x = np.asarray(x, dtype=np.float32)
        self.mu_ = x.mean(axis=0)
        self.sd_ = x.std(axis=0) + 1e-8
        xt = torch.tensor(self._standardise(x), dtype=torch.float32, device=self.device)

        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        n = len(xt)
        for epoch in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            total = 0.0
            for i in range(0, n, self.batch_size):
                b = xt[perm[i : i + self.batch_size]]
                opt.zero_grad()
                loss = loss_fn(self.model(b), b)
                loss.backward()
                opt.step()
                total += loss.item() * len(b)
            if verbose:
                print(f"  epoch {epoch + 1:3d}/{self.epochs}  loss {total / n:.5f}", flush=True)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        torch = self.torch
        x = np.asarray(x, dtype=np.float32)
        xt = torch.tensor(self._standardise(x), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            out = []
            for i in range(0, len(xt), 8192):
                b = xt[i : i + 8192]
                out.append(((self.model(b) - b) ** 2).mean(dim=1).cpu().numpy())
        return np.concatenate(out)


class AxolitlVAE:
    """A stand-in for AXOL1TL: a VAE scored by its latent means.

    Follows the architecture described in Gandrakota (for CMS), arXiv:2411.19506 --
    dense feed-forward encoder and decoder, latent size 8, and crucially only the
    *encoder* runs at inference: the anomaly score is approximated by the sum of
    squared latent means, sum_i mu_i^2.

    This is not the CMS model. It is not trained on Zero Bias data, not quantised,
    and not synthesised to an FPGA; it is the same architecture and scoring rule
    trained on the same simulated background as everything else here, so that the
    comparison in this repository is like-for-like.
    """

    name = "VAE, latent score (AXOL1TL-like)"

    def __init__(
        self,
        n_inputs: int,
        *,
        hidden: tuple[int, ...] = (32, 16),
        latent: int = 8,
        beta: float = 1e-3,
        epochs: int = 15,
        batch_size: int = 1024,
        lr: float = 1e-3,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        import torch
        import torch.nn as nn

        torch.manual_seed(seed)
        self.torch, self.nn = torch, nn
        self.epochs, self.batch_size, self.lr, self.beta = epochs, batch_size, lr, beta
        self.device, self.latent = device, latent

        def mlp(dims):
            layers = []
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                if i < len(dims) - 2:
                    layers.append(nn.ReLU())
            return nn.Sequential(*layers)

        self.enc = mlp([n_inputs, *hidden]).to(device)
        self.to_mu = nn.Linear(hidden[-1], latent).to(device)
        self.to_logvar = nn.Linear(hidden[-1], latent).to(device)
        self.dec = mlp([latent, *reversed(hidden), n_inputs]).to(device)
        self.mu_: np.ndarray | None = None
        self.sd_: np.ndarray | None = None

    def _modules(self):
        return (self.enc, self.to_mu, self.to_logvar, self.dec)

    @property
    def n_trainable_parameters(self) -> int:
        return sum(p.numel() for m in self._modules() for p in m.parameters()
                   if p.requires_grad)

    @property
    def n_inference_parameters(self) -> int:
        """Only the encoder is on the FPGA; the decoder is a training-time cost."""
        return sum(p.numel() for m in (self.enc, self.to_mu) for p in m.parameters()
                   if p.requires_grad)

    def _standardise(self, x):
        return (x - self.mu_) / self.sd_

    def fit(self, x: np.ndarray, *, verbose: bool = False) -> "AxolitlVAE":
        torch = self.torch
        x = np.asarray(x, dtype=np.float32)
        self.mu_, self.sd_ = x.mean(axis=0), x.std(axis=0) + 1e-8
        xt = torch.tensor(self._standardise(x), dtype=torch.float32, device=self.device)

        params = [p for m in self._modules() for p in m.parameters()]
        opt = torch.optim.Adam(params, lr=self.lr)
        n = len(xt)
        for epoch in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            tot = 0.0
            for i in range(0, n, self.batch_size):
                b = xt[perm[i : i + self.batch_size]]
                h = self.enc(b)
                mu, logvar = self.to_mu(h), self.to_logvar(h)
                std = torch.exp(0.5 * logvar)
                z = mu + std * torch.randn_like(std)
                rec = self.dec(z)
                mse = ((rec - b) ** 2).mean()
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = mse + self.beta * kl
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item() * len(b)
            if verbose:
                print(f"  epoch {epoch + 1:3d}/{self.epochs}  loss {tot / n:.5f}", flush=True)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        """sum_i mu_i^2 -- the quantity AXOL1TL actually computes in hardware."""
        torch = self.torch
        x = np.asarray(x, dtype=np.float32)
        xt = torch.tensor(self._standardise(x), dtype=torch.float32, device=self.device)
        out = []
        with torch.no_grad():
            for i in range(0, len(xt), 8192):
                mu = self.to_mu(self.enc(xt[i : i + 8192]))
                out.append((mu ** 2).sum(dim=1).cpu().numpy())
        return np.concatenate(out)
