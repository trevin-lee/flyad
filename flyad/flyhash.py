"""FlyHash: the projection neuron -> Kenyon cell expansion.

Dasgupta, Stevens & Navlakha, "A neural algorithm for a fundamental computing
problem", Science 358:793 (2017).

Three properties, all of which invert what a conventional autoencoder does:

  1. The projection *expands*, 50 -> 2000, instead of compressing to a
     bottleneck.
  2. The projection matrix is sparse, binary and random. It is never trained.
  3. A winner-take-all step driven by the APL neuron keeps only the top ~5% of
     Kenyon cells, producing a sparse binary tag.

The result is a locality-sensitive hash: similar inputs get overlapping tags.
"""

from __future__ import annotations

import numpy as np

N_KENYON_CELLS = 2000  # roughly the count in a real Drosophila mushroom body
PN_PER_KC = 6  # each KC samples ~6 projection neurons
WTA_FRACTION = 0.05  # APL inhibition leaves ~5% of KCs active


class FlyHash:
    """Sparse binary random projection followed by winner-take-all."""

    def __init__(
        self,
        n_inputs: int,
        *,
        n_kc: int = N_KENYON_CELLS,
        pn_per_kc: int = PN_PER_KC,
        wta_fraction: float = WTA_FRACTION,
        seed: int = 0,
    ) -> None:
        if pn_per_kc > n_inputs:
            raise ValueError(f"pn_per_kc={pn_per_kc} exceeds n_inputs={n_inputs}")
        self.n_inputs = n_inputs
        self.n_kc = n_kc
        self.pn_per_kc = pn_per_kc
        self.wta_fraction = wta_fraction
        self.n_winners = max(1, int(round(wta_fraction * n_kc)))
        self.seed = seed

        rng = np.random.default_rng(seed)
        # Each row of `sampled` lists the PNs feeding one Kenyon cell.
        self.sampled = np.stack(
            [rng.choice(n_inputs, size=pn_per_kc, replace=False) for _ in range(n_kc)]
        )

    @property
    def n_trainable_parameters(self) -> int:
        """Zero. The connectivity is random and fixed; nothing here is learned."""
        return 0

    def _block_rows(self, target_elements: int = 30_000_000) -> int:
        """Rows per block, so the (rows x n_kc) drive matrix stays bounded.

        The obvious vectorisation, ``pn[:, self.sampled].sum(-1)``, materialises
        a (batch, n_kc, pn_per_kc) array. At 200k events and 20k Kenyon cells
        that is 192 GB. Accumulating one sampled projection neuron at a time
        over blocks of rows keeps peak memory at a few hundred MB no matter how
        wide the expansion gets.
        """
        return max(1, target_elements // max(1, self.n_kc))

    def _drive(self, blk: np.ndarray) -> np.ndarray:
        acc = np.zeros((len(blk), self.n_kc), dtype=np.float64)
        for k in range(self.pn_per_kc):
            acc += blk[:, self.sampled[:, k]]
        return acc

    def kc_activity(self, pn: np.ndarray) -> np.ndarray:
        """Raw Kenyon cell drive: sum of each KC's sampled projection neurons."""
        pn = np.atleast_2d(np.asarray(pn, dtype=np.float64))
        out = np.empty((len(pn), self.n_kc), dtype=np.float64)
        step = self._block_rows()
        for i in range(0, len(pn), step):
            out[i : i + step] = self._drive(pn[i : i + step])
        return out

    def tag(self, pn: np.ndarray) -> np.ndarray:
        """Indices of the winning Kenyon cells, shape (batch, n_winners).

        Winner-take-all stands in for the APL neuron's feedback inhibition,
        which silences all but the most strongly driven KCs. Computed blockwise
        so the full drive matrix is never held at once.
        """
        pn = np.atleast_2d(np.asarray(pn, dtype=np.float64))
        out = np.empty((len(pn), self.n_winners), dtype=np.int64)
        step = self._block_rows()
        for i in range(0, len(pn), step):
            acc = self._drive(pn[i : i + step])
            # argpartition is O(n); we do not care about the order among winners.
            out[i : i + step] = np.argpartition(
                -acc, self.n_winners - 1, axis=1
            )[:, : self.n_winners]
        return out

    def tag_dense(self, pn: np.ndarray) -> np.ndarray:
        """The tag as a dense binary matrix, shape (batch, n_kc)."""
        idx = self.tag(pn)
        out = np.zeros((idx.shape[0], self.n_kc), dtype=bool)
        np.put_along_axis(out, idx, True, axis=1)
        return out


class WiredFlyHash:
    """FlyHash driven by an explicit (n_kc, n_inputs) connectivity matrix.

    `FlyHash` above *samples* connectivity matching the statistics Caron et al.
    reported. This class takes a matrix instead, so the measured FlyWire wiring
    can be dropped in and compared against a degree-matched random control with
    everything else held fixed.

    Note what such a comparison can and cannot show. Our input channels are jet
    patches, not odours, so no channel corresponds to any particular glomerulus.
    The test is therefore whether the real wiring's *statistics* -- its claw
    distribution, sparsity and weight spread -- help or hurt, not whether the
    specific glomerular identities matter, which this data cannot address.
    """

    def __init__(self, weights: np.ndarray, *, wta_fraction: float = WTA_FRACTION,
                 normalise: bool = True) -> None:
        W = np.asarray(weights, dtype=np.float64)
        if normalise:
            # synapse counts vary by orders of magnitude between Kenyon cells;
            # without this the winner-take-all just picks the best-innervated
            # cells every time, regardless of the stimulus
            scale = W.sum(axis=1, keepdims=True)
            W = np.divide(W, np.where(scale > 0, scale, 1.0))
        self.W = W
        self.n_kc, self.n_inputs = W.shape
        self.wta_fraction = wta_fraction
        self.n_winners = max(1, int(round(wta_fraction * self.n_kc)))

    @property
    def n_trainable_parameters(self) -> int:
        """Still zero. The matrix was measured in a fly, not fitted here."""
        return 0

    def _block_rows(self, target_elements: int = 30_000_000) -> int:
        return max(1, target_elements // max(1, self.n_kc))

    def tag(self, pn: np.ndarray) -> np.ndarray:
        pn = np.atleast_2d(np.asarray(pn, dtype=np.float64))
        out = np.empty((len(pn), self.n_winners), dtype=np.int64)
        step = self._block_rows()
        for i in range(0, len(pn), step):
            acc = pn[i : i + step] @ self.W.T
            out[i : i + step] = np.argpartition(
                -acc, self.n_winners - 1, axis=1)[:, : self.n_winners]
        return out
