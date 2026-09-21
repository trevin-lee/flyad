"""The mushroom body output neuron: a Bloom filter with graded weights.

Dasgupta, Sheehan, Stevens & Navlakha, "A neural data structure for novelty
detection", PNAS 115:13093 (2018).

Every Kenyon cell synapses onto the MBON. The synapses start strong. Each time
an odour is experienced, the synapses of the KCs it activated are depressed.
The MBON's response to a tag is therefore the mean surviving weight over that
tag's active KCs: high for something never smelled before, low for something
familiar.

Because depression is multiplicative, a KC hit n times has weight
(1 - alpha)^n ~ exp(-alpha * n). We therefore store activation *counts* and
apply the decay at scoring time -- numerically identical, but it lets the
depression rate be retuned without a second pass over the data.
"""

from __future__ import annotations

import numpy as np


class MushroomBody:
    """Novelty detector over FlyHash tags."""

    def __init__(self, n_kc: int, *, depression_rate: float | None = None) -> None:
        self.n_kc = n_kc
        self.counts = np.zeros(n_kc, dtype=np.float64)
        self.n_seen = 0
        self._depression_rate = depression_rate

    @property
    def n_trainable_parameters(self) -> int:
        """Also zero. `counts` is a tally, not a fitted parameter."""
        return 0

    def experience(self, tags: np.ndarray) -> None:
        """Depress the MBON synapses of every Kenyon cell in these tags.

        `tags` is (batch, n_winners) of KC indices, as returned by FlyHash.tag.
        This is the entire training procedure.
        """
        tags = np.atleast_2d(tags)
        np.add.at(self.counts, tags.ravel(), 1.0)
        self.n_seen += tags.shape[0]

    @property
    def depression_rate(self) -> float:
        """Depression per activation.

        If not set explicitly, choose it so that a *typical* Kenyon cell ends
        training with its weight depressed by roughly 1/e. A KC fires on a
        fraction (n_winners / n_kc) of events, so after N events the mean
        activation count is N * n_winners / n_kc; setting alpha to its
        reciprocal keeps the exponent in a range where the score still
        discriminates instead of saturating at zero.
        """
        if self._depression_rate is not None:
            return self._depression_rate
        mean_count = self.counts.mean()
        return 1.0 / mean_count if mean_count > 0 else 1.0

    def weights(self) -> np.ndarray:
        """Surviving MBON synaptic weight per Kenyon cell, in (0, 1]."""
        return np.exp(-self.depression_rate * self.counts)

    def novelty(self, tags: np.ndarray) -> np.ndarray:
        """Novelty score per event. Higher means less familiar.

        This is the MBON's firing rate: the mean surviving synaptic weight
        across the Kenyon cells the stimulus activated.
        """
        tags = np.atleast_2d(tags)
        return self.weights()[tags].mean(axis=1)

    def saturation(self) -> dict:
        """Diagnostics for whether the filter has been over-filled."""
        w = self.weights()
        return {
            "n_seen": self.n_seen,
            "kc_never_used": int((self.counts == 0).sum()),
            "mean_count": float(self.counts.mean()),
            "depression_rate": float(self.depression_rate),
            "mean_weight": float(w.mean()),
            "weight_spread": float(w.std()),
        }
