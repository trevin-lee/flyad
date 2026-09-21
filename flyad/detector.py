"""The assembled circuit: event -> glomeruli -> FlyHash -> mushroom body."""

from __future__ import annotations

import numpy as np

from .antennal_lobe import N_FLY_GLOMERULI, divisive_normalisation
from .flyhash import FlyHash, N_KENYON_CELLS, PN_PER_KC, WTA_FRACTION
from .mushroom_body import MushroomBody


class FlyAnomalyDetector:
    """One-class anomaly detector built from the fly olfactory circuit.

    Trained by showing it background events. There is no loss function, no
    optimiser and no gradient: `experience` simply depresses mushroom body
    synapses, and every parameter of the projection is random and frozen.
    """

    def __init__(
        self,
        *,
        n_inputs: int = N_FLY_GLOMERULI,
        n_kc: int = N_KENYON_CELLS,
        pn_per_kc: int = PN_PER_KC,
        wta_fraction: float = WTA_FRACTION,
        depression_rate: float | None = None,
        seed: int = 0,
    ) -> None:
        self.hash = FlyHash(
            n_inputs, n_kc=n_kc, pn_per_kc=pn_per_kc,
            wta_fraction=wta_fraction, seed=seed,
        )
        self.mb = MushroomBody(n_kc, depression_rate=depression_rate)

    @property
    def n_trainable_parameters(self) -> int:
        """Zero: nothing here is fitted by optimisation.

        No objective is minimised and no search is performed. The projection is
        drawn from a seed and frozen; winner-take-all is a rule, not a value.
        See `n_state_values` for the honest companion number.
        """
        return self.hash.n_trainable_parameters + self.mb.n_trainable_parameters

    @property
    def n_state_values(self) -> int:
        """Data-dependent numbers the detector holds: one count per Kenyon cell.

        The detector is nonparametric, not empty. Like a histogram or a Bloom
        filter it has capacity that grows with what it has seen -- but that
        capacity is tallied, never fitted. Quoting only the zero would be a
        half-truth.
        """
        return self.mb.n_kc

    def experience(self, pn: np.ndarray) -> None:
        """Show the fly some background. This is the whole training procedure."""
        self.mb.experience(self.hash.tag(pn))

    def score(self, pn: np.ndarray) -> np.ndarray:
        """Novelty per event. Higher means more anomalous."""
        return self.mb.novelty(self.hash.tag(pn))

    def decide(self, pn: np.ndarray, threshold: float) -> np.ndarray:
        """Boolean ANOMALY / NOT ANOMALY, i.e. which button the fly walks to."""
        return self.score(pn) >= threshold

    def diagnostics(self) -> dict:
        return self.mb.saturation()


def normalise_features(x: np.ndarray) -> np.ndarray:
    """Divisive normalisation for inputs that are already per-event features."""
    x = np.asarray(x, dtype=np.float64)
    x = np.abs(x)
    return divisive_normalisation(x)
