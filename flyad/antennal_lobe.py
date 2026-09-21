"""Antennal lobe: the part that is the same whatever the fly is smelling.

Only one operation here is dataset-independent, and it is the one that earns
its place -- divisive normalisation by lateral inhibition. Each study supplies
its own mapping from detector objects onto glomeruli:

    flyad.lhco.lobe     dijet constituents -> 50 glomeruli
    flyad.adc2021.lobe  L1 trigger objects -> 51 glomeruli
"""

from __future__ import annotations

import numpy as np

# A real Drosophila has roughly 50 glomeruli; both encodings land near it.
N_FLY_GLOMERULI = 50


def divisive_normalisation(
    orn: np.ndarray,
    *,
    r_max: float = 1.0,
    sigma: float = 0.05,
    m: float = 0.05,
    exponent: float = 1.5,
) -> np.ndarray:
    """Olsen-Bhandawat-Wilson divisive normalisation of the antennal lobe.

    PN_i = r_max * ORN_i^n / (sigma^n + m^n + sum_j ORN_j^n)

    In the fly this makes the PN representation roughly invariant to odour
    concentration. Applied to jets it is invariant to the overall pT scale,
    which is exactly what keeps the detector from simply rediscovering the
    trigger threshold.

    Operates on the last axis, so it accepts a single event or a batch.
    """
    orn = np.asarray(orn, dtype=np.float64)
    scale = np.sum(orn, axis=-1, keepdims=True)
    scale = np.where(scale > 0, scale, 1.0)
    x = orn / scale  # put every event on a common scale before the nonlinearity

    xn = np.power(x, exponent)
    denom = sigma**exponent + m**exponent + np.sum(xn, axis=-1, keepdims=True)
    return r_max * xn / denom


