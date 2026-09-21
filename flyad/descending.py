"""The descending pathway: mushroom body output to locomotor drive.

In the fly, MBON activity does not steer toward a goal. It biases *turning*, in
proportion to the valence the mushroom body assigns -- aversive input raises the
turn rate and the animal drifts away, appetitive input lowers it and the animal
lingers. Position is an emergent consequence, never a setpoint.

This module is therefore the only place a novelty score becomes movement, and it
is deliberately tiny: a sign, a gain, and a clip. Anything more elaborate --
a heading error, a target, a controller nulling a distance -- would be robotics
wearing a fly costume, because no such circuit exists in the animal.
"""

from __future__ import annotations

import numpy as np


def descending_drive(
    novelty: float,
    threshold: float,
    *,
    gain: float = 1.5,
    base: float = 1.0,
    max_turn: float = 0.55,
) -> np.ndarray:
    """MBON novelty -> a two-sided descending drive (left, right).

    The sign says which way to turn; the magnitude scales with how far the score
    sits from threshold, so a marginal event produces a weak, wavering turn and a
    confident one a decisive turn. Positive turn steers left, the ANOMALY side.

    The returned pair is the action expected by flygym's HybridTurningController,
    clipped to its (-0.5, 1.5) action space.

    On the gain. It was 9.0, which sounds decisive and is useless: with
    max_turn=0.55 the drive clips at an evidence margin of only 0.061, and
    measured against the real score distributions that saturates 83% of LHCO
    and 81% of ADC2021 events. Every confident event produced an identical
    readout, so the turn carried one bit rather than a magnitude. The median
    |evidence| is 0.18 (LHCO) and 0.13 (ADC2021), with 90th percentiles of 0.33
    and 0.41, so 1.5 puts roughly nine events in ten inside the linear range and
    the turn rate actually means something.
    """
    evidence = float(novelty) - float(threshold)
    turn = float(np.clip(gain * evidence, -max_turn, max_turn))
    return np.clip([base - turn, base + turn], -0.5, 1.5).astype(np.float32)
