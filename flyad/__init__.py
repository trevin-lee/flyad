"""A fruit fly decides whether your LHC event is new physics.

The circuit lives here; the two studies that feed it live in subpackages.

    flyad.lhco      LHC Olympics 2020 dijets (offline, resonant)
    flyad.adc2021   ADC2021 / AXOL1TL L1 trigger objects (online)
"""

from .antennal_lobe import N_FLY_GLOMERULI, divisive_normalisation
from .descending import descending_drive
from .detector import FlyAnomalyDetector
from .flyhash import FlyHash
from .mushroom_body import MushroomBody

__all__ = [
    "FlyAnomalyDetector",
    "FlyHash",
    "MushroomBody",
    "N_FLY_GLOMERULI",
    "descending_drive",
    "divisive_normalisation",
]
