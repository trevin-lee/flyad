"""The real AXOL1TL, driven through its CMS hls4ml emulator.

This is not a reimplementation. `external/AXOL1TL` is the CMS collaboration's
own bit-exact emulator of the anomaly detection model deployed in the Level-1
Global Trigger, quantised weights and all; we compile it and call it through a
thin ctypes shim. The score returned is the same quantity the trigger computes
in firmware.

IMPORTANT CAVEAT. AXOL1TL was trained on real CMS Zero Bias data, in raw L1
hardware units. ADC2021 is Delphes-simulated with a different detector card and
stores physical quantities. Converting one to the other (below) reproduces the
L1 scales but cannot reproduce the training distribution, so AXOL1TL is being
evaluated off-distribution here. A weak result on this benchmark is NOT
evidence about its performance in CMS.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
SHIM = ROOT / "external" / "shim" / "libaxoshim.so"
EMULATOR_LIB = ROOT / "external" / "hls4mlEmulatorExtras" / "lib64"
MODELS = {
    v: ROOT / "external" / "AXOL1TL" / f"AXOL1TL_{v}" / f"GTADModel_{v}.so"
    for v in ("v1", "v3", "v4", "v5", "v6")
}
# read_result's expected type differs by version; see external/shim/axo_shim.cpp
VARIANT = {"v1": 0, "v3": 1, "v4": 1, "v5": 2, "v6": 2}
# latent width, for reference: v1 is 13, v3/v4 are 8, v5/v6 emit the score directly
LATENT = {"v1": 13, "v3": 8, "v4": 8, "v5": None, "v6": None}

# CMS Level-1 hardware scales. The emulator's own offset/shift table confirms
# these: calorimeter objects carry a 144-bin azimuth (offset 72), muons a
# 576-bin one (offset ~286).
PT_LSB = 0.5                     # GeV per count
CALO_ETA_LSB = 0.0435
CALO_PHI_NBINS = 144
MUON_ETA_LSB = 0.010875
MUON_PHI_NBINS = 576

# slot layout, shared by ADC2021 and the emulator input vector
MET_SLOT = 0
EGAMMA = slice(1, 5)
MUON = slice(5, 9)
JET = slice(9, 19)


def to_hardware(events: np.ndarray) -> np.ndarray:
    """Convert ADC2021 (n, 19, 4) physical units to the emulator's (n, 57) input.

    pT in 0.5 GeV counts; eta in detector units; phi wrapped to [0, nbins).
    Muons use the finer muon scales, everything else the calorimeter scales.
    """
    ev = np.asarray(events, dtype=np.float64)
    n = len(ev)
    pt, eta, phi = ev[..., 0], ev[..., 1], ev[..., 2]

    out = np.zeros((n, 19, 3), dtype=np.float64)
    out[..., 0] = np.rint(pt / PT_LSB)

    is_muon = np.zeros(19, dtype=bool)
    is_muon[MUON] = True

    eta_lsb = np.where(is_muon, MUON_ETA_LSB, CALO_ETA_LSB)[None, :]
    nbins = np.where(is_muon, MUON_PHI_NBINS, CALO_PHI_NBINS)[None, :]

    out[..., 1] = np.rint(eta / eta_lsb)
    out[..., 2] = np.rint((phi % (2 * np.pi)) / (2 * np.pi) * nbins)

    # zero-padded slots stay zero in every component
    empty = pt <= 0
    out[empty] = 0.0
    # MET has no eta
    out[:, MET_SLOT, 1] = 0.0
    return np.ascontiguousarray(out.reshape(n, 57))


class Axol1tl:
    """Handle on one compiled AXOL1TL version."""

    def __init__(self, version: str = "v5") -> None:
        if not SHIM.exists():
            raise FileNotFoundError(
                f"{SHIM} not built. See external/ build steps in the README."
            )
        so = MODELS[version]
        if not so.exists():
            raise FileNotFoundError(f"{so} not built (run `make` in external/AXOL1TL)")

        # the emulator interface library must be resolvable before the model
        ctypes.CDLL(str(EMULATOR_LIB / "libemulator_interface.so"), mode=ctypes.RTLD_GLOBAL)
        self._lib = ctypes.CDLL(str(SHIM))
        self._lib.axo_open.restype = ctypes.c_void_p
        self._lib.axo_open.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self._lib.axo_close.argtypes = [ctypes.c_void_p]
        self._lib.axo_score.restype = ctypes.c_int
        self._lib.axo_score.argtypes = [
            ctypes.c_void_p,
            np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),
            ctypes.c_int,
            np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),
        ]
        self._h = self._lib.axo_open(str(so).encode(), VARIANT[version])
        if not self._h:
            raise RuntimeError(f"could not load {so}")
        self.version = version
        self.name = f"AXOL1TL {version} (CMS emulator)"

    @property
    def n_trainable_parameters(self) -> int:
        """Weights carried by the emulator, counted from its own weights.h.

        These are not fitted *here* -- they are CMS's, trained on Zero Bias and
        frozen. But the model is emphatically not parameter-free, and reporting
        0 for it would misrepresent the comparison against the fly.
        """
        wh = MODELS[self.version].parent / "NN" / "weights.h"
        if not wh.exists():
            return 0
        import re as _re
        return sum(int(m.group(3))
                   for m in _re.finditer(r"(\w+)\s+(\w+)\[(\d+)\]", wh.read_text()))

    @property
    def fitted_here(self) -> bool:
        return False

    def score_hardware(self, hw: np.ndarray) -> np.ndarray:
        """Score pre-converted (n, 57) hardware inputs."""
        hw = np.ascontiguousarray(hw, dtype=np.float64)
        out = np.zeros(len(hw), dtype=np.float64)
        rc = self._lib.axo_score(ctypes.c_void_p(self._h), hw.ravel(), len(hw), out)
        if rc == -2:
            raise RuntimeError(
                f"AXOL1TL {self.version} emulator threw -- the read_result type "
                "variant is wrong for this version"
            )
        if rc != 0:
            raise RuntimeError(f"axo_score failed with {rc}")
        return out

    def score(self, events: np.ndarray) -> np.ndarray:
        """Score raw ADC2021 (n, 19, 4) events."""
        return self.score_hardware(to_hardware(events))

    def close(self) -> None:
        if getattr(self, "_h", None):
            self._lib.axo_close(ctypes.c_void_p(self._h))
            self._h = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
