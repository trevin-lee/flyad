"""LHC Olympics encoding: turn a dijet event into a 50-glomerulus odour.

The fly has ~50 olfactory receptor classes. Each maps to one glomerulus in the
antennal lobe, and the projection neuron (PN) leaving that glomerulus carries a
divisively normalised version of its receptor's drive.

We give the fly 50 glomeruli of jet substructure: the two leading jets, each
binned into a 25-cell polar grid around its own axis.
"""

from __future__ import annotations

import numpy as np

from ..antennal_lobe import divisive_normalisation

N_RINGS = 5
N_SECTORS = 5
N_GLOMERULI_PER_JET = N_RINGS * N_SECTORS
N_GLOMERULI = 2 * N_GLOMERULI_PER_JET

# Ring edges in dR from the jet axis. Jets are core-dominated, so the rings are
# spaced finely near the axis and coarsely at the periphery.
RING_EDGES = np.array([0.0, 0.10, 0.20, 0.35, 0.55, 1.0])


def _wrap_phi(dphi: np.ndarray) -> np.ndarray:
    """Wrap an azimuthal difference into (-pi, pi]."""
    return (dphi + np.pi) % (2 * np.pi) - np.pi


def principal_axis_angle(deta: np.ndarray, dphi: np.ndarray, pt: np.ndarray) -> float:
    """pT-weighted principal axis of the constituents, in radians.

    Rotating by -this angle puts the jet's dominant substructure direction on a
    fixed axis, so the azimuthal sectors mean the same thing on every jet.
    """
    if pt.sum() <= 0 or len(pt) < 2:
        return 0.0
    w = pt / pt.sum()
    cxx = np.sum(w * deta * deta)
    cyy = np.sum(w * dphi * dphi)
    cxy = np.sum(w * deta * dphi)
    # principal eigenvector angle of [[cxx, cxy], [cxy, cyy]]
    return 0.5 * np.arctan2(2.0 * cxy, cxx - cyy)


def jet_polar_grid(
    deta: np.ndarray, dphi: np.ndarray, pt: np.ndarray
) -> np.ndarray:
    """Bin one jet's constituents into a (N_RINGS x N_SECTORS) pT grid.

    Returns a flat array of length N_GLOMERULI_PER_JET.
    """
    grid = np.zeros((N_RINGS, N_SECTORS), dtype=np.float64)
    if len(pt) == 0:
        return grid.ravel()

    # Rotate into the principal frame so sectors are comparable across jets.
    theta0 = principal_axis_angle(deta, dphi, pt)
    c, s = np.cos(-theta0), np.sin(-theta0)
    x = c * deta - s * dphi
    y = s * deta + c * dphi

    r = np.hypot(x, y)
    # Reflect so that the pT-weighted mass sits on a consistent side: removes
    # the residual 180-degree ambiguity of the principal axis.
    if np.sum(pt * y) < 0:
        y = -y
    ang = np.arctan2(y, x)

    ring = np.digitize(r, RING_EDGES) - 1
    sector = np.floor((ang + np.pi) / (2 * np.pi) * N_SECTORS).astype(int)
    sector = np.clip(sector, 0, N_SECTORS - 1)

    keep = (ring >= 0) & (ring < N_RINGS)
    np.add.at(grid, (ring[keep], sector[keep]), pt[keep])
    return grid.ravel()


def encode_event(
    jets: list[dict], *, normalise: bool = True
) -> np.ndarray:
    """Encode the two leading jets of an event into a 50-vector of PN activity.

    `jets` is a list of dicts with keys `eta`, `phi` (jet axis) and constituent
    arrays `c_pt`, `c_eta`, `c_phi`. Jets beyond the first two are ignored; if
    only one jet is present the second block is left at zero.
    """
    blocks = []
    for i in range(2):
        if i < len(jets):
            j = jets[i]
            deta = j["c_eta"] - j["eta"]
            dphi = _wrap_phi(j["c_phi"] - j["phi"])
            blocks.append(jet_polar_grid(deta, dphi, j["c_pt"]))
        else:
            blocks.append(np.zeros(N_GLOMERULI_PER_JET))
    orn = np.concatenate(blocks)
    return divisive_normalisation(orn) if normalise else orn
