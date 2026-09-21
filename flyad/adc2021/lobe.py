"""Antennal lobe for trigger-level events (ADC2021 / AXOL1TL representation).

AXOL1TL sees what the CMS Global Trigger sees: at most 10 jets, 4 muons,
4 electrons/photons, and one MET, each as a (pT, eta, phi) triple. That is the
same kind of object a fly's antennal lobe handles -- a small, fixed set of
channels, each reporting how much of something arrived from where.

We give the fly 51 glomeruli. Each is one (object type x eta band x azimuth
band) cell, activated by the summed pT of the objects falling in it, plus one
channel for MET itself. Binning makes the encoding permutation invariant within
an object type, which matters because jet 3 and jet 4 are interchangeable.

The eta boundaries are the detector's own, not round numbers: the barrel/
endcap transition for leptons and the tracker/HF boundaries for jets.
"""

from __future__ import annotations

import numpy as np

# slot layout of the [19, 4] ADC2021 array
MET_SLOT = 0
EGAMMA_SLOTS = slice(1, 5)
MUON_SLOTS = slice(5, 9)
JET_SLOTS = slice(9, 19)

# real CMS-ish boundaries: leptons stop at the endcap edge, jets run to the HF
LEPTON_ETA_EDGES = np.array([0.0, 0.8, 1.44, 2.5])       # barrel / transition / endcap
JET_ETA_EDGES = np.array([0.0, 1.3, 2.5, 3.0, 5.0])      # central / inner-fwd / gap / HF
N_DPHI = 5                                                # |dphi| to MET, 0..pi

N_EGAMMA = (len(LEPTON_ETA_EDGES) - 1) * N_DPHI          # 15
N_MUON = (len(LEPTON_ETA_EDGES) - 1) * N_DPHI            # 15
N_JET = (len(JET_ETA_EDGES) - 1) * N_DPHI                # 20
N_TRIGGER_GLOMERULI = N_EGAMMA + N_MUON + N_JET + 1      # 51


def _abs_dphi(phi: np.ndarray, ref: float) -> np.ndarray:
    """|phi - ref| folded into [0, pi]."""
    d = np.abs((phi - ref + np.pi) % (2 * np.pi) - np.pi)
    return d


def _block(pt, eta, phi, met_phi, eta_edges) -> np.ndarray:
    """pT-weighted (|eta| band x |dphi to MET| band) occupancy for one object type."""
    n_eta = len(eta_edges) - 1
    grid = np.zeros((n_eta, N_DPHI))
    real = pt > 0
    if not real.any():
        return grid.ravel()
    pt, eta, phi = pt[real], eta[real], phi[real]

    ie = np.digitize(np.abs(eta), eta_edges) - 1
    ie = np.clip(ie, 0, n_eta - 1)
    idphi = np.minimum(
        (_abs_dphi(phi, met_phi) / np.pi * N_DPHI).astype(int), N_DPHI - 1
    )
    np.add.at(grid, (ie, idphi), pt)
    return grid.ravel()


def encode_trigger_event(ev: np.ndarray) -> np.ndarray:
    """Encode one [19, 4] trigger event as 51 raw glomerular activations.

    Divisive normalisation is applied by the caller (batched), exactly as for
    the dijet encoding.
    """
    pt, eta, phi = ev[:, 0], ev[:, 1], ev[:, 2]
    met_pt, met_phi = pt[MET_SLOT], phi[MET_SLOT]

    return np.concatenate([
        _block(pt[EGAMMA_SLOTS], eta[EGAMMA_SLOTS], phi[EGAMMA_SLOTS],
               met_phi, LEPTON_ETA_EDGES),
        _block(pt[MUON_SLOTS], eta[MUON_SLOTS], phi[MUON_SLOTS],
               met_phi, LEPTON_ETA_EDGES),
        _block(pt[JET_SLOTS], eta[JET_SLOTS], phi[JET_SLOTS],
               met_phi, JET_ETA_EDGES),
        np.array([met_pt]),
    ])


def encode_trigger_batch(events: np.ndarray) -> np.ndarray:
    """Vectorised encoding of an (n, 19, 4) array -> (n, 51).

    Written as explicit scatter-adds rather than a Python loop because the
    background set has millions of events.
    """
    n = len(events)
    pt, eta, phi = events[..., 0], events[..., 1], events[..., 2]
    met_phi = phi[:, MET_SLOT][:, None]
    out = np.zeros((n, N_TRIGGER_GLOMERULI))

    off = 0
    for slots, edges in (
        (EGAMMA_SLOTS, LEPTON_ETA_EDGES),
        (MUON_SLOTS, LEPTON_ETA_EDGES),
        (JET_SLOTS, JET_ETA_EDGES),
    ):
        n_eta = len(edges) - 1
        p, e, f = pt[:, slots], eta[:, slots], phi[:, slots]
        real = p > 0
        ie = np.clip(np.digitize(np.abs(e), edges) - 1, 0, n_eta - 1)
        idp = np.minimum((_abs_dphi(f, met_phi) / np.pi * N_DPHI).astype(int), N_DPHI - 1)
        flat = ie * N_DPHI + idp                       # cell index within the block
        rows = np.repeat(np.arange(n)[:, None], p.shape[1], axis=1)
        np.add.at(out, (rows[real], off + flat[real]), p[real])
        off += n_eta * N_DPHI

    out[:, off] = pt[:, MET_SLOT]
    return out
