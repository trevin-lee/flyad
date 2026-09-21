"""Loading LHC Olympics 2020 events and clustering them into jets.

The R&D dataset stores one event per row: 700 hadrons x (pT, eta, phi),
zero-padded, followed by a truth label (0 = QCD background, 1 = Z' signal).
The label exists only for evaluation; the detector never sees it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import awkward as ak
import fastjet
import numpy as np

from .lobe import encode_event

N_HADRONS = 700
JET_RADIUS = 1.0
MIN_JET_PT = 20.0  # GeV


def n_events(path: str | Path) -> int:
    """Number of events in an LHCO HDF5 file, without loading it."""
    import tables

    with tables.open_file(str(path)) as f:
        return int(f.root.df.block0_values.shape[0])


def read_chunk(path: str | Path, start: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
    """Read rows [start, stop) as (hadrons, labels).

    `hadrons` has shape (n_events, N_HADRONS, 3) holding (pT, eta, phi).

    Read through PyTables rather than pandas: the R&D file stores integer
    column labels, which pandas' HDF reader rejects, and its Blosc filter is
    not available to bare h5py. PyTables bundles Blosc and lets us slice the
    underlying array directly, which is also faster than materialising a
    2101-column DataFrame per chunk.
    """
    import tables

    with tables.open_file(str(path)) as f:
        arr = f.root.df.block0_values[start:stop]

    labels = arr[:, 2100] if arr.shape[1] > 2100 else np.zeros(len(arr))
    hadrons = arr[:, :2100].reshape(-1, N_HADRONS, 3)
    return hadrons, labels


def cluster_chunk(hadrons: np.ndarray) -> ak.Array:
    """Run anti-kT R=1.0 over a chunk of events.

    Returns an awkward array of jets with their constituents attached. Hadrons
    are massless, as the dataset specifies.
    """
    pt, eta, phi = hadrons[..., 0], hadrons[..., 1], hadrons[..., 2]
    real = pt > 0  # strip the zero padding

    builder = {
        "px": ak.Array([p[m] * np.cos(f[m]) for p, f, m in zip(pt, phi, real)]),
        "py": ak.Array([p[m] * np.sin(f[m]) for p, f, m in zip(pt, phi, real)]),
        "pz": ak.Array([p[m] * np.sinh(e[m]) for p, e, m in zip(pt, eta, real)]),
        "E": ak.Array([p[m] * np.cosh(e[m]) for p, e, m in zip(pt, eta, real)]),
    }
    particles = ak.zip(builder)

    jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, JET_RADIUS)
    cs = fastjet.ClusterSequence(particles, jetdef)
    return cs.inclusive_jets(min_pt=MIN_JET_PT), cs.constituents(min_pt=MIN_JET_PT)


def _kinematics(px, py, pz, E):
    pt = np.hypot(px, py)
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = np.arcsinh(np.where(pt > 0, pz / pt, 0.0))
    phi = np.arctan2(py, px)
    return pt, eta, phi


def encode_chunk(hadrons: np.ndarray) -> np.ndarray:
    """Cluster a chunk and encode each event as 50 projection-neuron activities."""
    jets, consts = cluster_chunk(hadrons)
    out = np.zeros((len(hadrons), 50), dtype=np.float64)

    for i in range(len(hadrons)):
        ev_jets, ev_consts = jets[i], consts[i]
        jpt, jeta, jphi = _kinematics(
            np.asarray(ev_jets.px), np.asarray(ev_jets.py),
            np.asarray(ev_jets.pz), np.asarray(ev_jets.E),
        )
        order = np.argsort(-jpt)[:2]  # two leading jets

        jet_dicts = []
        for k in order:
            c = ev_consts[k]
            cpt, ceta, cphi = _kinematics(
                np.asarray(c.px), np.asarray(c.py),
                np.asarray(c.pz), np.asarray(c.E),
            )
            jet_dicts.append(
                {"eta": jeta[k], "phi": jphi[k],
                 "c_pt": cpt, "c_eta": ceta, "c_phi": cphi}
            )
        out[i] = encode_event(jet_dicts)
    return out


def iter_encoded(
    path: str | Path, *, chunk_size: int = 5000, limit: int | None = None
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Stream (pn_activity, labels) over an LHCO file in chunks."""
    total = n_events(path)
    if limit is not None:
        total = min(total, limit)
    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        hadrons, labels = read_chunk(path, start, stop)
        yield encode_chunk(hadrons), labels
