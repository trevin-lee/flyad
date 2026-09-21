"""The real thing: PN -> Kenyon cell connectivity, measured, from FlyWire.

Everything else in this repository *models* the circuit. This module loads it.

Sources (both public, no login):
  * connectivity  -- FlyWire v783 proofread connections, Zenodo 10676866
  * annotations   -- flyconnectome/flywire_annotations, Supplemental file 1

The point of having this is a falsifiable test. Caron et al. (2013) reported that
projection-neuron to Kenyon-cell wiring is random with respect to glomerular
identity, and FlyHash is built on that claim. If it holds, substituting the real
matrix for a degree-matched random one should change downstream performance not
at all. If it does change things, the real wiring carries structure the random
model discards.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "connectome"


@dataclass(frozen=True)
class Dataset:
    """One connectome release, and how to read it.

    Both ship Apache Arrow `.feather` connectivity, but nothing else lines up:
    column names, neuron-id spaces and annotation formats all differ. What the
    community did standardise is the biological vocabulary -- both use `ALPN`
    and `Kenyon_Cell` as class labels and identical glomerulus names -- so the
    analysis is portable even though the file readers are not.
    """

    key: str
    label: str
    connections: Path
    annotations: Path
    pre: str
    post: str
    weight: str
    ann_id: str
    ann_class: str
    ann_type: str
    ann_side: str
    side_values: dict            # canonical side -> this dataset's spelling


# FlyWire additionally carries per-connection neurotransmitter predictions
# (gaba/ach/glut/oct/ser/da) and the neuropil each synapse sits in. Those would
# let synaptic sign be read from data rather than assumed, which is currently
# the weakest assumption in the model -- see the README.
FLYWIRE = Dataset(
    key="flywire", label="FlyWire v783 (female brain, Nature 2024)",
    connections=DATA / "proofread_connections_783.feather",
    annotations=DATA / "flywire_neuron_annotations.tsv",
    pre="pre_pt_root_id", post="post_pt_root_id", weight="syn_count",
    ann_id="root_id", ann_class="cell_class", ann_type="cell_type",
    ann_side="side", side_values={"right": "right", "left": "left"},
)

MALE_CNS = Dataset(
    key="malecns", label="Male CNS v1.0 (Janelia + Google, Sept 2026)",
    connections=DATA / "malecns_connectome_weights.feather",
    annotations=DATA / "malecns_annotations.feather",
    pre="body_pre", post="body_post", weight="weight",
    ann_id="bodyId", ann_class="class", ann_type="type",
    ann_side="somaSide", side_values={"right": "R", "left": "L"},
)

DATASETS = {d.key: d for d in (FLYWIRE, MALE_CNS)}

# uniglomerular olfactory PNs are named <glomerulus>_<lineage>PN
_UNIGLOM = r"^([A-Za-z0-9]+?)_(?:ad|l|lv|il|v|m|t)+PN"


@dataclass
class MushroomBodyWiring:
    """Measured PN -> KC connectivity for one hemisphere."""

    weights: np.ndarray          # (n_kc, n_glomeruli), synapse counts
    glomeruli: list[str]
    kc_ids: np.ndarray
    kc_types: np.ndarray
    side: str
    dataset: str = "flywire"

    @property
    def n_kc(self) -> int:
        return self.weights.shape[0]

    @property
    def n_glomeruli(self) -> int:
        return self.weights.shape[1]

    def claws_per_kc(self) -> np.ndarray:
        """How many distinct glomeruli each Kenyon cell listens to."""
        return (self.weights > 0).sum(axis=1)

    def summary(self) -> dict:
        c = self.claws_per_kc()
        return {
            "dataset": self.dataset,
            "side": self.side,
            "n_kc": int(self.n_kc),
            "n_glomeruli": int(self.n_glomeruli),
            "total_synapses": int(self.weights.sum()),
            "mean_claws_per_kc": float(c.mean()),
            "median_claws_per_kc": float(np.median(c)),
            "kc_with_no_olfactory_input": int((c == 0).sum()),
        }


def load_annotations(ds: Dataset = FLYWIRE) -> pd.DataFrame:
    if not ds.annotations.exists():
        raise FileNotFoundError(f"{ds.annotations} missing -- see the README")
    if ds.annotations.suffix == ".feather":
        return pd.read_feather(ds.annotations)
    return pd.read_csv(ds.annotations, sep="\t", low_memory=False)


def load_wiring(dataset: str | Dataset = "malecns", side: str = "right", *,
                min_synapses: int = 1, n_glomeruli: int | None = None
                ) -> MushroomBodyWiring:
    """Build the measured (n_kc x n_glomeruli) matrix for one hemisphere.

    Kenyon cells are taken as `cell_class == 'Kenyon_Cell'` where available and
    otherwise by the `KC...` cell_type prefix; projection neurons as
    `cell_class == 'ALPN'` with a parseable glomerulus in their cell type.
    Multiglomerular and thermo/hygrosensory PNs are excluded, so the columns are
    exactly the uniglomerular olfactory channels FlyHash assumes.
    """
    ds = DATASETS[dataset] if isinstance(dataset, str) else dataset
    if not ds.connections.exists():
        raise FileNotFoundError(f"{ds.connections} missing -- see the README")

    ann = load_annotations(ds)
    ann = ann[ann[ds.ann_side] == ds.side_values[side]]

    pn = ann[ann[ds.ann_class] == "ALPN"].copy()
    pn["glom"] = pn[ds.ann_type].astype(str).str.extract(_UNIGLOM)[0]
    pn = pn.dropna(subset=["glom"])

    ct = ann[ds.ann_type].astype(str)
    kc = ann[(ann[ds.ann_class] == "Kenyon_Cell") | ct.str.startswith("KC")]

    glomeruli = sorted(pn["glom"].unique())
    gidx = {g: i for i, g in enumerate(glomeruli)}
    kc_ids = kc[ds.ann_id].to_numpy()
    kidx = {r: i for i, r in enumerate(kc_ids)}

    # the edge tables are large (152M rows for the male CNS); read only the
    # three columns we need and filter immediately
    conn = pd.read_feather(ds.connections, columns=[ds.pre, ds.post, ds.weight])
    pre_col, post_col, syn_col = ds.pre, ds.post, ds.weight

    pn_to_glom = dict(zip(pn[ds.ann_id], pn["glom"]))
    m = conn[conn[pre_col].isin(pn_to_glom) & conn[post_col].isin(kidx)]
    m = m[m[syn_col] >= min_synapses]

    W = np.zeros((len(kc_ids), len(glomeruli)), dtype=np.float64)
    rows = m[post_col].map(kidx).to_numpy()
    cols = m[pre_col].map(pn_to_glom).map(gidx).to_numpy()
    np.add.at(W, (rows, cols), m[syn_col].to_numpy())

    if n_glomeruli is not None and n_glomeruli < W.shape[1]:
        # Our input channels are jet-image patches, not odours, so no patch
        # corresponds to any particular glomerulus. When the encoding is
        # narrower than the animal's 55 channels we keep the best-innervated
        # ones -- an arbitrary but principled choice, and stated as such.
        keep = np.argsort(-W.sum(axis=0))[:n_glomeruli]
        keep.sort()
        W = W[:, keep]
        glomeruli = [glomeruli[i] for i in keep]

    return MushroomBodyWiring(
        weights=W, glomeruli=glomeruli, kc_ids=kc_ids,
        kc_types=kc[ds.ann_type].to_numpy(), side=side, dataset=ds.key,
    )


def degree_matched_shuffle(W: np.ndarray, *, seed: int = 0) -> np.ndarray:
    """A random matrix with each Kenyon cell's claw count preserved.

    This is the control the comparison needs. Comparing the real wiring against
    a matrix with a different number of connections would confound "is it
    random?" with "how many inputs does each cell get?" -- so the shuffle keeps
    every KC's in-degree and its synapse weights, and only reassigns which
    glomeruli they come from.
    """
    rng = np.random.default_rng(seed)
    out = np.zeros_like(W)
    n_glom = W.shape[1]
    for i, row in enumerate(W):
        nz = np.flatnonzero(row)
        if nz.size == 0:
            continue
        out[i, rng.choice(n_glom, size=nz.size, replace=False)] = row[nz]
    return out
