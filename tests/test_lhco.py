"""LHC Olympics dijet encoding."""

from __future__ import annotations

import numpy as np
import pytest

from flyad.antennal_lobe import divisive_normalisation
from flyad.lhco.lobe import N_GLOMERULI, encode_event, jet_polar_grid


def _jet(rng, n=60, spread=0.08, centre=(0.0, 0.0), scale=1.0):
    r = rng.exponential(spread, n)
    th = rng.uniform(0, 2 * np.pi, n)
    w = rng.exponential(1.0, n)
    return {
        "eta": centre[0], "phi": centre[1],
        "c_pt": w / w.sum() * 1000.0 * scale,
        "c_eta": centre[0] + r * np.cos(th),
        "c_phi": centre[1] + r * np.sin(th),
    }


class TestAntennalLobe:
    def test_divisive_normalisation_is_scale_invariant(self):
        """The whole reason this step earns its place: pT scale drops out."""
        rng = np.random.default_rng(0)
        x = rng.exponential(1.0, 50)
        assert np.allclose(
            divisive_normalisation(x), divisive_normalisation(x * 137.0), atol=1e-12
        )

    def test_encoding_is_permutation_invariant(self):
        """Jet constituents are an unordered set; binning must not care."""
        rng = np.random.default_rng(1)
        j = _jet(rng)
        a = encode_event([j, j])
        perm = rng.permutation(len(j["c_pt"]))
        jp = {**j, "c_pt": j["c_pt"][perm], "c_eta": j["c_eta"][perm],
              "c_phi": j["c_phi"][perm]}
        assert np.allclose(a, encode_event([jp, jp]))

    def test_encoding_has_expected_width(self):
        rng = np.random.default_rng(2)
        assert encode_event([_jet(rng), _jet(rng)]).shape == (N_GLOMERULI,)

    def test_missing_second_jet_is_zero_padded(self):
        rng = np.random.default_rng(3)
        out = encode_event([_jet(rng)], normalise=False)
        assert out.shape == (N_GLOMERULI,)
        assert np.all(out[25:] == 0)

    def test_polar_grid_separates_one_and_two_prong(self):
        """The physics the detector actually runs on."""
        rng = np.random.default_rng(4)
        one = _jet(rng, n=200, spread=0.05)
        g1 = jet_polar_grid(one["c_eta"] - one["eta"],
                            one["c_phi"] - one["phi"], one["c_pt"]).reshape(5, 5)
        two_eta = np.concatenate([one["c_eta"][:100] - 0.25, one["c_eta"][100:] + 0.25])
        two_phi = np.concatenate([one["c_phi"][:100] - 0.25, one["c_phi"][100:] + 0.25])
        g2 = jet_polar_grid(two_eta - one["eta"], two_phi - one["phi"],
                            one["c_pt"]).reshape(5, 5)
        core1 = g1[0].sum() / g1.sum()
        core2 = g2[0].sum() / g2.sum()
        assert core1 > core2, "one-prong must be more core-concentrated"
