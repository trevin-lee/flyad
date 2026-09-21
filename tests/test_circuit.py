"""The circuit itself: FlyHash, the mushroom body, the assembled detector."""

from __future__ import annotations

import numpy as np
import pytest

from flyad import FlyAnomalyDetector, FlyHash, MushroomBody
from flyad.antennal_lobe import divisive_normalisation


class TestFlyHash:
    def test_tag_is_sparse_and_the_right_size(self):
        h = FlyHash(50, n_kc=2000, wta_fraction=0.05, seed=0)
        tags = h.tag(np.random.default_rng(0).random((7, 50)))
        assert tags.shape == (7, 100)
        assert all(len(np.unique(t)) == 100 for t in tags)

    def test_nothing_is_trainable(self):
        assert FlyHash(50, seed=0).n_trainable_parameters == 0

    def test_is_deterministic_given_a_seed(self):
        x = np.random.default_rng(0).random((5, 50))
        assert np.array_equal(FlyHash(50, seed=3).tag(x), FlyHash(50, seed=3).tag(x))

    def test_locality_sensitivity(self):
        """The defining FlyHash property: near inputs share more Kenyon cells."""
        rng = np.random.default_rng(5)
        h = FlyHash(50, n_kc=2000, seed=0)
        x = rng.random(50)
        near = x + rng.normal(0, 0.01, 50)
        far = rng.random(50)
        t = [set(h.tag(v)[0]) for v in (x, near, far)]
        assert len(t[0] & t[1]) > len(t[0] & t[2])

    def test_blocking_does_not_change_the_result(self):
        """The memory fix must be numerically inert."""
        h = FlyHash(50, n_kc=500, seed=0)
        x = np.random.default_rng(6).random((400, 50))
        big = h.tag(x)
        h._block_rows = lambda *a, **k: 7  # force many small blocks
        assert np.array_equal(np.sort(big, axis=1), np.sort(h.tag(x), axis=1))


class TestMushroomBody:
    def test_familiar_scores_below_novel(self):
        rng = np.random.default_rng(7)
        h = FlyHash(50, n_kc=2000, seed=0)
        mb = MushroomBody(2000)
        familiar = rng.random((500, 50))
        mb.experience(h.tag(familiar))
        novel = rng.random((500, 50)) * 4.0
        assert mb.novelty(h.tag(familiar)).mean() < mb.novelty(h.tag(novel)).mean()

    def test_nothing_is_trainable(self):
        assert MushroomBody(2000).n_trainable_parameters == 0

    def test_weights_stay_in_unit_interval(self):
        h = FlyHash(50, n_kc=200, seed=0)
        mb = MushroomBody(200)
        mb.experience(h.tag(np.random.default_rng(8).random((300, 50))))
        w = mb.weights()
        assert np.all((w > 0) & (w <= 1))


class TestDetector:
    def test_end_to_end_has_no_parameters(self):
        assert FlyAnomalyDetector(n_inputs=50).n_trainable_parameters == 0

    def test_separates_a_shifted_population(self):
        rng = np.random.default_rng(9)
        det = FlyAnomalyDetector(n_inputs=50, n_kc=2000, seed=0)
        bkg = divisive_normalisation(rng.exponential(1.0, (4000, 50)))
        det.experience(bkg)
        sig = divisive_normalisation(rng.exponential(1.0, (1000, 50)) ** 3)
        assert det.score(sig).mean() > det.score(bkg[:1000]).mean()

    def test_decide_respects_the_threshold(self):
        rng = np.random.default_rng(10)
        det = FlyAnomalyDetector(n_inputs=50, seed=0)
        x = divisive_normalisation(rng.exponential(1.0, (200, 50)))
        det.experience(x)
        s = det.score(x)
        assert np.array_equal(det.decide(x, float(np.median(s))), s >= np.median(s))
