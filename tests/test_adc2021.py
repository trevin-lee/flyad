"""ADC2021 trigger encoding and the real AXOL1TL emulator."""

from __future__ import annotations

import numpy as np
import pytest

_axo = pytest.importorskip("flyad.adc2021.axol1tl",
                           reason="external/AXOL1TL not built")


class TestTriggerLobe:
    """The AXOL1TL / ADC2021 representation: 19 objects, [19, 4] per event."""

    @staticmethod
    def _event(rng, n_e=2, n_mu=1, n_jet=3):
        ev = np.zeros((19, 4))
        ev[0] = [rng.uniform(10, 200), 0.0, rng.uniform(-np.pi, np.pi), 1]  # MET
        for k in range(min(n_e, 4)):
            ev[1 + k] = [rng.uniform(20, 150), rng.uniform(-2.4, 2.4),
                         rng.uniform(-np.pi, np.pi), 2]
        for k in range(min(n_mu, 4)):
            ev[5 + k] = [rng.uniform(20, 150), rng.uniform(-2.4, 2.4),
                         rng.uniform(-np.pi, np.pi), 3]
        for k in range(min(n_jet, 10)):
            ev[9 + k] = [rng.uniform(30, 400), rng.uniform(-4.5, 4.5),
                         rng.uniform(-np.pi, np.pi), 4]
        return ev

    def test_width_is_51(self):
        from flyad.adc2021.lobe import N_TRIGGER_GLOMERULI, encode_trigger_event
        rng = np.random.default_rng(0)
        assert encode_trigger_event(self._event(rng)).shape == (N_TRIGGER_GLOMERULI,)
        assert N_TRIGGER_GLOMERULI == 51

    def test_batch_matches_per_event(self):
        from flyad.adc2021.lobe import encode_trigger_batch, encode_trigger_event
        rng = np.random.default_rng(1)
        evs = np.stack([self._event(rng) for _ in range(200)])
        assert np.allclose(
            encode_trigger_batch(evs),
            np.stack([encode_trigger_event(e) for e in evs]),
        )

    def test_conserves_pt(self):
        """Every object must land in exactly one glomerulus."""
        from flyad.adc2021.lobe import encode_trigger_batch
        rng = np.random.default_rng(2)
        evs = np.stack([self._event(rng, 4, 4, 10) for _ in range(100)])
        enc = encode_trigger_batch(evs)
        assert np.allclose(enc[:, :-1].sum(1), evs[:, 1:, 0].sum(1))
        assert np.allclose(enc[:, -1], evs[:, 0, 0])

    def test_permutation_invariant_within_object_type(self):
        """Jet 3 and jet 7 are interchangeable; the encoding must agree."""
        from flyad.adc2021.lobe import encode_trigger_event
        rng = np.random.default_rng(3)
        ev = self._event(rng, 3, 3, 6)
        shuffled = ev.copy()
        shuffled[9:19] = ev[9:19][rng.permutation(10)]
        assert np.allclose(encode_trigger_event(ev), encode_trigger_event(shuffled))

    def test_invariant_to_global_phi_rotation(self):
        """Azimuth is measured relative to MET, so a global rotation is free."""
        from flyad.adc2021.lobe import encode_trigger_event
        rng = np.random.default_rng(4)
        ev = self._event(rng, 2, 2, 4)
        rot = ev.copy()
        real = rot[:, 0] > 0
        rot[real, 2] = (rot[real, 2] + 1.234 + np.pi) % (2 * np.pi) - np.pi
        assert np.allclose(encode_trigger_event(ev), encode_trigger_event(rot))

    def test_empty_event_is_all_zero(self):
        from flyad.adc2021.lobe import encode_trigger_event
        assert np.all(encode_trigger_event(np.zeros((19, 4))) == 0)


# The emulator tests need external/ built; skip cleanly when it is not.
_axo = pytest.importorskip("flyad.adc2021.axol1tl", reason="external/AXOL1TL not built")


class TestHardwareConversion:
    """ADC2021 physical units -> raw L1 hardware values."""

    def test_pt_uses_half_gev_counts(self):
        from flyad.adc2021.axol1tl import to_hardware
        ev = np.zeros((1, 19, 4))
        ev[0, 9] = [100.0, 0.0, 0.0, 4]  # a 100 GeV jet
        assert to_hardware(ev)[0, 27] == 200.0

    def test_muons_use_the_finer_scales(self):
        """Muon phi is a 576-bin scale, calorimeter phi a 144-bin one."""
        from flyad.adc2021.axol1tl import to_hardware
        ev = np.zeros((1, 19, 4))
        ev[0, 5] = [50.0, 0.0, np.pi, 3]   # muon at phi = pi
        ev[0, 9] = [50.0, 0.0, np.pi, 4]   # jet at phi = pi
        hw = to_hardware(ev)[0]
        assert hw[5 * 3 + 2] == 288.0      # muon slot 5, phi: 576 / 2
        assert hw[9 * 3 + 2] == 72.0       # jet slot 9, phi:  144 / 2

    def test_phi_wraps_to_positive_bins(self):
        from flyad.adc2021.axol1tl import to_hardware
        ev = np.zeros((1, 19, 4))
        ev[0, 9] = [50.0, 0.0, -np.pi / 2, 4]
        assert to_hardware(ev)[0, 29] == 108.0   # -pi/2 -> 3/4 of 144

    def test_padding_and_met_eta_stay_zero(self):
        from flyad.adc2021.axol1tl import to_hardware
        ev = np.zeros((1, 19, 4))
        ev[0, 0] = [40.0, 3.3, 1.0, 1]       # MET carries a bogus eta
        hw = to_hardware(ev)[0]
        assert hw[1] == 0.0                   # MET has no eta
        assert np.all(hw[3:15] == 0.0)        # empty slots stay empty


class TestAxol1tlEmulator:
    """The CMS emulator itself, driven through the ctypes shim."""

    REFERENCE_INPUT = np.array(
        [2047, 0, 0, 255, 20, 140, 10, 8, 77, 7, -4, 62, 5, 34, 73, 4, 0, 255,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 1023, 19, 139, 234, 5, 71, 49, 63, 55, 44,
         41, 7, 25, -86, 107, 13, 37, 121, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        dtype=np.float64,
    )

    def test_reproduces_the_upstream_reference_value(self):
        """Golden test: cms-hls4ml/test_AXOL1TL prints 2542.75 for this input.

        This pins the entire toolchain at once -- the vendored HLS arbitrary
        precision types, the emulator build, the ctypes shim, and the
        read_result type variant. If any of them drifts, this breaks.
        """
        from flyad.adc2021.axol1tl import Axol1tl
        a = Axol1tl("v5")
        try:
            got = a.score_hardware(np.ascontiguousarray(self.REFERENCE_INPUT[None, :]))
            assert got[0] == pytest.approx(2542.75, abs=1e-9)
        finally:
            a.close()

    def test_every_deployed_version_loads_and_scores(self):
        from flyad.adc2021.axol1tl import Axol1tl
        for v in ("v1", "v3", "v4", "v5", "v6"):
            a = Axol1tl(v)
            try:
                s = a.score_hardware(np.ascontiguousarray(self.REFERENCE_INPUT[None, :]))
                assert np.isfinite(s).all() and s[0] > 0
            finally:
                a.close()

    def test_scoring_is_deterministic(self):
        from flyad.adc2021.axol1tl import Axol1tl
        a = Axol1tl("v5")
        try:
            x = np.ascontiguousarray(np.tile(self.REFERENCE_INPUT, (16, 1)))
            s = a.score_hardware(x)
            assert np.all(s == s[0])
        finally:
            a.close()

    def test_reports_its_real_weight_count(self):
        """It is pretrained, not parameter-free -- 2,362 weights for v5."""
        from flyad.adc2021.axol1tl import Axol1tl
        a = Axol1tl("v5")
        try:
            assert a.n_trainable_parameters == 2362
        finally:
            a.close()


class TestRetrainedArchitectures:
    def test_v5_topology_matches_the_emulator_weight_count(self):
        """The retrained net must have exactly as many weights as the real one."""
        from flyad.adc2021.axol1tl_retrain import Axol1tlV5Retrained
        assert Axol1tlV5Retrained(epochs=0).n_trainable_parameters == 2362

    def test_scales_parse_from_the_emulator_source(self):
        from flyad.adc2021.axol1tl_retrain import load_scales
        for v in ("v3", "v5"):
            off, sh = load_scales(v)
            assert off.shape == (57,) and sh.shape == (57,)

    def test_fly_declares_its_state_honestly(self):
        """0 fitted parameters, but 2,000 data-dependent counts."""
        from flyad import FlyAnomalyDetector
        d = FlyAnomalyDetector(n_inputs=51, n_kc=2000)
        assert d.n_trainable_parameters == 0
        assert d.n_state_values == 2000
