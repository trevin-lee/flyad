"""How the circuit's output becomes movement."""

from __future__ import annotations

import numpy as np
import pytest

class TestTetheredPreparation:
    """The spherical treadmill: fly fixed, legs real, ball rotation measured."""

    def test_novel_steers_one_way_and_familiar_the_other(self):
        from flyad.embodied_tethered import descending_drive
        novel = descending_drive(0.9, 0.4)
        familiar = descending_drive(0.1, 0.4)
        # positive turn = left = ANOMALY side, which means MORE drive on the right
        assert novel[1] > novel[0]
        assert familiar[0] > familiar[1]

    def test_drive_is_symmetric_about_the_threshold(self):
        from flyad.embodied_tethered import descending_drive
        a = descending_drive(0.4 + 0.02, 0.4, gain=1.0)
        b = descending_drive(0.4 - 0.02, 0.4, gain=1.0)
        assert a[0] == pytest.approx(b[1])
        assert a[1] == pytest.approx(b[0])

    def test_confidence_scales_the_turn(self):
        """A marginal event should turn weakly, a confident one decisively."""
        from flyad.embodied_tethered import descending_drive
        marginal = descending_drive(0.41, 0.4, gain=4.0)
        confident = descending_drive(0.90, 0.4, gain=4.0)
        assert abs(confident[1] - confident[0]) > abs(marginal[1] - marginal[0])

    def test_drive_stays_inside_the_controller_action_space(self):
        from flyad.embodied_tethered import descending_drive
        for nov in (0.0, 0.25, 0.5, 0.75, 1.0):
            d = descending_drive(nov, 0.5, gain=50.0)   # absurd gain
            assert np.all(d >= -0.5) and np.all(d <= 1.5)

    def test_arena_has_a_free_ball_joint_and_both_panels(self):
        """The treadmill must be free to rotate, or nothing is being measured."""
        from flyad.embodied_tethered import TreadmillArena
        arena = TreadmillArena()
        joints = [j.name for j in arena.root_element.find_all("joint")]
        assert any(j and "treadmill" in j for j in joints)
        geoms = [g.name for g in arena.root_element.find_all("geom")]
        assert "panel_anomaly" in geoms and "panel_normal" in geoms

    def test_panels_cannot_collide_with_the_fly(self):
        """They are indicators, not targets -- a fly cannot reach for a button."""
        from flyad.embodied_tethered import TreadmillArena
        arena = TreadmillArena()
        for g in arena.root_element.find_all("geom"):
            if g.name and g.name.startswith("panel"):
                assert g.contype == 0 and g.conaffinity == 0
