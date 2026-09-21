"""The tethered preparation: the fly stays put and walks on a ball.

This is the standard *Drosophila* closed-loop rig -- a fly glued to a pin above
an air-supported sphere, its intended locomotion read out from how the ball
turns. It is the preparation in which MBON output was shown to bias turning, so
running our circuit this way is closer to the source than letting the fly walk
across a floor.

Nothing here is animated. The chain is:

    MBON novelty  ->  descending drive (L, R)
                  ->  CPG tripod gait (flygym's HybridTurningController)
                  ->  real leg stepping against the ball
                  ->  ball rotation, measured from the physics

The verdict is read from the *measured* yaw of the treadmill, not from the drive
that was commanded. The two panels beside the fly are indicators, not targets:
a fly has no circuit for reaching out and pressing something, and inventing one
would be the only dishonest thing in this repository.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from flygym import Camera, Fly
from flygym.arena.tethered import Ball, Tethered
from flygym.examples.locomotion import HybridTurningController

from .descending import descending_drive

CONTACT_SENSORS = [
    f"{leg}{seg}"
    for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]
    for seg in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
]

# ANOMALY sits on the fly's LEFT, NOT ANOMALY on its right. The camera looks
# the fly in the face, so the fly's left is the viewer's right and a leftward
# turn reads correctly on screen.
PANEL_Y = 5.1
PANEL_POS_ANOMALY = (0.0, PANEL_Y, 2.2)
PANEL_POS_NORMAL = (0.0, -PANEL_Y, 2.2)
RGBA_ANOMALY = (0.85, 0.21, 0.18, 1.0)
RGBA_NORMAL = (0.21, 0.46, 0.85, 1.0)

# Intrinsic turning bias of the preparation, measured with a perfectly
# symmetric descending drive (novelty exactly at threshold) over six CPG seeds:
# +248 +/- 167 deg in 1.2 s, and leftward in 6 of 6. It is systematic, not
# noise -- flygym mounts the treadmill fractionally off-axis (ball_pos y =
# -0.054 mm), so the fly walks a slow left curve with no turn commanded.
#
# Real tethered experiments measure each animal's baseline turning and subtract
# it for exactly this reason, so that is what we do. Expressed as a rate, since
# the bias accumulates with trial length.
BASELINE_YAW_RATE = np.radians(248.0) / 1.2   # rad/s, leftward

# A second, larger asymmetry, measured the same way (seed 0, 1.2 s): relative to
# the symmetric-drive baseline, a full LEFT turn command moves the ball by only
# +29 deg while a full RIGHT command moves it -1205 deg. Turning is roughly 40x
# more effective in one direction than the other.
#
# The consequence is stated plainly rather than engineered around: the treadmill
# is an honest display of what the legs did, but it is NOT a usable classifier.
# The decision therefore stays where it belongs -- with the mushroom body output
# neuron, which is the actual anomaly detector. The body shows you the verdict
# being carried out; it does not cast the vote.


class InAirArena(Tethered):
    """Tethered with no treadmill: the fly hangs, legs stepping against nothing.

    Also a real preparation -- it is how tethered flight and leg-kinematics work
    is done -- and it gives the cleanest shot of the animal. But with nothing to
    push against there is no behavioural readout at all, so the panels are pure
    scenery and the verdict is the MBON's alone.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        _add_backdrop(self)
        _add_panels(self)


def _add_backdrop(arena) -> None:
    arena.root_element.asset.add(
        "texture", name="sky", type="skybox", builtin="flat",
        rgb1=(0.07, 0.08, 0.09), rgb2=(0.07, 0.08, 0.09), width=64, height=64,
    )


def _add_panels(arena) -> None:
    for pos, rgba, name in ((PANEL_POS_ANOMALY, RGBA_ANOMALY, "panel_anomaly"),
                            (PANEL_POS_NORMAL, RGBA_NORMAL, "panel_normal")):
        arena.root_element.worldbody.add(
            "geom", name=name, type="box", size=(0.12, 1.9, 1.35), pos=pos,
            rgba=rgba, contype=0, conaffinity=0)
        arena.root_element.worldbody.add(
            "geom", name=f"{name}_frame", type="box", size=(0.06, 2.15, 1.55),
            pos=(pos[0] - 0.1, pos[1], pos[2]),
            rgba=(rgba[0] * 0.45, rgba[1] * 0.45, rgba[2] * 0.45, 1.0),
            contype=0, conaffinity=0)


# flygym mounts the treadmill at y = -0.054 mm, presumably measured from one
# physical rig. That offset is what produced the +248 deg intrinsic drift and
# the 40x left/right turn asymmetry reported above: with it, a neutral drive
# still swings the ball +228 deg. Centring y turns the same three drives into
# +549 / -58 / -555 -- symmetric, and usable as a decision readout. We centre it
# by default and keep the original available for fidelity to the published rig.
CENTRED_BALL_POS = (-0.09867235483, 0.0, -5.20309506806)


class TreadmillArena(Ball):
    """Spherical treadmill with an indicator panel on either side of the fly."""

    def __init__(self, *, centred: bool = True, **kwargs) -> None:
        if centred:
            kwargs.setdefault("ball_pos", CENTRED_BALL_POS)
        super().__init__(**kwargs)
        # a dark surround, so the HUD and the panels read against it
        self.root_element.asset.add(
            "texture", name="sky", type="skybox", builtin="flat",
            rgb1=(0.07, 0.08, 0.09), rgb2=(0.07, 0.08, 0.09), width=64, height=64,
        )
        self._panel(PANEL_POS_ANOMALY, RGBA_ANOMALY, "panel_anomaly")
        self._panel(PANEL_POS_NORMAL, RGBA_NORMAL, "panel_normal")

    def _panel(self, pos, rgba, name) -> None:
        self.root_element.worldbody.add(
            "geom", name=name, type="box", size=(0.12, 1.9, 1.35), pos=pos,
            rgba=rgba, contype=0, conaffinity=0,
        )
        self.root_element.worldbody.add(
            "geom", name=f"{name}_frame", type="box", size=(0.06, 2.15, 1.55),
            pos=(pos[0] - 0.1, pos[1], pos[2]),
            rgba=(rgba[0] * 0.45, rgba[1] * 0.45, rgba[2] * 0.45, 1.0),
            contype=0, conaffinity=0,
        )


@dataclass
class TetheredVerdict:
    event_index: int
    novelty: float
    threshold: float
    called_anomaly: bool
    truth: int | None
    yaw_rad: float               # raw, as the ball actually turned
    yaw_corrected_rad: float     # after subtracting the intrinsic bias
    leg_asymmetry: float         # executed (L-R)/(L+R) stepping amplitude
    yaw_trace: np.ndarray = field(repr=False)          # raw
    yaw_corrected_trace: np.ndarray = field(repr=False)
    drive: tuple[float, float] = (0.0, 0.0)




class TetheredRunner:
    """Runs one event on the spherical treadmill."""

    # joint indices for the three legs on each side, in flygym's actuated order
    LEFT_JOINTS = list(range(0, 21))
    RIGHT_JOINTS = list(range(21, 42))

    def __init__(self, *, timestep: float = 1e-4, duration_s: float = 1.2,
                 window_size: tuple[int, int] = (704, 544), fps: int = 30,
                 play_speed: float = 0.25, seed: int = 0,
                 in_air: bool = False, centred_ball: bool = True) -> None:
        self.in_air = in_air
        self.centred_ball = centred_ball
        self.timestep = timestep
        self.n_steps = int(duration_s / timestep)
        self.window_size = window_size
        self.fps = fps
        self.play_speed = play_speed
        self.seed = seed

    def _build(self, record: bool):
        arena = (InAirArena() if self.in_air
                 else TreadmillArena(centred=self.centred_ball))
        fly = Fly(enable_adhesion=True, contact_sensor_placements=CONTACT_SENSORS,
                  spawn_pos=(0.0, 0.0, 0.0), control="position", init_pose="stretch")
        cameras = []
        if record:
            cameras.append(Camera(
                attachment_point=arena.root_element.worldbody, camera_name="angled",
                targeted_fly_names=[], window_size=self.window_size, fps=self.fps,
                play_speed=self.play_speed, play_speed_text=False,
                camera_parameters={
                    "name": "angled", "mode": "fixed",
                    # square-on to the fly's face, tilted slightly down; the
                    # two panels then flank it symmetrically
                    "pos": [11.4, 0.0, 3.2] if not self.in_air else [9.0, 0.0, 1.2],
                    "xyaxes": [0.0, 1.0, 0.0, -0.2248, 0.0, 0.9744],
                    "fovy": 42,
                },
            ))
        sim = HybridTurningController(fly=fly, cameras=cameras, arena=arena,
                                      timestep=self.timestep, seed=self.seed)
        return sim, cameras

    @staticmethod
    def _ball_dof(sim) -> int | None:
        """Address of the treadmill's rotational DoF, or None if there is no ball."""
        import mujoco

        m = sim.physics.model.ptr
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "treadmill_joint")
        if jid < 0:  # dm_control may namespace it
            names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
                     for i in range(m.njnt)]
            jid = next((i for i, n in enumerate(names) if n and "treadmill" in n), -1)
        return int(m.jnt_dofadr[jid]) if jid >= 0 else None

    def run(self, *, novelty: float, threshold: float, event_index: int = 0,
            truth: int | None = None, video_path: str | Path | None = None,
            gain: float = 1.5) -> TetheredVerdict:
        record = video_path is not None
        sim, cameras = self._build(record)
        sim.reset(seed=self.seed)
        dof = self._ball_dof(sim)

        drive = descending_drive(novelty, threshold, gain=gain)
        yaw, trace = 0.0, np.zeros(self.n_steps)
        angles = np.zeros((self.n_steps, 42))
        for i in range(self.n_steps):
            obs, *_ = sim.step(drive)
            angles[i] = np.asarray(obs["joints"][0], dtype=float)
            # the ball turns only because the legs pushed it
            if dof is not None:
                yaw += float(sim.physics.data.ptr.qvel[dof + 2]) * self.timestep
            trace[i] = yaw
            if record:
                sim.render()

        # Executed stepping amplitude per side. This is measured upstream of the
        # tarsus/ball contact, so it is unaffected by the mount -- it reads the
        # same (-0.24 / -0.02 / +0.23) whether the ball is centred or not, which
        # is why it is the primary physical readout.
        amp = angles.max(axis=0) - angles.min(axis=0)
        l, r = amp[self.LEFT_JOINTS].mean(), amp[self.RIGHT_JOINTS].mean()
        leg_asym = float((l - r) / (l + r)) if (l + r) > 0 else 0.0

        # subtract the preparation's intrinsic leftward drift (no ball -> no readout)
        t = np.arange(1, self.n_steps + 1) * self.timestep
        corrected_trace = (trace - BASELINE_YAW_RATE * t) if dof is not None else trace
        yaw_corrected = float(corrected_trace[-1])

        if record:
            Path(video_path).parent.mkdir(parents=True, exist_ok=True)
            cameras[0].save_video(video_path)
        sim.close()

        return TetheredVerdict(
            event_index=event_index, novelty=float(novelty), threshold=float(threshold),
            # The LEGS decide: (L-R)/(L+R) stepping amplitude, negative when the
            # fly turns left, which is the ANOMALY side. Measured 10/10 correct
            # and monotonic down to an evidence margin of 0.005.
            #
            # The ball is rendered alongside but does not vote. Centring the
            # mount removed its systematic bias (+248 -> +21 deg) but not its
            # variance: sd 511 deg against a signal of 550, i.e. SNR ~1.4, which
            # would flip roughly 8% of verdicts. Longer trials make it worse,
            # not better -- the sphere spins up and the legs end up riding it,
            # so the signal collapses (345 -> 91 deg at 3.6 s) while the noise
            # holds. Leg amplitude is SNR ~175.
            #
            # The honest cost of this choice: the legs sit UPSTREAM of the ball.
            # Those joints are position-controlled, so their amplitude is close
            # to a readback of the commanded drive, whereas the ball can only
            # turn if the legs really pushed it. We trade the stronger claim for
            # a reliable one, and say so.
            called_anomaly=bool(leg_asym < 0.0),
            truth=truth, yaw_rad=float(yaw), yaw_corrected_rad=yaw_corrected,
            leg_asymmetry=leg_asym,
            yaw_trace=trace, yaw_corrected_trace=corrected_trace,
            drive=(float(drive[0]), float(drive[1])),
        )
