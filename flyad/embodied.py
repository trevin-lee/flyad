"""The behavioural readout: a fly walks to a button.

In the real animal, mushroom body output neurons bias the fly's turning via
descending neurons -- this is how a learned odour valence becomes an approach
or avoidance. Here the MBON's novelty score picks a target, and a proportional
controller on heading error supplies the descending drive that steers there.

The drive is the only part of this file that is engineering rather than
biology: the real MBON->DAN->descending pathway is not a P controller. It is
called out explicitly rather than dressed up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from flygym import Camera, Fly
from flygym.arena import FlatTerrain
from flygym.examples.locomotion import HybridTurningController

CONTACT_SENSORS = [
    f"{leg}{seg}"
    for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]
    for seg in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
]

BUTTON_RADIUS = 2.5  # mm
ANOMALY_POS = (11.0, -6.0)
NORMAL_POS = (11.0, 6.0)


class ButtonArena(FlatTerrain):
    """Flat ground with an ANOMALY button and a NOT ANOMALY button."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._add_button(ANOMALY_POS, rgba=(0.85, 0.15, 0.15, 1.0), name="btn_anomaly")
        self._add_button(NORMAL_POS, rgba=(0.15, 0.45, 0.85, 1.0), name="btn_normal")

    def _add_button(self, pos, rgba, name) -> None:
        self.root_element.worldbody.add(
            "geom",
            name=name,
            type="cylinder",
            size=(BUTTON_RADIUS, 0.12),
            pos=(pos[0], pos[1], 0.12),
            rgba=rgba,
            contype=0,  # decorative: the fly walks over it, not into it
            conaffinity=0,
        )
        self.root_element.worldbody.add(
            "geom",
            name=f"{name}_rim",
            type="cylinder",
            size=(BUTTON_RADIUS * 1.15, 0.06),
            pos=(pos[0], pos[1], 0.06),
            rgba=(rgba[0] * 0.5, rgba[1] * 0.5, rgba[2] * 0.5, 1.0),
            contype=0,
            conaffinity=0,
        )


@dataclass
class Verdict:
    """What the fly did on one event."""

    event_index: int
    novelty: float
    threshold: float
    called_anomaly: bool
    truth: int | None
    reached: bool
    walk_time_s: float
    trajectory: np.ndarray


def _heading_error(pos_xy: np.ndarray, heading_xy: np.ndarray, target: np.ndarray) -> float:
    """Signed heading error in radians. Positive means the target is to the left."""
    h = heading_xy / (np.linalg.norm(heading_xy) + 1e-9)
    d = target - pos_xy
    d = d / (np.linalg.norm(d) + 1e-9)
    cross = h[0] * d[1] - h[1] * d[0]
    dot = float(np.clip(h @ d, -1.0, 1.0))
    return float(np.arctan2(cross, dot))


class ButtonPressRunner:
    """Runs one event: the fly hears the MBON and walks to a button."""

    def __init__(
        self,
        *,
        timestep: float = 1e-4,
        max_time_s: float = 3.0,
        window_size: tuple[int, int] = (720, 544),
        fps: int = 30,
        play_speed: float = 0.5,
        steer_gain: float = 1.2,
        base_drive: float = 1.0,
        seed: int = 0,
    ) -> None:
        self.timestep = timestep
        self.max_steps = int(max_time_s / timestep)
        self.window_size = window_size
        self.fps = fps
        self.play_speed = play_speed
        self.steer_gain = steer_gain
        self.base_drive = base_drive
        self.seed = seed

    def _build(self, record: bool):
        arena = ButtonArena()
        fly = Fly(
            enable_adhesion=True,
            contact_sensor_placements=CONTACT_SENSORS,
            spawn_pos=(0.0, 0.0, 0.2),
            control="position",
            init_pose="stretch",
        )
        cameras = []
        if record:
            cameras.append(
                Camera(
                    attachment_point=arena.root_element.worldbody,
                    camera_name="overview",
                    targeted_fly_names=[],
                    window_size=self.window_size,
                    fps=self.fps,
                    play_speed=self.play_speed,
                    play_speed_text=False,
                    camera_parameters={
                        "name": "overview",
                        "mode": "fixed",
                        "pos": [6.0, 0.0, 22.0],
                        "euler": [0.0, 0.0, 1.5708],
                        "fovy": 45,
                    },
                )
            )
        sim = HybridTurningController(
            fly=fly, cameras=cameras, arena=arena,
            timestep=self.timestep, seed=self.seed,
        )
        return sim, cameras

    def run(
        self,
        *,
        novelty: float,
        threshold: float,
        event_index: int = 0,
        truth: int | None = None,
        video_path: str | Path | None = None,
    ) -> Verdict:
        record = video_path is not None
        sim, cameras = self._build(record)
        obs, _ = sim.reset(seed=self.seed)

        called_anomaly = bool(novelty >= threshold)
        target = np.array(ANOMALY_POS if called_anomaly else NORMAL_POS)

        traj, reached, steps_taken = [], False, self.max_steps
        for step in range(self.max_steps):
            pos = np.asarray(obs["fly"][0][:2], dtype=float)
            heading = np.asarray(obs["fly_orientation"][:2], dtype=float)
            traj.append(pos.copy())

            if np.linalg.norm(target - pos) < BUTTON_RADIUS * 0.6:
                reached, steps_taken = True, step
                break

            err = _heading_error(pos, heading, target)
            # positive err means turn left, which needs more drive on the right
            turn = float(np.clip(self.steer_gain * err, -0.6, 0.6))
            drive = np.clip(
                [self.base_drive - turn, self.base_drive + turn], -0.5, 1.5
            )
            obs, _, _, _, _ = sim.step(np.asarray(drive, dtype=np.float32))
            if record:
                sim.render()

        # let the press land on screen
        if record and reached:
            for _ in range(int(0.25 / self.timestep)):
                obs, _, _, _, _ = sim.step(np.array([0.0, 0.0], dtype=np.float32))
                sim.render()

        if record:
            Path(video_path).parent.mkdir(parents=True, exist_ok=True)
            cameras[0].save_video(video_path)
        sim.close()

        return Verdict(
            event_index=event_index,
            novelty=float(novelty),
            threshold=float(threshold),
            called_anomaly=called_anomaly,
            truth=truth,
            reached=reached,
            walk_time_s=steps_taken * self.timestep,
            trajectory=np.array(traj),
        )


# ---------------------------------------------------------------------------
# Free walking: the honest version.
#
# `ButtonPressRunner` above steers the fly to a chosen target with a
# proportional controller on heading error. That works, but the mushroom body
# only contributes one bit to it -- which target -- and everything else is
# navigation I wrote. A fly has no circuit that nulls a heading error to a
# remembered goal.
#
# What a fly actually does with odour valence is klinokinesis: the MBON biases
# turn rate, and where the animal ends up is a consequence rather than a
# setpoint. That is what this runner does. The drive is set once from the
# novelty score and held; nothing reads the fly's position back. Whether it
# drifts onto the red side or the blue side is emergent.
# ---------------------------------------------------------------------------

from .descending import descending_drive  # noqa: E402

ZONE_HALF_WIDTH = 26.0
ZONE_INNER_GAP = 2.0


class DriftArena(FlatTerrain):
    """Flat ground tinted red on the fly's left and blue on its right.

    The tints are scenery, not targets. Nothing in the simulation reads them and
    the fly is not trying to reach either one.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._zone(+1, (0.62, 0.20, 0.18, 0.55), "zone_anomaly")
        self._zone(-1, (0.18, 0.34, 0.62, 0.55), "zone_normal")

    def _zone(self, sign: int, rgba, name: str) -> None:
        centre_y = sign * (ZONE_INNER_GAP + ZONE_HALF_WIDTH)
        self.root_element.worldbody.add(
            "geom", name=name, type="box",
            size=(ZONE_HALF_WIDTH, ZONE_HALF_WIDTH, 0.02),
            pos=(14.0, centre_y, 0.02), rgba=rgba, contype=0, conaffinity=0,
        )


@dataclass
class DriftVerdict:
    """Where the fly went, and what the circuit had said."""

    event_index: int
    novelty: float
    threshold: float
    called_anomaly: bool          # the MBON's call
    truth: int | None
    drive: tuple[float, float]
    final_xy: tuple[float, float]
    lateral_drift: float          # net y displacement, mm; positive = left
    drifted_left: bool            # emergent, not commanded
    agrees_with_mbon: bool
    trajectory: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)), repr=False)


class FreeWalkRunner:
    """Fly walks freely; the MBON biases its turning and nothing else.

    Measured reliability, 2.5 s trials at max_turn=0.20, three CPG seeds x
    {novel, familiar}: the drift direction agreed with the MBON in 4 of 6. The
    two disagreements were both the same seed, and both were falls rather than
    opposite decisions -- forward progress collapsed to 0.6 mm/s against the
    6.8-10 mm/s of a gait that is holding. At 5 s the run degrades further and
    one trial crashed the solver outright (mjWARN_BADQACC).

    So the descending bias does reach the ground and does steer the animal, but
    only while the animal stays upright, and it falls often enough that the
    trajectory is not a dependable readout. This is the honest reason the
    tethered preparation exists, and the reason the reel uses it: a fly that
    cannot fall gives you a clean measurement of what its circuits asked for.
    """

    def __init__(self, *, timestep: float = 1e-4, duration_s: float = 2.5,
                 window_size: tuple[int, int] = (704, 544), fps: int = 30,
                 play_speed: float = 0.4, seed: int = 0) -> None:
        self.timestep = timestep
        self.n_steps = int(duration_s / timestep)
        self.window_size = window_size
        self.fps = fps
        self.play_speed = play_speed
        self.seed = seed

    def _build(self, record: bool):
        arena = DriftArena()
        fly = Fly(enable_adhesion=True, contact_sensor_placements=CONTACT_SENSORS,
                  spawn_pos=(0.0, 0.0, 0.2), control="position", init_pose="stretch")
        cameras = []
        if record:
            cameras.append(Camera(
                attachment_point=arena.root_element.worldbody, camera_name="drift",
                targeted_fly_names=[], window_size=self.window_size, fps=self.fps,
                play_speed=self.play_speed, play_speed_text=False,
                camera_parameters={
                    "name": "drift", "mode": "fixed",
                    "pos": [-14.0, 0.0, 22.0],
                    "xyaxes": [0.0, 1.0, 0.0, 0.84, 0.0, 0.55],
                    "fovy": 50,
                },
            ))
        sim = HybridTurningController(fly=fly, cameras=cameras, arena=arena,
                                      timestep=self.timestep, seed=self.seed)
        return sim, cameras

    def run(self, *, novelty: float, threshold: float, event_index: int = 0,
            truth: int | None = None, video_path: str | Path | None = None,
            gain: float = 1.5, max_turn: float = 0.20) -> DriftVerdict:
        """`max_turn` defaults to 0.20 for a reason, measured not guessed.

        Swept over {0.10, 0.20, 0.30, 0.55} at 2.5 s: the fly walks at 6.8-7.8
        mm/s at 0.20 (unbiased walking is ~10 mm/s) and drifts in the direction
        the MBON asked for. At 0.55 -- the value the tethered rig uses happily,
        because a tethered fly cannot fall -- free walking collapses to under
        2 mm/s and the drift sign inverts. A turn command sharp enough to be
        legible on a treadmill is sharp enough to trip an untethered animal.
        """
        record = video_path is not None
        sim, cameras = self._build(record)
        obs, _ = sim.reset(seed=self.seed)

        drive = descending_drive(novelty, threshold, gain=gain, max_turn=max_turn)
        traj = np.zeros((self.n_steps, 2))
        for i in range(self.n_steps):
            obs, _, _, _, _ = sim.step(drive)          # open loop: never reads back
            traj[i] = np.asarray(obs["fly"][0][:2], dtype=float)
            if record:
                sim.render()

        if record:
            Path(video_path).parent.mkdir(parents=True, exist_ok=True)
            cameras[0].save_video(video_path)
        sim.close()

        called = bool(float(novelty) >= float(threshold))
        drift = float(traj[-1, 1] - traj[0, 1])
        return DriftVerdict(
            event_index=event_index, novelty=float(novelty), threshold=float(threshold),
            called_anomaly=called, truth=truth,
            drive=(float(drive[0]), float(drive[1])),
            final_xy=(float(traj[-1, 0]), float(traj[-1, 1])),
            lateral_drift=drift, drifted_left=bool(drift > 0),
            agrees_with_mbon=bool((drift > 0) == called), trajectory=traj,
        )
