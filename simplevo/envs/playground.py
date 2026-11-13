"""CPU Panda pick-and-place playground; optional dependency: pip install 'simplevo[playground]'.

Run `python -m simplevo.envs.playground --out runs/panda` for a scripted physics demonstration.
The model-facing adapter is `simplevo.envs.playground:make_env`; it never starts a model session.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict
from importlib.resources import files
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

from ..agent import run_episode
from ..backends import ScriptedSession
from ..content import decode_png, encode_png
from ..demonstration import Demonstration
from ..task import Embodiment, Environment, Outcome, TaskSpec

SETTINGS = yaml.safe_load(files("simplevo").joinpath("envs/playground.yaml").read_text(encoding="utf-8"))
EMBODIMENT = Embodiment(
    description=("A Franka Panda arm mounted at the world origin, on a horizontal surface at z=0 metres. "
                 "+x extends forward from the base, +y to its left, +z up. The working area is x=0.3..0.65, "
                 "y=-0.3..0.3. End-effector poses refer to the grasp target between the fingertips (link 11). "
                 "Quaternion order is w,x,y,z; [0,1,0,0] points the fingers down. Opening 1 is fully open "
                 "(8 cm gap), 0 fully closed. The red cube is 4 cm wide. The overview camera looks diagonally "
                 "across the table; overhead looks down, with +x toward the top and +y toward the left."),
    arms=("arm",), cameras=("overview", "overhead"))
TASK = TaskSpec("panda_block_to_pad", "Pick up the red cube and place it on the green pad, then release it.", EMBODIMENT)


def layout(scene):
    """Deterministic fixture layouts, used by the simulator and the offline reference trajectory."""
    rng = random.Random(int(scene))
    return [rng.uniform(0.43, 0.51), rng.uniform(-0.18, -0.10)], [rng.uniform(0.43, 0.51), rng.uniform(0.12, 0.20)]


class PandaTabletop(Environment):
    embodiment = EMBODIMENT

    def __init__(self, scene, *, record=False):
        try:
            import pybullet as p
            import pybullet_data
            from pybullet_utils.bullet_client import BulletClient
        except ImportError as error:
            raise ImportError("Install the playground with: pip install 'simplevo[playground]'") from error
        block_xy, self._pad_xy = layout(scene)
        self._p = BulletClient(connection_mode=p.DIRECT)
        self._closed = False
        self._steps, self._stable, self._outcome = 0, 0, None
        self._lifted = False
        self._record = record
        self.frames, self.states = [], []
        try:
            b = self._p
            b.setAdditionalSearchPath(pybullet_data.getDataPath())
            b.setGravity(0, 0, -9.81)
            b.setTimeStep(1 / SETTINGS["physics_hz"])
            b.setPhysicsEngineParameter(numSolverIterations=100, deterministicOverlappingPairs=1)
            table_collision = b.createCollisionShape(b.GEOM_BOX, halfExtents=[0.8, 0.65, 0.035])
            table_visual = b.createVisualShape(b.GEOM_BOX, halfExtents=[0.8, 0.65, 0.035],
                                               rgbaColor=[0.72, 0.77, 0.82, 1])
            b.createMultiBody(0, table_collision, table_visual, [0.4, 0, -0.035])
            self._robot = b.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
            for joint, value in enumerate([0, -0.4, 0, -2.4, 0, 2.0, math.pi / 4]):
                b.resetJointState(self._robot, joint, value)
            for joint in (9, 10):
                b.resetJointState(self._robot, joint, 0.04)
                b.changeDynamics(self._robot, joint, lateralFriction=1.5)
            half = SETTINGS["cube_half_size"]
            collision = b.createCollisionShape(b.GEOM_BOX, halfExtents=[half] * 3)
            visual = b.createVisualShape(b.GEOM_BOX, halfExtents=[half] * 3, rgbaColor=[0.85, 0.08, 0.06, 1])
            self._cube = b.createMultiBody(SETTINGS["cube_mass"], collision, visual, [*block_xy, half])
            b.changeDynamics(self._cube, -1, lateralFriction=1, spinningFriction=0.01)
            pad = b.createVisualShape(b.GEOM_BOX, halfExtents=[SETTINGS["pad_half_size"]] * 2 + [0.001],
                                      rgbaColor=[0.08, 0.65, 0.3, 1])
            b.createMultiBody(0, -1, pad, [*self._pad_xy, 0.001])
            self._views = {
                "overview": b.computeViewMatrix([1.15, -1.05, 0.95], [0.35, 0, 0.28], [0, 0, 1]),
                "overhead": b.computeViewMatrix([0.45, 0, 1.25], [0.45, 0, 0], [1, 0, 0]),
            }
            self._projection = b.computeProjectionMatrixFOV(48, SETTINGS["width"] / SETTINGS["height"], 0.02, 3)
            self._capture()
        except Exception:
            self.close()
            raise

    def state(self):
        b = self._p
        pose = b.getLinkState(self._robot, 11, computeForwardKinematics=True)
        q = pose[5]
        opening = sum(b.getJointState(self._robot, j)[0] for j in (9, 10)) / 0.08
        return {"arm": {"xyz_m": list(pose[4]), "quaternion_wxyz": [q[3], *q[:3]],
                        "gripper": float(np.clip(opening, 0, 1))}}

    def step(self, actions):
        b, executed = self._p, 0
        for action in actions:
            if self.ended:
                break
            arm = action["arm"]
            q = arm["quaternion_wxyz"]
            joints = b.calculateInverseKinematics(self._robot, 11, arm["xyz_m"], [*q[1:], q[0]],
                                                  maxNumIterations=100, residualThreshold=1e-5)
            b.setJointMotorControlArray(self._robot, list(range(7)), b.POSITION_CONTROL,
                                       targetPositions=joints[:7], forces=[SETTINGS["arm_force"]] * 7)
            b.setJointMotorControlArray(self._robot, [9, 10], b.POSITION_CONTROL,
                                       targetPositions=[0.04 * arm["gripper"]] * 2,
                                       forces=[SETTINGS["finger_force"]] * 2)
            for _ in range(SETTINGS["substeps"]):
                b.stepSimulation()
            self._steps += 1
            executed += 1
            position = b.getBasePositionAndOrientation(self._cube)[0]
            self._lifted = self._lifted or position[2] > 0.10
            velocity = b.getBaseVelocity(self._cube)[0]
            resting = (self._lifted and math.dist(position[:2], self._pad_xy) < SETTINGS["pad_half_size"] - SETTINGS["cube_half_size"]
                       and abs(position[2] - SETTINGS["cube_half_size"]) < 0.008
                       and np.linalg.norm(velocity) < 0.03 and self.state()["arm"]["gripper"] > 0.8)
            self._stable = self._stable + 1 if resting else 0
            if self._stable >= SETTINGS["settle_steps"]:
                self._outcome = Outcome(True, 1.0)
            elif self._steps >= SETTINGS["step_limit"]:
                self._outcome = Outcome(False, 0.0)
            self._capture()
        return executed

    def render(self, camera):
        if camera not in self._views:
            raise ValueError(f"unknown camera: {camera}")
        b = self._p
        result = b.getCameraImage(SETTINGS["width"], SETTINGS["height"], self._views[camera], self._projection,
                                  renderer=b.ER_TINY_RENDERER, flags=b.ER_NO_SEGMENTATION_MASK)
        rgb = np.asarray(result[2], dtype=np.uint8).reshape(SETTINGS["height"], SETTINGS["width"], 4)[..., :3]
        return encode_png(Image.fromarray(rgb))

    def _capture(self):
        if self._record:
            self.states.append(self.state())
            self.frames.append({camera: self.render(camera) for camera in self.embodiment.cameras})

    @property
    def ended(self):
        return self._outcome is not None

    @property
    def outcome(self):
        return self._outcome

    @property
    def steps_used(self):
        return self._steps

    def close(self):
        if not self._closed:
            self._p.disconnect()
            self._closed = True

    def export(self, out):
        """Export RGB/proprioception recording and a labelled preview of the scripted physics run."""
        if not self.frames:
            raise ValueError("record=True is required to export a demonstration")
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        demo = Demonstration(TASK.instruction,
                             {"arm": np.array([s["arm"]["xyz_m"] + s["arm"]["quaternion_wxyz"] for s in self.states])},
                             {"arm": np.array([s["arm"]["gripper"] for s in self.states])},
                             self.embodiment.cameras,
                             lambda camera, index: decode_png(self.frames[index][camera]))
        demo.save(out / "demo")
        previews = []
        for index in list(range(0, len(self.frames), 3)) + [len(self.frames) - 1]:
            picture = decode_png(self.frames[index]["overview"]).copy()
            draw = ImageDraw.Draw(picture)
            draw.rectangle((0, 0, picture.width, 27), fill="#122033")
            draw.text((12, 8), "SimplEvo playground | scripted physics demo", fill="white")
            previews.append(picture)
        previews[0].save(out / "preview.gif", save_all=True, append_images=previews[1:], duration=100, loop=0)
        previews[-1].save(out / "result.png")
        (out / "task.json").write_text(json.dumps(asdict(TASK), indent=2) + "\n", encoding="utf-8")
        (out / "simulation.yaml").write_text(yaml.safe_dump(SETTINGS), encoding="utf-8")


def make_env(scene):
    """Factory for the existing annotate/adapt/deploy CLI; creates no model or account connection."""
    return PandaTabletop(scene)


def reference_policy(scene):
    """Known-layout fixture for offline integration checks, not a learned or RGB-based policy."""
    block, pad = layout(scene)

    def policy(inputs, call):
        def move(xy, z, opening, steps=48):
            call("execute_eef", {"waypoints": [{"arm": {"xyz_m": [*xy, z],
                 "quaternion_wxyz": [0, 1, 0, 0], "gripper": opening}, "steps": steps,
                 "motion_steps": steps - 12}], "cameras": ["overview"], "reason": "offline reference trajectory"})
        move(block, 0.20, 1)
        move(block, 0.025, 1)
        call("set_gripper", {"side": "arm", "opening": 0, "steps": 40, "cameras": ["overview"]})
        move(block, 0.22, 0)
        move(pad, 0.22, 0)
        move(pad, 0.035, 0)
        call("set_gripper", {"side": "arm", "opening": 1, "steps": 40, "cameras": ["overview"]})
        move(pad, 0.22, 1)
        return "Reference trajectory completed."
    return policy


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scene", type=int, default=0)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("choose a new output directory")
    env = PandaTabletop(args.scene, record=True)
    try:
        result = run_episode(env, ScriptedSession(reference_policy(args.scene)), TASK, args.out / "episode")
        env.export(args.out)
        (args.out / "result.json").write_text(json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"out": str(args.out), "outcome": asdict(result.outcome), "driver": "scripted"}))
        return 0 if result.outcome.success else 1
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
