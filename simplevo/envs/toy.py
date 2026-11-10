"""A toy tabletop for tests and the offline example: no simulator, no model, no credentials.

Two point grippers above a table, one red block and one green pad. The task is to put the block on the pad. A
closed gripper holds the block only if it closes low enough right above it, which a first attempt has to find out.
The environment follows the information protocol: it gives RGB images and the measured gripper poses, never the
block's position.

``ToyModel`` is a scripted stand-in for the MLLM. It locates the block and the pad in the overview image, tries a
grasp height, and in the reflection turn writes what it learned into the working memory. It exists to exercise the
whole loop offline; it says nothing about how a real model behaves.
"""
from __future__ import annotations

import json
import math
import random
import re

from PIL import Image, ImageDraw

from ..content import decode_png, encode_png
from ..task import Embodiment, Environment, Outcome, TaskSpec

EMBODIMENT = Embodiment(
    description=("two point grippers above a table. World frame in metres: +x to the right (-0.5..0.5), +y away "
                 "from the robot (0..0.6), +z up, table at z = 0. Poses are gripper-tip poses; the orientation is "
                 "fixed. Gripper: 1 open, 0 closed. Cameras: overview (top-down, the whole table), left_wrist and "
                 "right_wrist (top-down close-ups around each gripper)."),
    arms=("left", "right"), cameras=("overview", "left_wrist", "right_wrist"))
TASK = TaskSpec("toy_block_to_pad", "Put the red block on the green pad.", EMBODIMENT)
PIXELS_PER_M = 320
WIDTH, HEIGHT = 320, 192
SPEED = 0.03            # metres an arm can travel per step
GRASP_HEIGHT = 0.04     # a gripper must close at or below this height, right above the block
REACH = 0.02
STEP_LIMIT = 400
IDENTITY = [1.0, 0.0, 0.0, 0.0]
COLORS = {"table": (222, 214, 200), "block": (214, 40, 40), "pad": (60, 170, 80),
          "left": (50, 90, 200), "right": (230, 150, 30)}


def to_pixel(x: float, y: float) -> tuple[float, float]:
    return (x + 0.5) * PIXELS_PER_M, (0.6 - y) * PIXELS_PER_M


def to_world(px: float, py: float) -> tuple[float, float]:
    return px / PIXELS_PER_M - 0.5, 0.6 - py / PIXELS_PER_M


class ToyTabletop(Environment):
    embodiment = EMBODIMENT

    def __init__(self, scene: int):
        rng = random.Random(scene)
        self.block = [rng.uniform(-0.3, 0.3), rng.uniform(0.2, 0.5)]
        while True:
            self.pad = [rng.uniform(-0.3, 0.3), rng.uniform(0.2, 0.5)]
            if math.dist(self.pad, self.block) > 0.15:
                break
        self.arms = {"left": [-0.25, 0.05, 0.25], "right": [0.25, 0.05, 0.25]}
        self.grippers = {"left": 1.0, "right": 1.0}
        self.holder = None
        self.lifted = False
        self._steps = 0
        self._outcome = None

    def state(self):
        return {arm: {"xyz_m": list(self.arms[arm]), "quaternion_wxyz": IDENTITY, "gripper": self.grippers[arm]}
                for arm in self.arms}

    def step(self, actions):
        executed = 0
        for action in actions:
            if self.ended:
                break
            for arm, position in self.arms.items():
                target = action[arm]["xyz_m"]
                distance = math.dist(position, target)
                scale = 1.0 if distance <= SPEED else SPEED / distance
                self.arms[arm] = [p + scale * (t - p) for p, t in zip(position, target)]
                was_open, self.grippers[arm] = self.grippers[arm] >= 0.5, float(action[arm]["gripper"])
                closed = self.grippers[arm] < 0.5
                if closed and was_open and self.holder is None and self.arms[arm][2] <= GRASP_HEIGHT \
                        and math.dist(self.arms[arm][:2], self.block) <= REACH:
                    self.holder = arm
                elif not closed and self.holder == arm:
                    self.holder = None
            if self.holder is not None:
                self.block = list(self.arms[self.holder][:2])
                self.lifted = self.lifted or self.arms[self.holder][2] > 0.05
            self._steps += 1
            executed += 1
            if self.holder is None and math.dist(self.block, self.pad) <= 0.03:
                self._outcome = Outcome(True, 1.0)
            elif self._steps >= STEP_LIMIT:
                self._outcome = Outcome(False, 0.5 if self.lifted else 0.0)
        return executed

    @property
    def ended(self):
        return self._outcome is not None

    @property
    def outcome(self):
        return self._outcome

    @property
    def steps_used(self):
        return self._steps

    def render(self, camera):
        picture = Image.new("RGB", (WIDTH, HEIGHT), COLORS["table"])
        draw = ImageDraw.Draw(picture)

        def square(center, half, color):
            px, py = to_pixel(*center)
            draw.rectangle([px - half, py - half, px + half, py + half], fill=color)
        square(self.pad, 9, COLORS["pad"])
        square(self.block, 5, COLORS["block"])
        for arm, position in self.arms.items():
            px, py = to_pixel(*position[:2])
            radius = 4 + 12 * position[2]          # a higher gripper looks larger from above
            draw.ellipse([px - radius, py - radius, px + radius, py + radius], outline=COLORS[arm], width=2,
                         fill=COLORS[arm] if self.grippers[arm] < 0.5 else None)
        if camera != "overview":
            px, py = to_pixel(*self.arms[camera.removesuffix("_wrist")][:2])
            picture = picture.crop((int(px) - 32, int(py) - 32, int(px) + 32, int(py) + 32)).resize((96, 96))
        return encode_png(picture)


# ----------------------------------------------------------------------------- a scripted stand-in for the model

def _centroid(picture: Image.Image, color) -> tuple[float, float] | None:
    points = [(x, y) for y in range(picture.height) for x in range(picture.width)
              if picture.getpixel((x, y)) == color]
    if not points:
        return None
    return sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points)


def _json(result) -> dict:
    return json.loads(next(item["text"] for item in result.content if item["text"].startswith("{")))


def _overview(items) -> Image.Image:
    index = next(i for i, item in enumerate(items) if item.get("text") == "RGB view: overview")
    return decode_png(items[index + 1]["png"])


class ToyModel:
    """Tries to grasp at a height taken from its notes (or a first guess that is too high) and writes a lesson."""

    FIRST_GUESS = 0.10

    def __init__(self):
        self.height = None
        self.attempt = 1

    def __call__(self, inputs, call):
        if inputs[0]["text"].startswith("The episode has ended"):
            return self.reflect(inputs[0]["text"], call)
        found = re.search(r"this is attempt (\d+)", inputs[0]["text"])
        self.attempt = int(found.group(1)) if found else 1
        notes = _json(call("read_task_notes", {}))["notes"] if "read_task_notes" in call.tools else ""
        learned = re.findall(r"close at z=(\d+\.\d+)", notes)
        self.height = float(learned[-1]) if learned else self.FIRST_GUESS
        scene = _overview(inputs)
        block, pad = to_world(*_centroid(scene, COLORS["block"])), to_world(*_centroid(scene, COLORS["pad"]))

        def move(x, y, z, gripper):
            return {"left": {"xyz_m": [x, y, z], "quaternion_wxyz": IDENTITY, "gripper": gripper}, "right": None,
                    "steps": 32, "motion_steps": 24}
        call("execute_eef", {"cameras": ["overview"], "reason": "reach and close on the block", "waypoints": [
            move(*block, 0.2, 1), move(*block, self.height, 1)]})
        call("set_gripper", {"side": "left", "opening": 0, "steps": 4, "cameras": ["left_wrist"]})
        # Lift with a sideways shift: a held block moves with the gripper in the top-down view, a missed one stays.
        lifted = call("execute_eef", {"cameras": ["overview"], "reason": "lift aside and check the grasp",
                                      "waypoints": [move(block[0] + 0.05, block[1], 0.2, 0)]})
        seen = _centroid(_overview(lifted.content), COLORS["block"])
        if seen is not None and math.dist(to_world(*seen), block) < 0.01:
            return f"The block stayed on the table after closing at z={self.height:.2f}."
        call("execute_eef", {"cameras": ["overview"], "reason": "carry to the pad and release", "waypoints": [
            move(*pad, 0.2, 0), move(*pad, 0.05, 0), move(*pad, 0.05, 1)]})
        return "The block is on the pad."

    def reflect(self, prompt, call):
        current = _json(call("read_task_notes", {}))
        if "success=1.0" in prompt:
            lesson = f"Lesson (attempt {self.attempt}): close at z={self.height:.2f} right above the block; it lifted."
        else:
            lower = max(0.02, self.height - 0.04)
            lesson = (f"Lesson (attempt {self.attempt}): closing at z={self.height:.2f} did not lift the block; "
                      f"next close at z={lower:.2f}.")
        call("write_task_notes", {"content": (current["notes"] + "\n" + lesson).strip(),
                                  "expected_sha256": current["sha256"]})
        return "Notes updated."
