"""Action formulation: chunks of absolute end-effector waypoints (paper Section 3.1).

An action ``a_t`` is one ``execute_eef`` call with up to ``max_waypoints`` waypoints. A waypoint sets, for each arm,
an end-effector pose and a gripper target (and a base pose on a mobile robot) and lasts a number of environment
steps. The harness turns a waypoint into per-step targets; nothing is quantized, planned or approved for the model.
``set_gripper`` and ``observe`` complete the interface.
"""
from __future__ import annotations

import numpy as np

from .schema import tool
from .task import Embodiment, Environment


def _vector(value, length: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (length,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain {length} finite numbers")
    return array


def unit(quaternion) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=float)
    return quaternion / np.linalg.norm(quaternion)


def slerp(start, end, alpha: float) -> np.ndarray:
    """Spherical interpolation between two unit quaternions (wxyz), along the shorter arc."""
    start, end = unit(start), unit(end)
    dot = float(np.dot(start, end))
    if dot < 0:
        end, dot = -end, -dot
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    if theta < 1e-6:
        return unit(start * (1 - alpha) + end * alpha)
    return unit((np.sin((1 - alpha) * theta) * start + np.sin(alpha * theta) * end) / np.sin(theta))


def rotation_deg(start, end) -> float:
    """Angle of the rotation that takes one unit quaternion to another, in degrees."""
    return float(np.degrees(2 * np.arccos(np.clip(abs(float(np.dot(unit(start), unit(end)))), 0.0, 1.0))))


def tool_specs(embodiment: Embodiment) -> list[dict]:
    def vec(n):
        return {"type": "array", "items": {"type": "number"}, "minItems": n, "maxItems": n}

    limit = embodiment.max_steps
    arm = {"type": "object", "additionalProperties": False, "required": ["xyz_m", "quaternion_wxyz", "gripper"],
           "properties": {"xyz_m": vec(3), "quaternion_wxyz": vec(4),
                          "gripper": {"type": "number", "minimum": 0, "maximum": 1}}}
    waypoint = {name: {"anyOf": [arm, {"type": "null"}]} for name in embodiment.arms}
    if embodiment.has_base:
        waypoint["base"] = {"anyOf": [{"type": "null"}, {
            "type": "object", "additionalProperties": False, "required": ["xy_m", "yaw_rad"],
            "properties": {"xy_m": vec(2), "yaw_rad": {"type": "number"}}}]}
    required = list(waypoint) + ["steps"]
    waypoint["steps"] = {"type": "integer", "minimum": 1, "maximum": limit}
    waypoint["motion_steps"] = {"type": "integer", "minimum": 0, "maximum": limit}
    cameras = {"type": "array", "items": {"type": "string", "enum": list(embodiment.cameras)}, "minItems": 1,
               "maxItems": len(embodiment.cameras)}
    base = " A null base holds the measured base pose." if embodiment.has_base else ""
    return [
        tool("execute_eef", (
            f"Execute 1..{embodiment.max_waypoints} absolute end-effector waypoints in order. Each arm target has "
            "xyz(m) and a unit quaternion(wxyz) in the same frame as the measured state, plus a gripper command in "
            "[0,1]: 1 open, 0 closed. No translation or angle quantization. All arms may move together. A null arm "
            f"holds its measured pose and prior gripper command.{base} Each waypoint consumes 'steps' environment "
            f"steps (1..{limit}). Optional motion_steps (0..steps, default 0) linearly ramps position and "
            "spherically interpolates rotation from the measured pose to the target over that many steps, then "
            "holds for the remainder. Gripper commands apply immediately, so close in a separate stationary "
            "waypoint. With motion_steps=0 the full target applies immediately: extra steps are holding time, not "
            "slow motion, and a large jump may not be tracked. All commands are absolute. No automatic grasp "
            "approval or task-stage rules are imposed. Returns the measured robot state and the selected RGB "
            "cameras after the last waypoint; a sequence has no intermediate observation. You choose target poses, "
            "timing, observation cadence, camera subset and when to batch waypoints."),
            {"waypoints": {"type": "array", "minItems": 1, "maxItems": embodiment.max_waypoints,
                           "items": {"type": "object", "additionalProperties": False, "required": required,
                                     "properties": waypoint}},
             "cameras": cameras, "reason": {"type": "string"}}),
        tool("set_gripper", (
            "Send a normalized gripper command to one arm: opening=0 fully closed, opening=1 fully open. Holds "
            f"every arm at its measured pose, applies immediately and waits the requested 1..{limit} steps. Use "
            "this for closing at rest. Judge contact and grasps from the RGB views. Returns the selected RGB views "
            "and the measured robot state."),
            {"side": {"type": "string", "enum": list(embodiment.arms)},
             "opening": {"type": "number", "minimum": 0, "maximum": 1},
             "steps": {"type": "integer", "minimum": 1, "maximum": limit}, "cameras": cameras}),
        tool("observe", (
            "Return selected RGB views and the measured robot state from the latest executed step. Does not advance "
            "the environment. Use execute_eef with null targets to wait and refresh the scene."),
            {"cameras": cameras}),
    ]


class Controller:
    """Turns validated tool arguments into per-step actions for an ``Environment``."""

    def __init__(self, env: Environment):
        self.env = env
        self.embodiment = env.embodiment
        state = env.state()
        self.grippers = {arm: float(state[arm]["gripper"]) for arm in self.embodiment.arms}  # last commands
        self.waypoints = 0

    def _measured(self):
        state = self.env.state()
        arms = {arm: (_vector(state[arm]["xyz_m"], 3, arm + " position"),
                      unit(_vector(state[arm]["quaternion_wxyz"], 4, arm + " quaternion")))
                for arm in self.embodiment.arms}
        base = None
        if self.embodiment.has_base:
            base = (_vector(state["base"]["xy_m"], 2, "base position"), float(state["base"]["yaw_rad"]))
        return arms, base

    def _target(self, waypoint: dict, arms, base):
        target = {}
        for arm in self.embodiment.arms:
            wanted = waypoint[arm]
            if wanted is None:
                target[arm] = (*arms[arm], self.grippers[arm])
                continue
            quaternion = _vector(wanted["quaternion_wxyz"], 4, arm + " quaternion")
            if abs(np.linalg.norm(quaternion) - 1) > 0.02:
                raise ValueError("quaternions must be unit length (small rounding error is normalized)")
            target[arm] = (_vector(wanted["xyz_m"], 3, arm + " position"), unit(quaternion), float(wanted["gripper"]))
        if base is not None:
            wanted = waypoint.get("base")
            target["base"] = base if wanted is None else (_vector(wanted["xy_m"], 2, "base position"),
                                                          float(wanted["yaw_rad"]))
        return target

    def _hold(self, arms, base, grippers) -> dict:
        action = {arm: {"xyz_m": arms[arm][0].tolist(), "quaternion_wxyz": arms[arm][1].tolist(),
                        "gripper": grippers[arm]} for arm in self.embodiment.arms}
        if base is not None:
            action["base"] = {"xy_m": base[0].tolist(), "yaw_rad": base[1]}
        return action

    def waypoint_actions(self, waypoint: dict) -> tuple[list[dict], dict]:
        """Per-step targets of one waypoint: a ramp from the measured pose over ``motion_steps`` (0 = the target
        at once), then the target held for the rest of ``steps``. Gripper commands apply from the first step."""
        steps, motion = waypoint["steps"], waypoint.get("motion_steps", 0)
        if motion > steps:
            raise ValueError("motion_steps must be between 0 and steps")
        arms, base = self._measured()
        target = self._target(waypoint, arms, base)
        actions = []
        for step in range(1, steps + 1):
            alpha = 1.0 if not motion else min(step / motion, 1.0)
            action = {}
            for arm in self.embodiment.arms:
                xyz, quaternion, gripper = target[arm]
                action[arm] = {"xyz_m": (arms[arm][0] + alpha * (xyz - arms[arm][0])).tolist(),
                               "quaternion_wxyz": slerp(arms[arm][1], quaternion, alpha).tolist(), "gripper": gripper}
            if base is not None:
                xy, yaw = target["base"]
                turn = (yaw - base[1] + np.pi) % (2 * np.pi) - np.pi
                action["base"] = {"xy_m": (base[0] + alpha * (xy - base[0])).tolist(),
                                  "yaw_rad": float(base[1] + alpha * turn)}
            actions.append(action)
        return actions, target

    def execute(self, arguments: dict) -> list[dict]:
        """Run the waypoints of one ``execute_eef`` call; returns one summary per started waypoint."""
        summaries = []
        for index, waypoint in enumerate(arguments["waypoints"]):
            if self.env.ended:
                break
            actions, target = self.waypoint_actions(waypoint)
            executed = self.env.step(actions)
            self.grippers = {arm: target[arm][2] for arm in self.embodiment.arms}
            self.waypoints += 1
            arms, _ = self._measured()
            summaries.append({"waypoint": index, "steps_executed": executed, "position_error_m": {
                arm: round(float(np.linalg.norm(arms[arm][0] - target[arm][0])), 4) for arm in self.embodiment.arms}})
        return summaries

    def set_gripper(self, arguments: dict) -> list[dict]:
        arms, base = self._measured()
        grippers = {**self.grippers, arguments["side"]: float(arguments["opening"])}
        executed = self.env.step([self._hold(arms, base, grippers) for _ in range(arguments["steps"])])
        self.grippers = grippers
        return [{"gripper": arguments["side"], "opening": grippers[arguments["side"]], "steps_executed": executed}]

    def hold_until_end(self, chunk: int = 32, limit: int = 1_000_000) -> int:
        """Keep the robot still until the evaluator ends the episode; returns the steps spent holding."""
        held = 0
        while not self.env.ended and held < limit:
            arms, base = self._measured()
            executed = self.env.step([self._hold(arms, base, self.grippers) for _ in range(chunk)])
            if executed <= 0:
                break
            held += executed
        return held


def model_state(env: Environment) -> dict:
    """The measured state as shown to the model: rounded, with the elapsed steps and never the step limit."""
    def rounded(value):
        if isinstance(value, dict):
            return {key: rounded(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [rounded(item) for item in value]
        return round(float(value), 5)
    return {**rounded(env.state()), "env_steps_used": env.steps_used}
