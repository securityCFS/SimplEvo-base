"""Task formulation: direct control under a strict information protocol (paper Section 3.1).

The agent is a fixed-weight MLLM plus this harness. At every decision step the model receives

    o_t = (I_t, q_t)   RGB images from the cameras it asks for, and the measured robot state,

and issues ``a_t``: a chunk of absolute end-effector waypoints (``simplevo.control``). Beyond ``o_t`` it receives
only the instruction with the robot constants, the task memory and the working memory. Depth, camera parameters and
object states are never part of the interface: an ``Environment`` adapter must not put them into ``state()``.

A benchmark is connected by implementing ``Environment`` and describing its robot in an ``Embodiment``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Embodiment:
    """Robot constants stated with every instruction and used to build the control tools.

    ``description`` is free text for the model: the world frame, where the arms are mounted, the end-effector
    convention (which link, which axis the fingers extend along), the cameras and the gripper convention.
    """
    description: str
    arms: tuple[str, ...] = ("left", "right")
    cameras: tuple[str, ...] = ("overview", "left_wrist", "right_wrist")
    has_base: bool = False
    max_waypoints: int = 6        # waypoints per execute_eef call
    max_steps: int = 64           # environment steps per waypoint

    def __post_init__(self):
        if not self.arms or not self.cameras:
            raise ValueError("an embodiment needs at least one arm and one camera")


@dataclass(frozen=True)
class TaskSpec:
    """One task: its instruction and robot. ``guidance`` is optional extra text (e.g. ``CAREFUL_GRASP``)."""
    name: str
    instruction: str
    embodiment: Embodiment
    guidance: str = ""


@dataclass(frozen=True)
class Outcome:
    """Terminal outcome reported by the benchmark evaluator: the episode-level reward r_k."""
    success: bool
    score: float | None = None


class Environment(ABC):
    """One live episode of one task in one scene.

    State format (``state()`` and every per-step action use the same frame and units)::

        {arm: {"xyz_m": [x, y, z], "quaternion_wxyz": [w, x, y, z], "gripper": g}, ...,
         "base": {"xy_m": [x, y], "yaw_rad": yaw}}          # "base" only for a mobile robot

    ``gripper`` is normalized: 1 open, 0 closed. The pose is the measured end-effector pose. Converting end-effector
    targets into joint commands (inverse kinematics) is the adapter's job.
    """
    embodiment: Embodiment

    @abstractmethod
    def state(self) -> dict:
        """Measured proprioception q_t. No object poses, no depth, no camera parameters."""

    @abstractmethod
    def render(self, camera: str) -> bytes:
        """PNG bytes of one RGB camera at the latest executed step."""

    @abstractmethod
    def step(self, actions: list[dict]) -> int:
        """Execute per-step actions in order, one environment step each; stop early when the episode ends.
        Returns the number of steps executed."""

    @property
    @abstractmethod
    def ended(self) -> bool:
        """True once the evaluator registered success or the step limit was reached."""

    @property
    @abstractmethod
    def outcome(self) -> Outcome | None:
        """The evaluator's outcome; None until the episode has ended."""

    @property
    @abstractmethod
    def steps_used(self) -> int:
        """Environment steps executed so far."""

    def close(self) -> None:
        """Release the episode."""


BASE_INSTRUCTIONS = (
    "You control one robot episode. Use only the supplied RGB and measured robot state and the declared robot "
    "tools. Do not read external files, use shell, browse, or delegate to other agents. The environment judges "
    "success independently. Stop when it reports a terminal result. Repeated observe at the same step shows the "
    "same scene. A closed gripper or finger contact alone does not prove a secure grasp. Choose continuous poses, "
    "motion timing and observation cadence yourself.")

CAREFUL_GRASP = (
    "Manipulation guidance: after visually aligning, close at a stationary pose and allow settling. Test each grasp "
    "by lifting only 0.01 m first, using motion_steps of several steps, then hold briefly and request the active "
    "wrist view plus the overview. Keep orientation fixed during this test. Check the RGB views to confirm the "
    "object rises with the gripper before further lifting or transport. If it remains on the table or slips, lower "
    "and adjust local contact depth and centering using current images. There is no fixed limit on failed grasp "
    "attempts. Avoid large instantaneous target jumps while holding an object; use motion_steps for controlled "
    "transport.")


def task_prompt(task: TaskSpec, extra: tuple[str, ...] = ()) -> str:
    """The first user message of an episode: instruction l_tau with the robot constants, then the protocol.

    ``extra`` holds the paragraphs added by the memories (task-memory and working-memory guidance).
    """
    parts = [
        "Complete this single task yourself using the tools: " + task.instruction,
        "Robot: " + task.embodiment.description,
        "You have one continuous conversation and may choose any continuous end-effector target and any waypoint "
        "batching. Use execute_eef sequentially, not parallel calls. Use observations to judge motion and object "
        "manipulation. The environment determines task success independently. Do not start another episode. "
        "Resource limits: unlimited tool calls; the environment enforces its own step limit. Select only useful "
        "camera views after the initial observation. When finished, state the observed outcome concisely.",
    ]
    if task.guidance:
        parts.append(task.guidance)
    parts += [paragraph for paragraph in extra if paragraph]
    return "\n".join(parts)
