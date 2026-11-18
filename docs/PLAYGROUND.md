# Panda pick-and-place playground

A Franka Panda picks up a red cube and releases it on a green target. The environment uses PyBullet contact
physics, joint motors and inverse kinematics; both RGB cameras render on the CPU in headless mode. The robot
and mesh assets come from the `pybullet_data` package installed with PyBullet.

## Run the offline case

Use Python 3.11 for the prebuilt PyBullet wheel:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[playground]"
python -m simplevo.envs.playground --scene 0 --out runs/panda
```

Open `runs/panda/preview.gif` to see the trajectory. The command also writes:

- `result.png` and `result.json`: final view and terminal outcome;
- `task.json`: the task and robot description for the standard CLI;
- `demo/`: camera frames and measured end-effector/gripper trajectories, ready for annotation;
- `episode/policy/`: the tool requests, observations and result produced by the regular episode runner;
- `simulation.yaml`: the physics and rendering settings used for the case.

This command uses a known-layout reference trajectory through the same `execute_eef` and `set_gripper` tools
used by a model. It runs locally without a model connection or credentials. The preview demonstrates the
simulator and tool integration; the existing `simplevo toy` example demonstrates scripted memory adaptation.

## Connect a model

`simplevo.envs.playground:make_env` implements the standard `Environment` factory. It exposes RGB images from
`overview` and `overhead`, plus measured gripper pose and opening. Object coordinates and evaluator state stay
inside the environment. Each scene ID seeds a reproducible layout; use disjoint IDs for adaptation and testing.

The following commands use the model backend configured by the person running them. Run these separately when
ready to connect a model; generating the offline preview never runs them.

```bash
simplevo annotate --demo runs/panda/demo --task runs/panda/task.json --out runs/panda-task-memory
simplevo adapt --env simplevo.envs.playground:make_env --task runs/panda/task.json \
  --scenes 0 1 2 --demo runs/panda/demo --task-memory runs/panda-task-memory \
  --chain runs/panda-chain --out runs/panda-train
simplevo fix --chain runs/panda-chain --out runs/panda-fixed
simplevo deploy --env simplevo.envs.playground:make_env --task runs/panda/task.json \
  --scenes 100 101 102 --task-memory runs/panda-task-memory --memory runs/panda-fixed --out runs/panda-test
# Direct-control baseline on the same test scenes, without working memory
simplevo baseline --env simplevo.envs.playground:make_env --task runs/panda/task.json \
  --scenes 100 101 102 --task-memory runs/panda-task-memory --demo runs/panda/demo --out runs/panda-baseline
```

Model and memory hyperparameters use the existing `--config experiment.yaml` interface. The simulation defaults
are in [`playground.yaml`](../simplevo/envs/playground.yaml). The public tool interface accepts full 3D poses;
for this tabletop task a downward-facing quaternion `[0, 1, 0, 0]` is convenient. The host maps end-effector
commands to Panda joint targets and advances eight physics substeps per control step.

Success requires lifting the cube, releasing it inside the pad, and letting it rest for twelve consecutive
control steps. The environment ends at success or its step limit. Separate instances use separate Bullet
clients; `close()` disconnects only the instance it owns.

## Offline checks

```bash
python -m pip install -e ".[playground,test]"
python -m pytest tests/test_playground.py -q
```

The checks exercise three layouts through the full episode runner, RGB/proprioception outputs, timeout handling,
and independent simulator clients. They use `ScriptedSession`, with no model subprocess or login.

PyBullet API reference: [official quickstart guide](https://github.com/bulletphysics/bullet3/blob/master/docs/pybullet_quickstart_guide/PyBulletQuickstartGuide.md.html).
