"""Command line: ``python -m simplevo <command>``.

    toy        run the whole method offline on the toy tabletop with a scripted stand-in for the model
    annotate   write a task memory from one demonstration (needs a model)
    adapt      online adaptation on training scenes (needs a model and an environment adapter)
    fix        seal the working memory of a chain
    deploy     run new scenes with a fixed working memory, or with none
    baseline   direct control without working memory: no reflection, nothing carried between episodes. With
               --task-memory and --demo it is the paper's direct-control baseline; without them the model gets
               the instruction alone

``--env`` names an environment factory as ``package.module:function``; it is called with one scene identifier and
returns a ``simplevo.task.Environment``. ``--task`` is a JSON file::

    {"name": ..., "instruction": ..., "guidance": "",
     "embodiment": {"description": ..., "arms": [...], "cameras": [...], "has_base": false}}
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from .agent import adapt, deploy
from .annotate import annotate
from .backends import CodexSession, ScriptedSession
from .chain import AdaptationChain
from .config import load_config
from .demonstration import Demonstration
from .task import Embodiment, TaskSpec
from .task_memory import TaskMemory
from .working_memory import CARRY


def load_task(path: Path) -> TaskSpec:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    robot = data["embodiment"]
    embodiment = Embodiment(robot["description"], tuple(robot.get("arms", ("left", "right"))),
                            tuple(robot.get("cameras", ("overview", "left_wrist", "right_wrist"))),
                            robot.get("has_base", False), robot.get("max_waypoints", 6),
                            robot.get("max_steps", 64))
    return TaskSpec(data["name"], data["instruction"], embodiment, data.get("guidance", ""))


def load_factory(name: str):
    module, _, function = name.partition(":")
    if not module or not function or ":" in function:
        raise ValueError("environment factory must have the form package.module:function")
    factory = getattr(importlib.import_module(module), function)
    if not callable(factory):
        raise ValueError("environment factory must be callable")
    return factory


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def summary(records) -> dict:
    return {"episodes": len(records), "successes": sum(bool(r.outcome and r.outcome.success) for r in records),
            "outcomes": [r.attempt() for r in records]}


def _model(parser):
    parser.add_argument("--model", default=None)
    parser.add_argument("--effort", default=None)


def _episodes(parser):
    parser.add_argument("--env", required=True, help="environment factory, package.module:function")
    parser.add_argument("--task", type=Path, required=True, help="task JSON file")
    parser.add_argument("--scenes", nargs="+", required=True, help="scene identifiers, in order")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--task-memory", type=Path, help="folder written by `annotate`")
    parser.add_argument("--demo", type=Path, help="demonstration folder the model may revisit")
    _model(parser)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="simplevo", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    toy = commands.add_parser("toy")
    toy.add_argument("--out", type=Path, required=True)
    toy.add_argument("--train", type=positive_int, default=None, help="training scenes")
    toy.add_argument("--test", type=positive_int, default=None, help="test scenes")
    note = commands.add_parser("annotate")
    note.add_argument("--demo", type=Path, required=True)
    note.add_argument("--task", type=Path, required=True)
    note.add_argument("--out", type=Path, required=True)
    _model(note)
    train = commands.add_parser("adapt")
    _episodes(train)
    train.add_argument("--chain", type=Path, required=True)
    train.add_argument("--carry", choices=tuple(CARRY),
                       help="new chains default to lessons+outcomes; existing chains retain their setting")
    train.add_argument("--hide-outcome", action="store_true", default=None, help="reflection without the evaluator's result")
    seal = commands.add_parser("fix")
    seal.add_argument("--chain", type=Path, required=True)
    seal.add_argument("--out", type=Path, required=True)
    test = commands.add_parser("deploy")
    _episodes(test)
    test.add_argument("--memory", type=Path, help="fixed working memory written by `fix`; omit for none")
    raw = commands.add_parser("baseline", help="direct control without working memory")
    _episodes(raw)
    for command in (toy, note, train, seal, test, raw):
        command.add_argument("--config", type=Path, help="YAML overrides; see simplevo/defaults.yaml")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if hasattr(args, "model") and args.model is not None:
        config["model"]["name"] = args.model
    if hasattr(args, "effort") and args.effort is not None:
        config["model"]["effort"] = args.effort
    if getattr(args, "hide_outcome", None):
        config["episode"]["show_outcome"] = False

    def new_chain(path, task):
        # Reopening without a YAML file preserves the chain's persisted budgets.
        configured = args.config is not None or not (path / "chain.json").exists()
        return AdaptationChain(path, task=task,
                               carry=getattr(args, "carry", None) or (config["memory"]["carry"] if configured else None),
                               notes_limit=config["memory"]["notes_limit"] if configured else None)

    if args.command == "toy":
        from .envs.toy import TASK, ToyModel, ToyTabletop

        def scripted(_):
            return ScriptedSession(ToyModel())
        args.train = args.train if args.train is not None else config["toy"]["train"]
        args.test = args.test if args.test is not None else config["toy"]["test"]
        config["toy"].update(train=args.train, test=args.test)
        chain = new_chain(args.out / "chain", TASK.name)
        before = deploy(None, range(1000, 1000 + args.test), ToyTabletop, scripted, TASK, args.out / "test_before", config=config)
        trained = adapt(chain, range(args.train), ToyTabletop, scripted, TASK, args.out / "train", config=config)
        chain.fix(args.out / "fixed_memory")
        after = deploy(args.out / "fixed_memory", range(1000, 1000 + args.test), ToyTabletop, scripted, TASK,
                       args.out / "test_after", config=config)
        print(json.dumps({"test_without_working_memory": summary(before), "adaptation": summary(trained),
                          "test_with_fixed_working_memory": summary(after), "working_memory": chain.notes()},
                         indent=1))
        return 0
    if args.command == "fix":
        print(json.dumps(AdaptationChain(args.chain).fix(args.out), indent=1))
        return 0

    task = load_task(args.task)

    def session(folder):
        return CodexSession(folder, model=config["model"]["name"], effort=config["model"]["effort"],
                            rpc_timeout_s=config["model"]["rpc_timeout_s"],
                            turn_timeout_s=config["model"]["turn_timeout_s"])
    if args.command == "annotate":
        model = session(args.out / "session")
        try:
            annotate(model, Demonstration.load(args.demo), task.embodiment, args.out, config=config)
        finally:
            model.close()
        print(json.dumps({"task_memory": str(args.out / "task_memory.json")}))
        return 0
    options = {"config": config, "task_memory": TaskMemory.load(args.task_memory) if args.task_memory else None,
               "demonstration": Demonstration.load(args.demo) if args.demo else None}
    factory = load_factory(args.env)
    if args.command == "adapt":
        chain = new_chain(args.chain, task.name)
        records = adapt(chain, args.scenes, factory, session, task, args.out,
                        **options)
    else:
        records = deploy(getattr(args, "memory", None), args.scenes, factory, session, task, args.out, **options)
    print(json.dumps(summary(records), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
