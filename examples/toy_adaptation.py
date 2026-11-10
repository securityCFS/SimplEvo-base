"""The method through its Python API, on the toy tabletop with a scripted stand-in for the model.

    python examples/toy_adaptation.py runs/toy_example

Replace ``ToyTabletop`` with a benchmark adapter and ``ScriptedSession(ToyModel())`` with a model backend
(for example ``CodexSession(folder)``) to run the same three phases for real.
"""
from pathlib import Path
import sys

from simplevo import AdaptationChain, adapt, deploy
from simplevo.backends import ScriptedSession
from simplevo.envs.toy import TASK, ToyModel, ToyTabletop

out = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/toy_example")
train_scenes, test_scenes = [0, 1, 2], [100, 101, 102]


def make_session(folder):
    return ScriptedSession(ToyModel())


# Online adaptation: act, reflect, commit, one training scene after another.
chain = AdaptationChain(out / "chain", task=TASK.name)
for record in adapt(chain, train_scenes, ToyTabletop, make_session, TASK, out / "train"):
    print("adaptation episode:", record.attempt())
print("working memory after adaptation:\n" + chain.notes())

# Fix the working memory, then read it, unchanged, in new scenes.
chain.fix(out / "fixed_memory")
for record in deploy(out / "fixed_memory", test_scenes, ToyTabletop, make_session, TASK, out / "test"):
    print("test episode:", record.attempt())
