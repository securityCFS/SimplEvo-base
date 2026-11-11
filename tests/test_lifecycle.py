"""Failure paths must release owned resources and must never replay an episode."""
import pytest

from simplevo import AdaptationChain, Outcome, adapt, deploy, run_episode
from simplevo.backends import ScriptedSession
from simplevo.envs.toy import TASK, ToyTabletop
from simplevo.working_memory import REFLECTION_CALLS


@pytest.mark.parametrize("phase", ["adapt", "deploy"])
def test_environment_closed_when_session_factory_fails(tmp_path, phase):
    closed = []

    class Env(ToyTabletop):
        def close(self):
            closed.append(True)

    def fail(_):
        raise RuntimeError("session construction failed")

    with pytest.raises(RuntimeError, match="session construction"):
        if phase == "adapt":
            adapt(AdaptationChain(tmp_path / "chain", task=TASK.name), [0], Env, fail, TASK, tmp_path / "run")
        else:
            deploy(None, [0], Env, fail, TASK, tmp_path / "run")
    assert closed == [True]


def test_environment_closed_even_when_session_close_raises(tmp_path):
    closed = []

    class Env(ToyTabletop):
        def close(self):
            closed.append(True)

    class Session(ScriptedSession):
        def close(self):
            raise RuntimeError("close failed")

    with pytest.raises(RuntimeError, match="close failed"):
        deploy(None, [0], Env, lambda _: Session(lambda inputs, call: "done"), TASK, tmp_path / "run")
    assert closed == [True]


def test_existing_deployment_is_rejected_before_opening_resources(tmp_path):
    (tmp_path / "run/001").mkdir(parents=True)
    opened = []

    def factory(scene):
        opened.append(scene)
        return ToyTabletop(scene)

    with pytest.raises(RuntimeError, match="exists|already"):
        deploy(None, [0], factory, lambda _: ScriptedSession(lambda inputs, call: "done"), TASK, tmp_path / "run")
    assert opened == []


def test_stalled_environment_does_not_create_a_terminal_result(tmp_path):
    class Stalled(ToyTabletop):
        def step(self, actions):
            return 0

    with pytest.raises(RuntimeError, match="ended|terminal|progress"):
        run_episode(Stalled(0), ScriptedSession(lambda inputs, call: "done"), TASK, tmp_path / "run")
    assert not (tmp_path / "run/policy/result.json").exists()


def test_environment_may_report_a_terminal_outcome_without_another_step(tmp_path):
    class Terminal(ToyTabletop):
        def step(self, actions):
            self._outcome = Outcome(False, 0)
            return 0

    record = run_episode(Terminal(0), ScriptedSession(lambda inputs, call: "done"), TASK, tmp_path / "run")
    assert record.outcome == Outcome(False, 0) and record.held_steps == 0


def test_reflection_turn_stops_at_its_call_budget(tmp_path):
    calls = []

    def model(inputs, call):
        if inputs[0]["text"].startswith("The episode has ended."):
            for _ in range(REFLECTION_CALLS + 5):
                call("read_task_notes", {})
                calls.append(True)
        return "done"

    chain = AdaptationChain(tmp_path / "chain", task=TASK.name)
    records = adapt(chain, [0], ToyTabletop, lambda _: ScriptedSession(model), TASK, tmp_path / "run")
    assert len(calls) == REFLECTION_CALLS
    assert records[0].reflection["calls"] == REFLECTION_CALLS
