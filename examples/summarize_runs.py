"""Success rate and mean score of `baseline` or `deploy` runs, averaged per task as in the paper.

    python examples/summarize_runs.py runs/task_a/baseline runs/task_b/baseline

Each folder holds the test episodes of one task (``NNN/policy/result.json``). A task's success rate is its
successes over its episodes. The overall success rate and score are means over tasks, so every task counts once
whatever its number of episodes. Episode folders without a result are listed as unfinished and left out of the
rates, and the summary is then marked incomplete.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def summarize(folders) -> dict:
    tasks = {}
    for folder in map(Path, folders):
        episodes = sorted(path for path in folder.glob("[0-9][0-9][0-9]") if path.is_dir())
        outcomes, unfinished = [], []
        for episode in episodes:
            result = episode / "policy" / "result.json"
            if result.exists():
                outcomes.append(json.loads(result.read_text(encoding="utf-8"))["outcome"])
            else:
                unfinished.append(episode.name)
        if not outcomes:
            raise ValueError(f"{folder}: no finished episodes")
        successes = sum(bool(outcome["success"]) for outcome in outcomes)
        scores = [outcome["score"] for outcome in outcomes if outcome.get("score") is not None]
        tasks[str(folder)] = {
            "episodes": len(outcomes), "successes": successes,
            "success_rate": round(100 * successes / len(outcomes), 2),
            "mean_score": round(sum(scores) / len(scores), 2) if len(scores) == len(outcomes) else None,
            "unfinished": unfinished}
    rates = [task["success_rate"] for task in tasks.values()]
    scores = [task["mean_score"] for task in tasks.values()]
    return {"tasks": tasks,
            "overall": {"tasks": len(tasks), "success_rate": round(sum(rates) / len(rates), 2),
                        "mean_score": round(sum(scores) / len(scores), 2) if None not in scores else None,
                        "complete": not any(task["unfinished"] for task in tasks.values())}}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    print(json.dumps(summarize(sys.argv[1:]), indent=1))
