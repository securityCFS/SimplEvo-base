"""The adaptation chain and the fixed memory (paper Section 3.4).

During online adaptation the episodes of a task run one after another along a chain that starts with an empty
working memory. Before episode k the chain builds a read-only *view* of its committed state; after the episode it
commits exactly once: the attempt record, the episode archive and, if the model rewrote them, the new notes. When
adaptation ends, ``fix`` seals the working memory. Deployment reads that snapshot in every new scene and never
writes to it, so nothing from a deployment scene can enter the memory.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

from .working_memory import CARRY, NOTES_FILE, NOTES_LIMIT, MemoryRefusal, check_size, episode_calls, key_calls, sha256

ARCHIVE_FILES = ("request.json", "response_summary.json")
FIXED_FILES = (NOTES_FILE, "attempts.jsonl", "view.json")


def copy_tree(source: Path, dest: Path) -> None:
    """Copy evidence into independent files; later edits must not mutate a committed archive."""
    shutil.copytree(source, dest)


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=1) + "\n", encoding="utf-8", newline="\n")


class AdaptationChain:
    """Committed state of one task's adaptation, one view per episode and one commit per episode."""

    def __init__(self, root: Path, *, task: str | None = None, carry: str | None = None,
                 notes_limit: int | None = None):
        self.root = Path(root)
        meta = self.root / "chain.json"
        if meta.exists():
            self.meta = json.loads(meta.read_text(encoding="utf-8"))
            if task is not None and self.meta["task"] != task:
                raise MemoryRefusal(f"{self.root} belongs to task {self.meta['task']}")
            if carry is not None and self.meta["carry"] != carry:
                raise MemoryRefusal(f"chain carry setting is {self.meta['carry']!r}, not {carry!r}")
            if notes_limit is not None and self.meta["notes_limit"] != notes_limit:
                raise MemoryRefusal("notes_limit cannot change on an existing chain")
        else:
            carry = "lessons+outcomes" if carry is None else carry
            notes_limit = NOTES_LIMIT if notes_limit is None else notes_limit
            if carry not in CARRY or not task:
                raise MemoryRefusal(f"a new chain needs a task and a carry setting from {tuple(CARRY)}")
            if type(notes_limit) is not int or notes_limit < 1:
                raise MemoryRefusal("notes_limit must be a positive integer")
            (self.root / "committed" / "episodes").mkdir(parents=True)
            (self.root / "committed" / NOTES_FILE).write_text("", encoding="utf-8", newline="\n")
            (self.root / "committed" / "attempts.jsonl").write_text("", encoding="utf-8", newline="\n")
            self.meta = {"task": task, "carry": carry, "notes_limit": notes_limit, "created": time.time()}
            _write_json(meta, self.meta)
        self.task, self.carry, self.notes_limit = self.meta["task"], self.meta["carry"], self.meta["notes_limit"]
        self.committed = self.root / "committed"
        self._validate_state()

    def _validate_state(self) -> None:
        """Refuse interrupted or edited commits instead of sealing or silently resuming them.

        A chain has one writer. Multi-file commits are not a database transaction;
        an interrupted commit requires inspection of the retained episode evidence.
        """
        commits = self.commits()
        positions = list(range(1, len(commits) + 1))
        lines = (self.committed / "attempts.jsonl").read_text(encoding="utf-8").splitlines()
        attempts = [json.loads(line) for line in lines if line]
        archives = sorted(path.name for path in (self.committed / "episodes").iterdir())
        if ([row.get("position") for row in commits] != positions
                or [row.get("position") for row in attempts] != positions
                or archives != sorted(f"{position:03d}" for position in positions)):
            raise MemoryRefusal("incomplete chain commit; inspect the episode before resuming")
        if commits and commits[-1]["hash"] != self.state_hash():
            raise MemoryRefusal("committed chain evidence changed; inspect before resuming")

    def commits(self) -> list[dict]:
        path = self.root / "commits.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line] if path.exists() else []

    def next_position(self) -> int:
        return len(self.commits()) + 1

    def notes(self) -> str:
        return (self.committed / NOTES_FILE).read_text(encoding="utf-8")

    def state_hash(self) -> str:
        digest = hashlib.sha256()
        for path in sorted(p for p in self.committed.rglob("*") if p.is_file()):
            digest.update(str(path.relative_to(self.committed)).encode() + b"\0" + path.read_bytes())
        return digest.hexdigest()

    def build_view(self, position: int, dest: Path) -> dict:
        """The read-only view of episode ``position`` (1-based), filtered by the carry setting. Only the next
        position can be built, and only once."""
        self._validate_state()
        if position != self.next_position():
            raise MemoryRefusal(f"the chain expects position {self.next_position()}, not {position}")
        dest = Path(dest)
        if dest.exists():
            raise MemoryRefusal(f"{dest} exists; a view is built once")
        rules = CARRY[self.carry]
        (dest / "episodes").mkdir(parents=True)
        notes = self.notes() if rules["notes"] else ""
        (dest / NOTES_FILE).write_text(notes, encoding="utf-8", newline="\n")
        (dest / "attempts.jsonl").write_text(
            (self.committed / "attempts.jsonl").read_text(encoding="utf-8") if rules["attempts"] else "",
            encoding="utf-8", newline="\n")
        visible = []
        if rules["archive"]:
            for source in sorted((self.committed / "episodes").iterdir()):
                copy_tree(source, dest / "episodes" / source.name)
                visible.append(int(source.name))
        view = {"task": self.task, "carry": self.carry, "position": position, "visible_positions": visible,
                "notes_limit": self.notes_limit, "parent_hash": self.state_hash(),
                "notes_sha256": sha256(notes.encode())}
        _write_json(dest / "view.json", view)
        return view

    def commit(self, position: int, *, episode: Path, staging: Path, attempt: dict, arms) -> dict:
        """Commit episode ``position`` once: archive what the model received, append the attempt record, and
        take the episode's notes if it wrote any (during the episode or in the reflection turn)."""
        self._validate_state()
        if position != self.next_position():
            raise MemoryRefusal(f"commit out of order: expected position {self.next_position()}, got {position}")
        parent, policy = self.state_hash(), Path(episode) / "policy"
        view = json.loads((Path(episode) / "memory_view" / "view.json").read_text(encoding="utf-8"))
        if (view["parent_hash"] != parent or view["position"] != position
                or view["task"] != self.task or view["carry"] != self.carry):
            raise MemoryRefusal("episode view has a stale parent or belongs to another task or position")
        versions = Path(staging) / "versions"
        wrote = versions.is_dir() and any(versions.iterdir())
        notes = (Path(staging) / NOTES_FILE).read_text(encoding="utf-8") if wrote else self.notes()
        check_size(notes, self.notes_limit)
        attempt = {**attempt, "position": position, "key_calls": key_calls(policy, arms)}
        archive = self.committed / "episodes" / f"{position:03d}" / "policy"
        archive.mkdir(parents=True)
        if (policy / "initial_input").is_dir():
            copy_tree(policy / "initial_input", archive / "initial_input")
        for number in episode_calls(policy):
            source, target = policy / "tools" / f"call_{number:05d}", archive / "tools" / f"call_{number:05d}"
            target.mkdir(parents=True)
            for name in ARCHIVE_FILES:
                if (source / name).exists():
                    shutil.copyfile(source / name, target / name)
            if (source / "observation").is_dir():
                copy_tree(source / "observation", target / "observation")
        if wrote:
            (self.committed / NOTES_FILE).write_text(notes, encoding="utf-8", newline="\n")
        with (self.committed / "attempts.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(attempt) + "\n")
        entry = {"position": position, "time": time.time(), "parent_hash": parent, "hash": self.state_hash(),
                 "notes": "rewritten" if wrote else "unchanged"}
        with (self.root / "commits.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(entry) + "\n")
        return entry

    def fix(self, dest: Path) -> dict:
        """Seal the working memory after the last adaptation episode: M^work_K, the notes and attempt records."""
        self._validate_state()
        dest = Path(dest)
        if dest.exists():
            raise MemoryRefusal(f"{dest} exists; a fixed memory is written once")
        dest.mkdir(parents=True)
        position = self.next_position()
        carry = "lessons+outcomes" if CARRY[self.carry]["archive"] else self.carry   # frames are never deployed
        notes = self.notes() if CARRY[carry]["notes"] else ""
        (dest / NOTES_FILE).write_text(notes, encoding="utf-8", newline="\n")
        shutil.copyfile(self.committed / "attempts.jsonl", dest / "attempts.jsonl")
        _write_json(dest / "view.json", {
            "task": self.task, "carry": carry, "position": position, "visible_positions": [],
            "notes_limit": self.notes_limit, "parent_hash": self.state_hash(), "notes_sha256": sha256(notes.encode())})
        files = {name: sha256((dest / name).read_bytes()) for name in FIXED_FILES}
        seal = {"task": self.task, "fixed_after_position": position - 1, "files": files,
                "snapshot_id": sha256(json.dumps(files, sort_keys=True).encode())}
        _write_json(dest / "fixed_memory.json", seal)
        return seal


def validate_fixed(path: Path) -> dict:
    """Check a fixed memory against its seal; raises if any file changed."""
    path = Path(path)
    seal = json.loads((path / "fixed_memory.json").read_text(encoding="utf-8"))
    if set(seal["files"]) != set(FIXED_FILES):
        raise ValueError("fixed memory file manifest mismatch")
    for name in FIXED_FILES:
        if (path / name).is_symlink() or sha256((path / name).read_bytes()) != seal["files"][name]:
            raise ValueError(f"fixed memory changed: {name}")
    if sha256(json.dumps(seal["files"], sort_keys=True).encode()) != seal["snapshot_id"]:
        raise ValueError("fixed memory snapshot id mismatch")
    view = json.loads((path / "view.json").read_text(encoding="utf-8"))
    if view["task"] != seal["task"]:
        raise ValueError("fixed memory task mismatch")
    if view["carry"] not in CARRY or CARRY[view["carry"]]["archive"]:
        raise ValueError("fixed memory has an invalid carry setting")
    if view["notes_sha256"] != seal["files"][NOTES_FILE]:
        raise ValueError("fixed memory notes hash mismatch")
    attempts = [json.loads(line) for line in (path / "attempts.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if view["position"] != seal["fixed_after_position"] + 1 or view["visible_positions"] or \
            [row["position"] for row in attempts] != list(range(1, seal["fixed_after_position"] + 1)):
        raise ValueError("fixed memory does not match its adaptation boundary")
    return seal


def fixed_view(snapshot: Path, dest: Path) -> dict:
    """A private copy of a fixed memory for one deployment episode (the snapshot itself is never opened for
    writing)."""
    seal = validate_fixed(snapshot)
    dest = Path(dest)
    (dest / "episodes").mkdir(parents=True)
    for name in FIXED_FILES:
        shutil.copyfile(Path(snapshot) / name, dest / name)
    return seal
