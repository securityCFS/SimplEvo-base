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
import os
from pathlib import Path
import shutil
import time

from .working_memory import CARRY, NOTES_FILE, NOTES_LIMIT, MemoryRefusal, check_size, episode_calls, key_calls, sha256

ARCHIVE_FILES = ("request.json", "response_summary.json")
FIXED_FILES = (NOTES_FILE, "attempts.jsonl", "view.json")


def link_tree(source: Path, dest: Path) -> None:
    """Hard-link a folder tree so an archive costs no copy; falls back to copying across filesystems."""
    Path(dest).mkdir(parents=True, exist_ok=True)
    for path in sorted(Path(source).rglob("*")):
        target = Path(dest) / path.relative_to(source)
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(path, target)
            except OSError:
                shutil.copyfile(path, target)


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=1) + "\n")


class AdaptationChain:
    """Committed state of one task's adaptation, one view per episode and one commit per episode."""

    def __init__(self, root: Path, *, task: str | None = None, carry: str = "lessons+outcomes",
                 notes_limit: int = NOTES_LIMIT):
        self.root = Path(root)
        meta = self.root / "chain.json"
        if meta.exists():
            self.meta = json.loads(meta.read_text())
            if task is not None and self.meta["task"] != task:
                raise MemoryRefusal(f"{self.root} belongs to task {self.meta['task']}")
        else:
            if carry not in CARRY or not task:
                raise MemoryRefusal(f"a new chain needs a task and a carry setting from {tuple(CARRY)}")
            (self.root / "committed" / "episodes").mkdir(parents=True)
            (self.root / "committed" / NOTES_FILE).write_text("")
            (self.root / "committed" / "attempts.jsonl").write_text("")
            self.meta = {"task": task, "carry": carry, "notes_limit": notes_limit, "created": time.time()}
            _write_json(meta, self.meta)
        self.task, self.carry, self.notes_limit = self.meta["task"], self.meta["carry"], self.meta["notes_limit"]
        self.committed = self.root / "committed"

    def commits(self) -> list[dict]:
        path = self.root / "commits.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []

    def next_position(self) -> int:
        return len(self.commits()) + 1

    def notes(self) -> str:
        return (self.committed / NOTES_FILE).read_text()

    def state_hash(self) -> str:
        digest = hashlib.sha256()
        for path in sorted(p for p in self.committed.rglob("*") if p.is_file()):
            digest.update(str(path.relative_to(self.committed)).encode() + b"\0" + path.read_bytes())
        return digest.hexdigest()

    def build_view(self, position: int, dest: Path) -> dict:
        """The read-only view of episode ``position`` (1-based), filtered by the carry setting. Only the next
        position can be built, and only once."""
        if position != self.next_position():
            raise MemoryRefusal(f"the chain expects position {self.next_position()}, not {position}")
        dest = Path(dest)
        if dest.exists():
            raise MemoryRefusal(f"{dest} exists; a view is built once")
        rules = CARRY[self.carry]
        (dest / "episodes").mkdir(parents=True)
        notes = self.notes() if rules["notes"] else ""
        (dest / NOTES_FILE).write_text(notes)
        (dest / "attempts.jsonl").write_text(
            (self.committed / "attempts.jsonl").read_text() if rules["attempts"] else "")
        visible = []
        if rules["archive"]:
            for source in sorted((self.committed / "episodes").iterdir()):
                link_tree(source, dest / "episodes" / source.name)
                visible.append(int(source.name))
        view = {"task": self.task, "carry": self.carry, "position": position, "visible_positions": visible,
                "notes_limit": self.notes_limit, "parent_hash": self.state_hash(),
                "notes_sha256": sha256(notes.encode())}
        _write_json(dest / "view.json", view)
        return view

    def commit(self, position: int, *, episode: Path, staging: Path, attempt: dict, arms) -> dict:
        """Commit episode ``position`` once: archive what the model received, append the attempt record, and
        take the episode's notes if it wrote any (during the episode or in the reflection turn)."""
        if position != self.next_position():
            raise MemoryRefusal(f"commit out of order: expected position {self.next_position()}, got {position}")
        parent, policy = self.state_hash(), Path(episode) / "policy"
        archive = self.committed / "episodes" / f"{position:03d}" / "policy"
        archive.mkdir(parents=True)
        if (policy / "initial_input").is_dir():
            link_tree(policy / "initial_input", archive / "initial_input")
        for number in episode_calls(policy):
            source, target = policy / "tools" / f"call_{number:05d}", archive / "tools" / f"call_{number:05d}"
            target.mkdir(parents=True)
            for name in ARCHIVE_FILES:
                if (source / name).exists():
                    shutil.copyfile(source / name, target / name)
            if (source / "observation").is_dir():
                link_tree(source / "observation", target / "observation")
        versions = Path(staging) / "versions"
        wrote = versions.is_dir() and any(versions.iterdir())
        if wrote:
            notes = (Path(staging) / NOTES_FILE).read_text()
            check_size(notes, self.notes_limit)
            (self.committed / NOTES_FILE).write_text(notes)
        with (self.committed / "attempts.jsonl").open("a") as stream:
            stream.write(json.dumps({"position": position, **attempt, "key_calls": key_calls(policy, arms)}) + "\n")
        entry = {"position": position, "time": time.time(), "parent_hash": parent, "hash": self.state_hash(),
                 "notes": "rewritten" if wrote else "unchanged"}
        with (self.root / "commits.jsonl").open("a") as stream:
            stream.write(json.dumps(entry) + "\n")
        return entry

    def fix(self, dest: Path) -> dict:
        """Seal the working memory after the last adaptation episode: M^work_K, the notes and attempt records."""
        dest = Path(dest)
        if dest.exists():
            raise MemoryRefusal(f"{dest} exists; a fixed memory is written once")
        dest.mkdir(parents=True)
        position = self.next_position()
        carry = "lessons+outcomes" if CARRY[self.carry]["archive"] else self.carry   # frames are never deployed
        notes = self.notes() if CARRY[carry]["notes"] else ""
        (dest / NOTES_FILE).write_text(notes)
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
    seal = json.loads((path / "fixed_memory.json").read_text())
    for name in FIXED_FILES:
        if (path / name).is_symlink() or sha256((path / name).read_bytes()) != seal["files"][name]:
            raise ValueError(f"fixed memory changed: {name}")
    if sha256(json.dumps(seal["files"], sort_keys=True).encode()) != seal["snapshot_id"]:
        raise ValueError("fixed memory snapshot id mismatch")
    view = json.loads((path / "view.json").read_text())
    attempts = [json.loads(line) for line in (path / "attempts.jsonl").read_text().splitlines() if line]
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
