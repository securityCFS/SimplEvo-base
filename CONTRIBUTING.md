# Contributing

SimplEvo-base contains the benchmark-independent control and memory loop. Keep benchmark-specific adapters,
credentials, model-generated memories and experiment outputs outside the package.

## Development

Use Python 3.10 or newer. From a source checkout:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest -q
ruff check simplevo tests examples
python -m simplevo toy --out runs/toy
```

The tests and toy example run locally on CPU without a model, simulator or login. Use a new output directory for
each run. Existing episode folders are retained as evidence and cannot be silently replayed.

The Codex backend tests use `tests/fake_codex.py`, a local JSON-RPC server. They cover content conversion,
model-catalog pagination, turn completion, interruption and process cleanup. They do not measure model quality
or establish compatibility with every Codex CLI version. See [the adapter guide](docs/ADAPTER.md) for the protocol
and the optional installed-CLI check.

## Changes

Open a pull request describing the problem, the resulting behavior and the checks you ran. Add a regression test
for runtime or data-integrity fixes. Keep prompts and the information protocol explicit: adapters expose RGB and
measured robot state; test scenes must not update the fixed memory.

Each adaptation chain has one writer. Never share a chain between concurrent workers. If a run stops after
executing actions, preserve its episode folder and inspect it before deciding how to continue. Do not delete it
just to make a rerun pass. A partial commit or changed archive is refused on resume and sealing.

When reporting failures, include the command, Python/package versions and a minimal example. Remove credentials,
private observations and account details from logs before sharing them.

## Distribution checks

Follow [the release checklist](docs/RELEASE.md). The source distribution includes examples, documentation and the
offline test server; the wheel contains the library, metadata and license. CI tests both editable and wheel
installations. Publishing a package or benchmark result is a separate maintainer action.
