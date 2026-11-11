# Preparing a release

This repository releases a reusable method implementation. The toy example validates the data flow with a
scripted model; it does not reproduce the paper's benchmark results. Keep those two kinds of evidence separate.

## Check the source

1. Work on a release branch and review the diff. Run the offline tests and lint checks in `CONTRIBUTING.md`.
2. Run the toy command and `examples/toy_adaptation.py` with separate, new output directories. Confirm that the
   fixed snapshot validates after deployment and that every episode has a terminal `policy/result.json`.
3. Review the files that will be published. Keep credentials, machine-specific paths, private data, generated
   memories, logs and run folders out of the release. The existing paper figures and video are included in the
   source distribution; confirm their attribution and redistribution rights when replacing them.
4. Keep the version in `pyproject.toml` and `simplevo/__init__.py` consistent. Update `CHANGELOG.md` and the citation
   if the release or paper metadata changes.

## Build and install

```bash
python -m build
python -m twine check dist/*
python3 -m venv .venv-wheel
.venv-wheel/bin/python -m pip install dist/*.whl "pytest>=8,<10"
```

Test the wheel outside the checkout so source files cannot hide a missing package file:

```bash
repo="$PWD"
mkdir -p runs/wheel-check
cd runs/wheel-check
"$repo/.venv-wheel/bin/python" -m pytest -c /dev/null "$repo/tests" -q
"$repo/.venv-wheel/bin/simplevo" toy --out toy
"$repo/.venv-wheel/bin/python" "$repo/examples/toy_adaptation.py" toy-api
```

Run these commands in a new checkout or with new output paths. CI also builds a wheel from the source archive to
verify that it contains the offline server, examples and documentation. Neither workflow uploads a release.

## Record the verification boundary

Record the commit, operating system, Python and dependency versions, test result, package checksums and tested
Codex CLI version. Distinguish an installed-CLI protocol check from a live model episode. Model-backed benchmark
validation needs a benchmark adapter, data and model access; those are not shipped in this repository.

After reviewing the candidate, maintainers choose the tag, merge and publication destination. A local build or
passing offline suite does not mean a tag, GitHub release or PyPI package has been published.
