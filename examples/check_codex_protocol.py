"""Check the installed CLI's handshake and dynamic-tool registration without model inference.

    python examples/check_codex_protocol.py --out runs/codex-check --model gpt-6-astra --effort high

The CLI must be installed and configured with access to the selected model.
The check opens an ephemeral thread, sends no turn and closes its own app-server.
"""
import argparse
import json
from pathlib import Path

from simplevo.backends import CodexSession
from simplevo.control import tool_specs
from simplevo.envs.toy import TASK


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--effort", default="high")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("use a new output directory")
    session = CodexSession(args.out, model=args.model, effort=args.effort, rpc_timeout_s=20)
    try:
        session.start(tool_specs(TASK.embodiment), "Use only the supplied robot tools.")
        result = {"thread_started": bool(session.thread_id), "model": args.model, "effort": args.effort,
                  "declared_tools": 3, "model_turns": 0}
    finally:
        session.close()
    (args.out / "protocol_check.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
