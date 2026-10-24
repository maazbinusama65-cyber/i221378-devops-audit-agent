#!/usr/bin/env python3
"""DevOps Repository Auditing Agent: CLI entry point.

    python agent.py --repo https://github.com/<owner>/<repo> --output report.json

Runs unattended. Exit codes: 0 = report written, 2 = bad input / repo unreachable,
3 = LLM unavailable or misconfigured, 1 = unexpected internal error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

from auditor import AuditAgent, AuditConfig, Completer
from llm import LLMClient, LLMError
from redact import redact
from repo import RepoError, build_overview, checkout
from tools import RepoTools, ToolError

EXIT_OK, EXIT_ERROR, EXIT_INPUT, EXIT_LLM = 0, 1, 2, 3


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="agent.py", description="AI agent that audits a public GitHub repository for DevOps best practices.")
    parser.add_argument("--repo", required=True, help="public GitHub repository URL")
    parser.add_argument("--output", required=True, help="path of the JSON report to write")
    parser.add_argument("--trace", help="path for the JSONL tool-call trace (default: <output>.trace.jsonl)")
    parser.add_argument("--max-steps", type=int, default=24, help="maximum LLM reasoning steps (default 24)")
    parser.add_argument("--max-tool-calls", type=int, default=70, help="maximum tool executions (default 70)")
    parser.add_argument("--time-budget", type=float, default=900.0, help="wall-clock budget in seconds")
    parser.add_argument("--no-verify", action="store_true", help="skip the second-opinion review of findings")
    parser.add_argument("--quiet", action="store_true", help="suppress progress output on stderr")
    return parser.parse_args(argv)


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # python-dotenv is optional at runtime
        return
    load_dotenv(Path.cwd() / ".env", override=False)
    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def build_report(repository: str, violations: list[dict[str, Any]]) -> dict[str, Any]:
    return {"repository": repository, "violations": violations}


def main(argv: list[str] | None = None,
         llm_factory: Callable[[], Completer] | None = None) -> int:
    args = parse_args(argv)
    _load_dotenv()
    log = (lambda _msg: None) if args.quiet else (lambda msg: print(redact(msg), file=sys.stderr))
    output = Path(args.output)
    trace_path = Path(args.trace) if args.trace else output.with_suffix(output.suffix + ".trace.jsonl")

    try:
        llm = llm_factory() if llm_factory else LLMClient.from_env()
    except LLMError as exc:
        print(f"error: {redact(str(exc))}", file=sys.stderr)
        return EXIT_LLM
    except ImportError as exc:
        print(f"error: missing dependency ({exc}); run `pip install -r requirements.txt`", file=sys.stderr)
        return EXIT_ERROR

    try:
        with checkout(args.repo) as root:
            tools = RepoTools(root)
            if not tools.files:
                raise RepoError("repository contains no tracked files")
            before = tools.git_status() if tools.is_git else ""
            log(f"auditing {args.repo} ({len(tools.files)} files)")
            config = AuditConfig(max_steps=args.max_steps, max_tool_calls=args.max_tool_calls,
                                 time_budget_s=args.time_budget, verify=not args.no_verify)
            agent = AuditAgent(llm, tools, build_overview(tools, args.repo), config, log=log)
            result = agent.run()
            after = tools.git_status() if tools.is_git else ""
            if before != after:  # the tools are read-only; this would indicate a bug
                result.warnings.append("repository state changed during the audit")
    except RepoError as exc:
        print(f"error: {redact(str(exc))}", file=sys.stderr)
        return EXIT_INPUT
    except LLMError as exc:
        print(f"error: LLM unavailable: {redact(str(exc))}", file=sys.stderr)
        return EXIT_LLM
    except (ToolError, OSError) as exc:
        print(f"error: {redact(str(exc))}", file=sys.stderr)
        return EXIT_ERROR

    report = build_report(args.repo.strip(), result.violations)
    try:
        _write_atomic(output, json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        trace_lines = [json.dumps(entry, ensure_ascii=False) for entry in result.trace]
        _write_atomic(trace_path, "\n".join(trace_lines) + ("\n" if trace_lines else ""))
    except OSError as exc:
        print(f"error: could not write output: {exc}", file=sys.stderr)
        return EXIT_ERROR
    for warning in result.warnings:
        print(f"warning: {redact(warning)}", file=sys.stderr)
    log(f"done: {len(result.violations)} violation(s), {result.steps} steps, "
        f"{result.tool_calls} tool calls -> {output}")
    return EXIT_OK


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        sys.exit(130)
