"""The agent loop: Observe -> Reason -> Select tool -> Execute -> Interpret -> Report.

The LLM decides every investigative step. This module only (1) feeds it observations,
(2) executes the tool calls it requests, (3) validates the findings it records against
the real repository, and (4) enforces budgets so the run always terminates.
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from llm import ContextTooLarge, LLMError, Message
from prompts import FINALISE_PROMPT, VERIFY_INSTRUCTIONS, build_system_prompt, format_observation, format_tool_results
from redact import redact
from schema import deduplicate, normalise_violation
from tools import RepoTools


class Completer(Protocol):
    def complete(self, messages: list[Message], max_tokens: int = ...) -> str: ...


# Inspection areas and the tool that evidences having looked at them. Used only to challenge a
# premature "done": the model may still answer that an area does not apply to this repository.
COVERAGE_TOOLS = {
    "tracked temporary/generated/local files and .gitignore suitability": "find_tracked_artifacts",
    "credentials and secrets": "scan_secrets",
    "CI/CD workflows (or the absence of any CI)": "analyze_workflow",
    "commit history": "git_log",
}
MIN_STEPS_BEFORE_DONE = 5
MAX_PUSHBACKS = 2


@dataclass
class AuditConfig:
    max_steps: int = 24
    max_tool_calls: int = 70
    max_calls_per_turn: int = 4
    time_budget_s: float = 1500.0
    context_chars: int = 22_000
    min_context_chars: int = 9_000
    keep_recent_messages: int = 4
    max_parse_failures: int = 3
    verify: bool = True


@dataclass
class AuditResult:
    violations: list[dict[str, Any]]
    trace: list[dict[str, Any]] = field(default_factory=list)
    steps: int = 0
    tool_calls: int = 0
    llm_calls: int = 0
    warnings: list[str] = field(default_factory=list)


def extract_json(text: str) -> dict[str, Any] | None:
    """Pull the first JSON object out of an LLM reply (tolerates code fences/prose)."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    candidates = [fence.group(1)] if fence else []
    candidates.append(text)
    for candidate in candidates:
        start = candidate.find("{")
        while start != -1:
            depth, in_str, escaped = 0, False, False
            for index in range(start, len(candidate)):
                char = candidate[index]
                if in_str:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        in_str = False
                    continue
                if char == '"':
                    in_str = True
                elif char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            parsed = json.loads(candidate[start:index + 1])
                        except json.JSONDecodeError:
                            break
                        return parsed if isinstance(parsed, dict) else None
            start = candidate.find("{", start + 1)
    return None


class AuditAgent:
    def __init__(self, llm: Completer, tools: RepoTools, overview: str,
                 config: AuditConfig | None = None, log: Callable[[str], None] | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.llm = llm
        self.tools = tools
        self.overview = overview
        self.cfg = config or AuditConfig()
        self._log = log or (lambda msg: print(msg, file=sys.stderr))
        self._clock = clock
        self.messages: list[Message] = [
            {"role": "system", "content": build_system_prompt(tools.describe())},
            {"role": "user", "content": format_observation(overview)},
        ]
        self.findings: list[dict[str, Any]] = []
        self.trace: list[dict[str, Any]] = []
        self.warnings: list[str] = []
        self.memory = ""
        self.used_tools: set[str] = set()
        self._pushbacks = 0
        self._tool_calls = 0
        self._call_counts: dict[str, int] = {}

    # ---------------------------------------------------------------------- run
    def run(self) -> AuditResult:
        started = self._clock()
        step = 0
        parse_failures = 0
        notes: list[str] = []
        finished = False
        llm_failed = False
        while step < self.cfg.max_steps and not finished:
            if self._clock() - started > self.cfg.time_budget_s:
                self.warnings.append("time budget exhausted; finishing early")
                break
            if self._tool_calls >= self.cfg.max_tool_calls:
                self.warnings.append("tool-call budget exhausted; finishing early")
                break
            self._compress_context()
            try:
                reply = self._ask()
            except LLMError as exc:
                if step == 0 and not self.findings:
                    raise
                self.warnings.append(f"LLM failed mid-run ({exc}); reporting what was verified so far")
                llm_failed = True
                break
            step += 1
            action = extract_json(reply)
            if action is None:
                parse_failures += 1
                self._log(f"[step {step}] unparseable reply ({parse_failures}/{self.cfg.max_parse_failures})")
                if parse_failures >= self.cfg.max_parse_failures:
                    self.warnings.append("LLM repeatedly returned invalid JSON; finishing early")
                    break
                self.messages.append({"role": "assistant", "content": reply[:2000]})
                self.messages.append({"role": "user", "content": (
                    "Your reply was not a single valid JSON object. Reply again with ONLY the JSON "
                    "object described in the protocol.")})
                continue
            parse_failures = 0
            self.messages.append({"role": "assistant", "content": self._compact_action(action)})

            thought = redact(str(action.get("thought", "")))[:600]
            self._log(f"[step {step}] {thought[:160]}")
            if isinstance(action.get("notes"), str) and action["notes"].strip():
                self.memory = redact(action["notes"].strip())[:1200]
            notes = self._record_findings(action.get("record_findings"), step)
            calls = self._plan_calls(action.get("tool_calls"), notes)
            if action.get("done") is True and not calls:
                challenge = self._premature_done(step)
                if challenge:
                    notes.append(challenge)
                    self.trace.append({"step": step, "event": "done_challenged"})
                    self.messages.append({"role": "user", "content": format_tool_results(
                        [], self._ledger(), notes, self.cfg.max_steps - step, self.memory,
                        sorted(self.tools.inspected))})
                    continue
                finished = True
                self.trace.append({"step": step, "event": "done", "thought": thought})
                break
            if not calls:
                notes.append("You requested no tool calls and did not set done=true. "
                             "Call a tool to continue, or reply with done=true to finish.")
            results = self._execute(calls, step, thought)
            self.messages.append({"role": "user", "content": format_tool_results(
                results, self._ledger(), notes, self.cfg.max_steps - step,
                self.memory, sorted(self.tools.inspected))})

        if not finished and not llm_failed:
            self._finalise(step)
        violations = self._finalise_findings(skip_verify=llm_failed)
        return AuditResult(violations=violations, trace=self.trace, steps=step,
                           tool_calls=self._tool_calls, llm_calls=getattr(self.llm, "calls", 0),
                           warnings=self.warnings)

    @staticmethod
    def _compact_action(action: dict[str, Any]) -> str:
        """Keep the model's own turn in history, but bounded (findings live in the ledger)."""
        compact: dict[str, Any] = {"thought": redact(str(action.get("thought", "")))[:300]}
        if isinstance(action.get("tool_calls"), list):
            compact["tool_calls"] = action["tool_calls"][:4]
        if action.get("record_findings"):
            compact["record_findings"] = f"{len(action['record_findings'])} finding(s) submitted (see ledger)"
        if action.get("done") is True:
            compact["done"] = True
        return redact(json.dumps(compact, ensure_ascii=False))[:1500]

    def _premature_done(self, step: int) -> str | None:
        """Challenge a "done" that comes before several inspection areas were touched."""
        if self._pushbacks >= MAX_PUSHBACKS:
            return None
        missing = [area for area, tool in COVERAGE_TOOLS.items() if tool not in self.used_tools]
        if step >= MIN_STEPS_BEFORE_DONE and len(missing) < 3:
            return None
        if not missing:
            return None
        self._pushbacks += 1
        return ("You declared done, but no tool has yet examined: " + "; ".join(missing) + ". "
                "Decide for each whether it applies to THIS repository. Check what is relevant (a missing "
                "CI setup, tracked generated files, missing tests or lint, unpinned dependencies and "
                "exposed credentials are all findings when the evidence is there), record findings you can "
                "support, and only then finish. If an area truly does not apply, say so in your notes "
                "and reply done=true again.")

    # -------------------------------------------------------------- tool calls
    def _plan_calls(self, raw: Any, notes: list[str]) -> list[tuple[str, dict[str, Any]]]:
        if not isinstance(raw, list):
            return []
        calls: list[tuple[str, dict[str, Any]]] = []
        for item in raw[: self.cfg.max_calls_per_turn]:
            if isinstance(item, dict) and isinstance(item.get("tool"), str):
                args = item.get("args") if isinstance(item.get("args"), dict) else {}
                calls.append((item["tool"], args))
        if len(raw) > self.cfg.max_calls_per_turn:
            notes.append(f"Only the first {self.cfg.max_calls_per_turn} tool calls were executed.")
        return calls

    def _execute(self, calls: list[tuple[str, dict[str, Any]]], step: int,
                 thought: str) -> list[tuple[str, dict[str, Any], str]]:
        results = []
        for tool, args in calls:
            key = f"{tool}:{json.dumps(args, sort_keys=True)}"
            self._call_counts[key] = self._call_counts.get(key, 0) + 1
            if self._call_counts[key] > 2:
                output = ("ERROR: identical call already made twice; its result is above in the "
                          "conversation. Use a different path/range or move on.")
            else:
                self._tool_calls += 1
                self.used_tools.add(tool)
                output = self.tools.dispatch(tool, args)
                limit = max(1500, self.cfg.context_chars // (2 * max(1, len(calls))))
                if len(output) > limit:
                    output = output[:limit] + f"\n... [truncated to {limit} chars for the context budget]"
            self._log(f"    -> {tool}({json.dumps(args)[:100]}) = {len(output)} chars")
            self.trace.append({
                "step": step, "event": "tool_call", "thought": thought, "tool": tool, "args": args,
                "output_chars": len(output), "output_preview": redact(output)[:300],
                "error": output.startswith("ERROR:"),
            })
            results.append((tool, args, output))
        return results

    # ------------------------------------------------------------------ findings
    def _record_findings(self, raw: Any, step: int) -> list[str]:
        notes: list[str] = []
        if raw in (None, []):
            return notes
        if not isinstance(raw, list):
            return ["record_findings must be a list of finding objects"]
        for item in raw:
            violation, error = normalise_violation(item, self.tools)
            if violation is None:
                notes.append(f"Finding REJECTED: {error}. Input was: "
                             f"{redact(json.dumps(item, ensure_ascii=False))[:200]}")
                self.trace.append({"step": step, "event": "finding_rejected", "reason": error})
                continue
            merged = deduplicate([*self.findings, violation])
            if len(merged) == len(self.findings):
                notes.append(f"Finding ignored as duplicate: {violation['description'][:80]}")
                continue
            self.findings = merged
            notes.append(f"Finding recorded: {violation['rule_id']} {violation['file'] or '-'}")
            self.trace.append({"step": step, "event": "finding_recorded", "rule_id": violation["rule_id"],
                               "file": violation["file"], "line": violation["line"],
                               "severity": violation["severity"]})
        return notes

    def _ledger(self) -> str:
        if not self.findings:
            return "(empty: nothing recorded yet)"
        return "\n".join(
            f"- {v['rule_id']} [{v['severity']}] {v['file'] or '-'}:{v['line'] or '-'} {v['description'][:90]}"
            for v in self.findings)

    def _finalise(self, step: int) -> None:
        """Budget ran out (or the loop stalled): ask once for anything still unreported."""
        self.messages.append({"role": "user", "content": FINALISE_PROMPT + "\n\nFINDINGS LEDGER:\n" + self._ledger()})
        try:
            reply = self._ask()
        except LLMError as exc:
            self.warnings.append(f"final LLM call failed: {exc}")
            return
        action = extract_json(reply)
        if action is None:
            self.warnings.append("final reply was not valid JSON; kept the ledger as is")
            return
        for note in self._record_findings(action.get("record_findings"), step + 1):
            self._log(f"    final: {note}")

    # ------------------------------------------------------------ verification
    def _finalise_findings(self, skip_verify: bool) -> list[dict[str, Any]]:
        findings = deduplicate(self.findings)
        if not self.cfg.verify or skip_verify or not findings:
            return findings
        kept: list[dict[str, Any]] = []
        for start in range(0, len(findings), 12):
            batch = findings[start:start + 12]
            try:
                verdicts = self._verify_batch(batch)
            except LLMError as exc:
                self.warnings.append(f"verification pass failed ({exc}); findings kept unverified")
                kept.extend(batch)
                continue
            for index, finding in enumerate(batch):
                verdict = verdicts.get(index)
                if verdict is not None and verdict.get("keep") is False:
                    reason = redact(str(verdict.get("reason", "")))[:160]
                    self.trace.append({"event": "finding_dropped_by_review", "rule_id": finding["rule_id"],
                                       "file": finding["file"], "reason": reason})
                    self._log(f"    review dropped {finding['rule_id']} {finding['file']}: {reason}")
                else:
                    kept.append(finding)
        return kept

    def _excerpt(self, finding: dict[str, Any]) -> str:
        path, line = finding["file"], finding["line"]
        if path is None:
            return "(no file: claim about absence; repository top-level files: " + \
                ", ".join(sorted({f.split('/')[0] for f in self.tools.files})[:60]) + ")"
        text = self.tools._read_text(path)  # noqa: SLF001 - same package, read-only helper
        if text is None:
            return "(file unreadable)"
        lines = text.splitlines()
        lo, hi = (max(1, line - 6), min(len(lines), line + 6)) if line else (1, min(len(lines), 40))
        return redact("\n".join(f"{n}: {lines[n - 1][:200]}" for n in range(lo, hi + 1)))

    def _verify_batch(self, batch: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
        payload = []
        for index, finding in enumerate(batch):
            payload.append({
                "id": index, "rule_id": finding["rule_id"], "claim": finding["description"],
                "evidence_cited": finding["evidence"], "file": finding["file"], "line": finding["line"],
                "excerpt": self._excerpt(finding),
            })
        reply = self.llm.complete([
            {"role": "system", "content": VERIFY_INSTRUCTIONS},
            {"role": "user", "content": json.dumps({"findings": payload}, ensure_ascii=False)},
        ], max_tokens=1500)
        parsed = extract_json(reply) or {}
        verdicts = parsed.get("verdicts")
        out: dict[int, dict[str, Any]] = {}
        if isinstance(verdicts, list):
            for item in verdicts:
                if isinstance(item, dict) and isinstance(item.get("id"), int):
                    out[item["id"]] = item
        return out

    # ----------------------------------------------------------- context budget
    def _ask(self) -> str:
        """Query the LLM; if the request is too large, shrink the context and retry."""
        for _attempt in range(4):
            self._compress_context()
            try:
                return self.llm.complete(self.messages)
            except ContextTooLarge as exc:
                new_limit = max(self.cfg.min_context_chars, int(self.cfg.context_chars * 0.6))
                if new_limit >= self.cfg.context_chars:
                    raise
                self.warnings.append(f"context limit hit; shrinking history to {new_limit} chars")
                self._log(f"    context too large ({str(exc)[:60]}); shrinking to {new_limit} chars")
                self.cfg.context_chars = new_limit
        raise ContextTooLarge("request still too large after shrinking the history")

    def _compress_context(self) -> None:
        """Trim old tool output (oldest first) until the transcript fits the context budget."""

        def total() -> int:
            return sum(len(m["content"]) for m in self.messages)

        def trim(index: int, keep: int) -> None:
            message = self.messages[index]
            content = message["content"]
            if message["role"] != "user" or len(content) <= keep + 200 or "FINDINGS LEDGER:" not in content:
                return
            tail = content[content.rfind("FINDINGS LEDGER:"):] if "FINDINGS LEDGER:" in content else ""
            message["content"] = content[:keep] + "\n... [older tool output trimmed to save context] ...\n" + tail

        if total() <= self.cfg.context_chars:
            return
        recent_from = max(2, len(self.messages) - self.cfg.keep_recent_messages)
        for index in range(2, recent_from):
            trim(index, 400)
            if total() <= self.cfg.context_chars:
                return
        for index in range(recent_from, len(self.messages) - 1):
            trim(index, 1500)
            if total() <= self.cfg.context_chars:
                return
        trim(len(self.messages) - 1, max(2000, self.cfg.context_chars // 3))
        # Last resort: forget the oldest exchange (assistant+user pair); the ledger keeps the findings.
        while total() > self.cfg.context_chars and len(self.messages) > 4:
            del self.messages[2:4]
