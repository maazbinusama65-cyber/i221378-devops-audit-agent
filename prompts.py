"""Prompt construction for the audit agent.

Nothing here is specific to any repository: the prompt teaches the *method*
(observe, hypothesise, verify with a tool, record evidence) and the evidence rules.
"""

from __future__ import annotations

import json

from schema import CATEGORIES, RULE_CATALOG, SEVERITIES

PROTOCOL = """\
Every reply you send MUST be a single JSON object and nothing else:
{
  "thought": "what you observed in the last results and why you are doing the next step",
  "notes": "your running memory: facts learned + what is still unchecked (max ~800 chars)",
  "tool_calls": [ {"tool": "<tool name>", "args": { ... }} ],
  "record_findings": [ {
      "rule_id": "SEC-001", "category": "security", "severity": "high",
      "file": "path/in/repo or null", "line": 12 or null,
      "description": "short statement of the violation",
      "evidence": "what you actually saw (never include secret values)",
      "recommendation": "concrete fix"
  } ],
  "done": false
}
- "notes": REPLACE your notes every turn. Old tool output is trimmed to save context, so your notes
  and the findings ledger are your only long-term memory. Record key facts (languages, package
  manager, CI triggers, what the workflows run, what you already ruled out) and a to-do list.
- "tool_calls": 1-4 independent calls per turn. Omit it (or use []) when finishing.
- "record_findings": add a finding as soon as the evidence is in front of you; the harness
  validates it against the real repository and tells you if it is rejected. Do not re-record
  findings already in the ledger. Omit when there is nothing new.
- Set "done": true only when you have investigated every inspection area that applies and
  recorded everything you can support. Reply with done=true and no tool_calls to finish.
"""

GUIDANCE = f"""\
INSPECTION AREAS: repository hygiene, dependency management, code quality, testing, GitHub
Actions/CI, security, configuration, collaboration, build and release.
Categories to use: {', '.join(CATEGORIES)}. Severities: {', '.join(SEVERITIES)}.

HOW TO INVESTIGATE (derive the plan from what you observe; do not follow a fixed script):
1. Orient: look at the file tree, identify languages, frameworks, package managers, whether the
   project is an application, a library or a service, and what CI/build/test tooling it uses.
2. Form hypotheses from observations and verify each with a tool before recording it. Follow up:
   if a workflow looks odd, read the files it references; if a dependency file exists, compare it
   with the lockfile/other manifests; if a secret scan hits, read the lines to see if it is real.
3. Inspect CI workflows with analyze_workflow, then read_file for anything it flags. Check what
   triggers them, whether tests/lint/build really run and can actually fail the build, token
   permissions, and how untrusted input (PRs, issue text, branch names) reaches `run:` steps.
4. Check git_log for suspicious history (e.g. credential removal commits, giant generated commits).
5. Run scan_secrets and find_tracked_artifacts; treat their output as leads, not conclusions.

EVIDENCE RULES (precision matters as much as recall; false positives are penalised):
- Report a violation only when a tool result you have seen supports it. Cite the file and line.
- An absent optional file or tool is NOT a defect when an equivalent exists: pyproject.toml vs
  requirements.txt; ruff/flake8 config inside pyproject.toml/setup.cfg/tox.ini/pre-commit; tests run
  through tox/nox/make/CI; Poetry/uv/pip-tools lockfiles; Dockerfile-based builds; etc. Libraries
  normally do not commit lockfiles; applications usually should.
- Do NOT claim anything about GitHub branch protection, required reviews, repository settings or
  secrets stored in GitHub: those cannot be seen from files.
- Example/template files (.env.example, test fixtures, documentation snippets, placeholder values
  like "changeme"/"your-key-here", `${{{{ secrets.X }}}}` references) are acceptable, not violations.
- Not defects (do not report): version-tag pinning of GitHub's own actions (actions/*, github/*) or of
  well-known publishers (docker/*); requirements.txt where every dependency is pinned with `==` (that
  already is a reproducible spec: a lockfile is only expected for tools that have one, e.g. Poetry,
  npm, Cargo); a missing optional file when the same purpose is served elsewhere. Only report a
  dependency finding when you can point at the unpinned/inconsistent line.
- Never write or quote a secret value anywhere. Describe it ("hardcoded AWS key in config.py:17").
- Use file=null and line=null for findings about something missing (no tests, no .gitignore).
  One root cause = one finding; do not split or duplicate. Prefer fewer, well-supported findings.
- Severity: critical = exposed live credential or remotely exploitable workflow flaw; high = CI gate
  that cannot fail, unsafe token/permissions, no tests at all, risky deploy automation; medium =
  missing lockfile/lint/gitignore, unpinned dependencies, local config committed; low = hygiene.
- If a tool says UNAVAILABLE or errors, say so in your thought; never invent its result.
- A healthy repository may have zero or very few findings. Returning an empty list is correct then.

RULE IDS (use the closest; the prefix must match the category):
""" + "\n".join(f"  {rule}: {text}" for rule, text in RULE_CATALOG.items())


def build_system_prompt(tool_descriptions: dict[str, dict[str, str]]) -> str:
    tools = "\n".join(
        f"- {name}({', '.join(f'{p}: {d}' for p, d in params.items())})" for name, params in tool_descriptions.items())
    return (
        "You are an autonomous DevOps repository auditor. You inspect a cloned GitHub repository "
        "through read-only tools, decide yourself which tool to use next based on what you have "
        "observed, and record evidence-backed DevOps best-practice violations. You never modify "
        "the repository and never execute its code.\n\n"
        f"TOOLS (call only these):\n{tools}\n\n{PROTOCOL}\n{GUIDANCE}"
    )


def format_observation(overview: str) -> str:
    return (
        "INITIAL OBSERVATION of the repository:\n" + overview +
        "\n\nDecide your first tool calls. Reply with the JSON object described in the protocol."
    )


def format_tool_results(results: list[tuple[str, dict, str]], ledger: str, notes: list[str],
                        steps_left: int, memory: str = "", inspected: list[str] | None = None) -> str:
    blocks = []
    for index, (tool, args, output) in enumerate(results, start=1):
        blocks.append(f"[{index}] {tool}({json.dumps(args, ensure_ascii=False)})\n{output}")
    parts = ["TOOL RESULTS:\n" + "\n\n".join(blocks)] if blocks else []
    if notes:
        parts.append("HARNESS NOTES:\n" + "\n".join(f"- {n}" for n in notes))
    if memory:
        parts.append(f"YOUR NOTES (from your last turn):\n{memory}")
    if inspected:
        parts.append("FILES ALREADY READ OR SEARCHED: " + ", ".join(inspected[:60]))
    parts.append(f"FINDINGS LEDGER:\n{ledger}")
    parts.append(f"Steps remaining: {steps_left}. Reply with the next JSON object.")
    return "\n\n".join(parts)


FINALISE_PROMPT = (
    "The investigation budget is exhausted (or you stopped). Record any remaining findings you can "
    "already support from the evidence above, then finish. Reply with a JSON object with "
    '"record_findings" (only NEW findings) and "done": true. Do not request more tools.'
)

VERIFY_INSTRUCTIONS = (
    "You are the second-opinion reviewer in a DevOps audit. For each candidate finding you get the "
    "claim, the evidence cited and the real source excerpt. Decide whether the excerpt genuinely "
    "supports the claim under these rules: an equivalent solution makes an absent optional file/tool "
    "acceptable; example/template/placeholder values are not violations; nothing may be inferred about "
    "GitHub branch-protection settings; unsupported or speculative claims must be dropped.\n"
    'Reply ONLY with JSON: {"verdicts": [{"id": <int>, "keep": true|false, "reason": "<short>"}]}. '
    "Keep a finding unless the excerpt clearly contradicts it or the claim is speculative."
)
