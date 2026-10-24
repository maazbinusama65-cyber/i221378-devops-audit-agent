"""Structured, line-aware reading of GitHub Actions workflow files.

The summary is *evidence for the LLM*, not a verdict: it lists the structure of
a workflow plus neutral facts ("no permissions block", "step ends in `|| true`").
The agent decides whether a fact is a genuine violation.
"""

from __future__ import annotations

import re
from typing import Any

import yaml

TEST_COMMAND_HINT = re.compile(
    r"\b(pytest|py\.test|unittest|tox|nox|npm (?:run )?test|npm t\b|yarn test|pnpm (?:run )?test|"
    r"go test|cargo test|mvn(?:w)? .*\b(?:test|verify)\b|gradle(?:w)? .*\btest\b|dotnet test|"
    r"rspec|bundle exec rake test|phpunit|make (?:test|check)|ctest|vitest|jest|mocha)\b",
    re.IGNORECASE,
)
LINT_COMMAND_HINT = re.compile(
    r"\b(ruff|flake8|pylint|black|isort|mypy|eslint|prettier|golangci-lint|clippy|rubocop|"
    r"pre-commit|stylelint|tsc)\b",
    re.IGNORECASE,
)
BUILD_COMMAND_HINT = re.compile(
    r"\b(docker build|docker/build-push-action|python -m build|npm run build|yarn build|"
    r"pnpm build|go build|cargo build|mvn(?:w)? .*package|gradle(?:w)? .*build|make build|"
    r"poetry build|setup\.py)\b",
    re.IGNORECASE,
)
INJECTION_CONTEXT = re.compile(
    r"\$\{\{\s*github\.(?:event\.(?:issue|pull_request|comment|review|head_commit|commits|"
    r"discussion|pages|workflow_run)\.[\w.\[\]*]*(?:title|body|message|name|label|ref|email|"
    r"default_branch)|head_ref)\s*\}\}"
)
PINNED_SHA = re.compile(r"^[0-9a-f]{40}$")


class LDict(dict):
    """dict that remembers the source line of the mapping and of each key."""

    line: int = 0

    def __init__(self) -> None:
        super().__init__()
        self.klines: dict[Any, int] = {}


class LList(list):
    line: int = 0


def _construct(node: yaml.Node) -> Any:
    if isinstance(node, yaml.MappingNode):
        out = LDict()
        out.line = node.start_mark.line + 1
        for key_node, value_node in node.value:
            key = _construct(key_node)
            out[key] = _construct(value_node)
            out.klines[key] = key_node.start_mark.line + 1
        return out
    if isinstance(node, yaml.SequenceNode):
        seq = LList(_construct(item) for item in node.value)
        seq.line = node.start_mark.line + 1
        return seq
    loader = yaml.SafeLoader("")
    try:
        return loader.construct_object(node, deep=True)
    finally:
        loader.dispose()


def parse_workflow(text: str) -> LDict:
    loader = yaml.SafeLoader(text)
    try:
        node = loader.get_single_node()
    finally:
        loader.dispose()
    if node is None:
        raise ValueError("workflow file is empty")
    data = _construct(node)
    if not isinstance(data, LDict):
        raise ValueError("workflow root is not a mapping")
    return data


def _triggers(doc: LDict) -> tuple[Any, int]:
    # PyYAML (YAML 1.1) parses the bare key `on` as boolean True.
    for key in ("on", True):
        if key in doc:
            return doc[key], doc.klines.get(key, doc.line)
    return None, doc.line


def _describe_triggers(trig: Any) -> str:
    if trig is None:
        return "(none declared)"
    if isinstance(trig, str):
        return trig
    if isinstance(trig, list):
        return ", ".join(str(t) for t in trig)
    if isinstance(trig, dict):
        parts = []
        for name, cfg in trig.items():
            if isinstance(cfg, dict) and cfg:
                details = "; ".join(f"{k}={v}" for k, v in cfg.items())
                parts.append(f"{name} ({details})")
            else:
                parts.append(str(name))
        return ", ".join(parts)
    return str(trig)


def _trigger_names(trig: Any) -> set[str]:
    if isinstance(trig, str):
        return {trig}
    if isinstance(trig, list):
        return {str(t) for t in trig}
    if isinstance(trig, dict):
        return {str(t) for t in trig}
    return set()


def _first_line(value: str, limit: int = 110) -> str:
    line = value.strip().splitlines()[0] if value.strip() else ""
    return line if len(line) <= limit else line[: limit - 3] + "..."


def _truthy_false(value: Any) -> bool:
    return value is False or (isinstance(value, str) and value.strip().lower() in {"false", "0"})


def summarize_workflow(path: str, text: str) -> str:
    """Render a compact structural summary plus neutral observations."""
    try:
        doc = parse_workflow(text)
    except (yaml.YAMLError, ValueError) as exc:
        return f"{path}: could not be parsed as YAML ({str(exc).splitlines()[0]}). Read it with read_file."

    facts: list[str] = []
    lines: list[str] = [f"WORKFLOW {path}  name={doc.get('name')!r}"]

    trig, trig_line = _triggers(doc)
    lines.append(f"  triggers (line {trig_line}): {_describe_triggers(trig)}")
    names = _trigger_names(trig)
    if "pull_request_target" in names:
        facts.append(f"L{trig_line}: uses the pull_request_target trigger (runs with base-repo secrets/write token)")
    if "workflow_run" in names:
        facts.append(f"L{trig_line}: uses the workflow_run trigger (privileged context)")
    if isinstance(trig, dict):
        for name in ("push", "pull_request"):
            cfg = trig.get(name)
            if isinstance(cfg, dict):
                for filt in ("branches", "branches-ignore", "paths", "paths-ignore", "types"):
                    if filt in cfg:
                        lines.append(f"    {name}.{filt} = {cfg[filt]}")
                if "branches" in cfg and not cfg["branches"]:
                    facts.append(f"L{trig_line}: {name}.branches is empty, so the trigger may never match")

    top_perm = doc.get("permissions", "<absent>")
    perm_line = doc.klines.get("permissions")
    lines.append(f"  top-level permissions: {top_perm if top_perm != '<absent>' else '(absent)'}")
    if top_perm in ("write-all", "write"):
        facts.append(f"L{perm_line}: top-level permissions is {top_perm!r} (grants every scope write access)")
    elif isinstance(top_perm, dict):
        for scope, level in top_perm.items():
            if level == "write" and scope in {"contents", "packages", "actions", "id-token"}:
                facts.append(f"L{perm_line}: permissions.{scope}=write at workflow level")

    env = doc.get("env")
    if isinstance(env, dict):
        for key, value in env.items():
            if isinstance(value, str) and re.search(r"(?i)(token|secret|password|key)", str(key)) \
                    and "${{" not in value and len(value) >= 8:
                facts.append(f"L{doc.klines.get('env')}: env {key} holds a literal value, not a secrets reference")

    jobs = doc.get("jobs")
    any_tests = any_lint = any_build = False
    job_count = 0
    if isinstance(jobs, dict):
        for job_id, job in jobs.items():
            if not isinstance(job, dict):
                continue
            job_count += 1
            jline = jobs.klines.get(job_id, 0)
            lines.append(f"  job {job_id!r} (line {jline}) runs-on={job.get('runs-on')} "
                         f"needs={job.get('needs')} if={job.get('if')!r}")
            if "permissions" in job:
                lines.append(f"    job permissions: {job['permissions']}")
                if job["permissions"] in ("write-all",):
                    facts.append(f"L{job.klines.get('permissions')}: job {job_id} has permissions write-all")
            if _truthy_false(job.get("if")):
                facts.append(f"L{job.klines.get('if')}: job {job_id} is disabled with `if: false`")
            if job.get("continue-on-error") in (True, "true"):
                facts.append(f"L{job.klines.get('continue-on-error')}: job {job_id} has continue-on-error: true")
            steps = job.get("steps")
            if not isinstance(steps, list):
                continue
            for step in steps:
                if not isinstance(step, dict):
                    continue
                sline = step.line
                uses, run = step.get("uses"), step.get("run")
                label = f"uses {uses}" if uses else f"run: {_first_line(str(run))}" if run else "(step)"
                extras = []
                if step.get("name"):
                    extras.append(f"name={step['name']!r}")
                if "if" in step:
                    extras.append(f"if={step['if']!r}")
                if step.get("continue-on-error") in (True, "true"):
                    extras.append("continue-on-error=true")
                lines.append(f"    L{sline}: {label} {' '.join(extras)}".rstrip())

                if uses:
                    _inspect_uses_step(step, str(uses), job_id, facts, sline)
                if isinstance(run, str):
                    any_tests |= bool(TEST_COMMAND_HINT.search(run))
                    any_lint |= bool(LINT_COMMAND_HINT.search(run))
                    any_build |= bool(BUILD_COMMAND_HINT.search(run))
                    _inspect_run_step(run, job_id, facts, sline)
                if uses:
                    any_build |= bool(BUILD_COMMAND_HINT.search(str(uses)))
                    any_lint |= bool(LINT_COMMAND_HINT.search(str(uses)))
                if _truthy_false(step.get("if")):
                    facts.append(f"L{step.line}: step is disabled with `if: false` ({label[:60]})")
                if step.get("continue-on-error") in (True, "true"):
                    gate = TEST_COMMAND_HINT.search(str(run or "")) or LINT_COMMAND_HINT.search(str(run or ""))
                    facts.append(
                        f"L{step.klines.get('continue-on-error', sline)}: continue-on-error: true on step "
                        f"({label[:60]}){' - this step is a test/lint gate' if gate else ''}")
            if "permissions" not in job and top_perm == "<absent>":
                lines.append("    (no permissions block at job or workflow level)")

    if top_perm == "<absent>":
        has_job_perms = isinstance(jobs, dict) and all(
            isinstance(j, dict) and "permissions" in j for j in jobs.values()) and job_count > 0
        if not has_job_perms:
            facts.append("no `permissions:` block at workflow level (GITHUB_TOKEN gets the repo default scope)")

    lines.append(f"  detected in run/uses commands: tests={any_tests} lint={any_lint} build={any_build}")
    lines.append("  OBSERVATIONS (facts for you to judge, not verdicts):")
    lines.extend(f"    - {fact}" for fact in facts) if facts else lines.append("    - none")
    return "\n".join(lines)


def _inspect_uses_step(step: LDict, uses: str, job_id: str, facts: list[str], line: int) -> None:
    if uses.startswith("./") or uses.startswith("docker://"):
        return
    if "@" in uses:
        action, ref = uses.rsplit("@", 1)
        if ref in {"main", "master", "latest", "HEAD"}:
            facts.append(f"L{line}: action {action} is pinned to the moving ref @{ref}")
        elif not PINNED_SHA.match(ref) and not action.startswith(("actions/", "github/")):
            facts.append(f"L{line}: third-party action {action}@{ref} is not pinned to a commit SHA")
    else:
        facts.append(f"L{line}: action {uses} has no version ref")
    if uses.startswith("actions/checkout"):
        with_block = step.get("with")
        if isinstance(with_block, dict):
            ref = str(with_block.get("ref", ""))
            if "pull_request" in ref or "head" in ref:
                facts.append(f"L{line}: checkout of an untrusted PR ref ({ref}); dangerous under pull_request_target")
            if with_block.get("persist-credentials") in (True, "true"):
                facts.append(f"L{line}: checkout sets persist-credentials: true")


def _inspect_run_step(run: str, job_id: str, facts: list[str], line: int) -> None:
    for raw in run.splitlines():
        cmd = raw.strip()
        if not cmd or cmd.startswith("#"):
            continue
        is_gate = TEST_COMMAND_HINT.search(cmd) or LINT_COMMAND_HINT.search(cmd)
        if is_gate and re.search(r"\|\|\s*(true|echo|exit 0|:)", cmd):
            facts.append(f"L{line}: gate command swallows failure: `{_first_line(cmd, 90)}`")
        if is_gate and re.search(r"(--exit-zero|--no-fail|--passWithNoTests|--collect-only)", cmd):
            facts.append(f"L{line}: gate command is configured not to fail the build: `{_first_line(cmd, 90)}`")
        if re.search(r"curl[^|\n]*\|\s*(sudo\s+)?(ba)?sh", cmd) or re.search(r"wget[^|\n]*\|\s*(sudo\s+)?(ba)?sh", cmd):
            facts.append(f"L{line}: pipes a remote script into a shell: `{_first_line(cmd, 90)}`")
        if re.search(r"git (push|commit).*(--force|-f\b)", cmd):
            facts.append(f"L{line}: force-push in workflow: `{_first_line(cmd, 90)}`")
        if re.search(r"gh pr merge|--auto|automerge", cmd, re.IGNORECASE):
            facts.append(f"L{line}: automated merge command: `{_first_line(cmd, 90)}`")
        if re.search(r"echo\s+.*\$\{\{\s*secrets\.", cmd) or re.search(r"(printenv|env\s*$|set -x)", cmd):
            facts.append(f"L{line}: command may print secrets/environment to logs: `{_first_line(cmd, 90)}`")
    if INJECTION_CONTEXT.search(run):
        facts.append(f"L{line}: untrusted github.event field interpolated directly inside `run:` (script injection)")
