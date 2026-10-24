"""Violation schema: rule catalog, normalisation and de-duplication."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from redact import redact

CATEGORIES = (
    "repository_hygiene", "dependency_management", "code_quality", "testing", "ci",
    "security", "configuration", "collaboration", "build_release",
)
SEVERITIES = ("critical", "high", "medium", "low")
_SEVERITY_RANK = {name: i for i, name in enumerate(SEVERITIES)}

CATEGORY_PREFIX = {
    "repository_hygiene": "REPO", "dependency_management": "DEP", "code_quality": "QUAL",
    "testing": "TEST", "ci": "CI", "security": "SEC", "configuration": "CFG",
    "collaboration": "COLLAB", "build_release": "BUILD",
}
_CATEGORY_ALIASES = {
    "hygiene": "repository_hygiene", "repository": "repository_hygiene", "repo": "repository_hygiene",
    "dependency": "dependency_management", "dependencies": "dependency_management",
    "quality": "code_quality", "lint": "code_quality", "test": "testing", "tests": "testing",
    "github_actions": "ci", "ci_cd": "ci", "cicd": "ci", "ci/cd": "ci", "actions": "ci",
    "config": "configuration", "build": "build_release", "release": "build_release",
    "build_and_release": "build_release", "build-release": "build_release",
}

# Stable rule ids the LLM is steered towards (kept short so the prompt stays small).
RULE_CATALOG: dict[str, str] = {
    "REPO-001": "README missing or empty",
    "REPO-002": ".gitignore missing or unsuitable for the project's languages/tools",
    "REPO-003": "temporary, generated or environment-specific files are tracked in git",
    "DEP-001": "dependency specifications are inconsistent (manifest vs requirements vs lockfile)",
    "DEP-002": "no dependency manifest, or a lockfile-based tool is used with its lockfile missing",
    "DEP-003": "weak or missing version constraints on dependencies",
    "QUAL-001": "no linting/formatting configuration or enforcement where none is equivalent",
    "QUAL-002": "obvious maintainability problem (bare except, dead/debug code, giant functions)",
    "TEST-001": "no automated tests for non-trivial code",
    "TEST-002": "tests are broken, skipped wholesale, or cannot pass as written",
    "TEST-003": "CI never executes the test suite",
    "CI-001": "incorrect or ineffective workflow triggers (branches/paths/events)",
    "CI-002": "CI gate is ineffective (continue-on-error, `|| true`, `if: false`, exit-zero)",
    "CI-003": "validation step is skipped, disabled or commented out",
    "CI-004": "third-party action pinned to a moving branch ref (@main/@master) or unversioned",
    "SEC-001": "hardcoded credential, token or private key",
    "SEC-002": "unsafe GitHub Actions token permissions (write-all, missing least-privilege)",
    "SEC-003": "workflow injection risk (untrusted input in run:, pull_request_target + PR checkout)",
    "SEC-004": "secret-bearing file committed (.env, key material, state files)",
    "SEC-005": "insecure practice (disabled TLS verification, curl|sh, secrets printed to logs)",
    "CFG-001": "machine-specific absolute path or hostname in committed config/code",
    "CFG-002": "local/IDE/personal settings committed",
    "CFG-003": "poor environment separation (debug/prod settings mixed, no env-based config)",
    "COLLAB-001": "unsafe pull-request automation (auto-merge/auto-approve without checks)",
    "COLLAB-002": "missing contribution/review guidance where the project clearly needs it",
    "BUILD-001": "no build validation (e.g. Dockerfile/package never built in CI)",
    "BUILD-002": "risky release or deployment automation (deploy on every push, no gates)",
    "BUILD-003": "container/build definition risk (latest tag, root user, secrets in image)",
}
_RULE_ID = re.compile(r"^[A-Z]{2,6}-\d{3}$")
_PREFIX_CATEGORY = {prefix: cat for cat, prefix in CATEGORY_PREFIX.items()}


@dataclass(frozen=True)
class Rejection:
    reason: str


def _clean_text(value: Any, limit: int) -> str:
    text = redact(re.sub(r"\s+", " ", str(value or "")).strip())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _normalise_category(raw: Any, rule_id: str | None) -> str | None:
    key = str(raw or "").strip().lower().replace(" ", "_").replace("-", "_")
    if key in CATEGORIES:
        return key
    if key in _CATEGORY_ALIASES:
        return _CATEGORY_ALIASES[key]
    if rule_id and rule_id.split("-")[0] in _PREFIX_CATEGORY:
        return _PREFIX_CATEGORY[rule_id.split("-")[0]]
    return None


def normalise_violation(raw: Any, tools: Any) -> tuple[dict[str, Any] | None, str | None]:
    """Validate one LLM-proposed finding against the real repository.

    Returns ``(violation, None)`` or ``(None, reason)``; the reason is fed back to the
    LLM so it can correct itself. ``tools`` is the :class:`tools.RepoTools` for the repo.
    """
    if not isinstance(raw, dict):
        return None, "finding must be a JSON object"
    rule_id = str(raw.get("rule_id") or "").strip().upper()
    category = _normalise_category(raw.get("category"), rule_id)
    if category is None:
        return None, f"category must be one of {', '.join(CATEGORIES)}"
    if not _RULE_ID.match(rule_id) or _PREFIX_CATEGORY.get(rule_id.split("-")[0]) != category:
        rule_id = f"{CATEGORY_PREFIX[category]}-099"
    severity = str(raw.get("severity") or "").strip().lower()
    if severity not in SEVERITIES:
        return None, f"severity must be one of {', '.join(SEVERITIES)}"

    description = _clean_text(raw.get("description"), 240)
    evidence = _clean_text(raw.get("evidence"), 500)
    recommendation = _clean_text(raw.get("recommendation"), 400)
    if not (description and evidence and recommendation):
        return None, "description, evidence and recommendation are all required"

    file_value = raw.get("file")
    file_path: str | None = None
    if file_value not in (None, "", "null", "None"):
        file_path = str(file_value).replace("\\", "/").strip()
        while file_path.startswith("./"):
            file_path = file_path[2:]
        if not tools.has_file(file_path):
            return None, (f"file {file_path!r} is not a tracked file in the repository; "
                          "use null for 'missing file' findings or fix the path")
        if file_path not in tools.inspected:
            return None, (f"you have not inspected {file_path!r} yet; read it (or search/analyze it) "
                          "before reporting a finding there")

    line: int | None = None
    line_value = raw.get("line")
    if file_path is not None and line_value not in (None, "", "null"):
        try:
            line = int(line_value)
        except (TypeError, ValueError):
            return None, "line must be an integer or null"
        total = tools.file_line_count(file_path)
        if line < 1 or (total is not None and line > total):
            return None, f"line {line} is outside {file_path} ({total} lines); fix it or use null"

    return {
        "rule_id": rule_id,
        "category": category,
        "severity": severity,
        "file": file_path,
        "line": line,
        "description": description,
        "evidence": evidence,
        "recommendation": recommendation,
    }, None


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2}


def _is_duplicate(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if a["file"] != b["file"]:
        return False
    if a["rule_id"] == b["rule_id"] and (a["line"] == b["line"] or a["line"] is None or b["line"] is None):
        return True
    if a["category"] != b["category"]:
        return False
    ta, tb = _tokens(a["description"]), _tokens(b["description"])
    if not ta or not tb:
        return False
    return len(ta & tb) / len(ta | tb) >= 0.7


def deduplicate(violations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop repeated findings, keeping the highest-severity copy; order by severity."""
    kept: list[dict[str, Any]] = []
    for violation in sorted(violations, key=lambda v: _SEVERITY_RANK[v["severity"]]):
        if not any(_is_duplicate(violation, existing) for existing in kept):
            kept.append(violation)
    return sorted(kept, key=lambda v: (_SEVERITY_RANK[v["severity"]], v["category"], v["file"] or "",
                                       v["line"] or 0))
