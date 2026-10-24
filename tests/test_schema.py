import pytest

from schema import deduplicate, normalise_violation
from tools import RepoTools


@pytest.fixture
def tools(sample_repo):
    t = RepoTools(sample_repo)
    t.dispatch("read_file", {"path": "config.py"})
    return t


def finding(**over):
    base = {
        "rule_id": "SEC-001", "category": "security", "severity": "high", "file": "config.py", "line": 1,
        "description": "Hardcoded API credential", "evidence": "Literal key assigned to API_KEY",
        "recommendation": "Use an environment variable",
    }
    return {**base, **over}


def test_valid_finding_has_exactly_the_contract_fields(tools):
    violation, error = normalise_violation(finding(), tools)
    assert error is None
    assert set(violation) == {"rule_id", "category", "severity", "file", "line",
                              "description", "evidence", "recommendation"}


def test_evidence_is_redacted(tools):
    violation, _ = normalise_violation(finding(evidence="key is ghp_" + "a" * 36), tools)
    assert "ghp_" not in violation["evidence"]


@pytest.mark.parametrize("override,fragment", [
    ({"severity": "urgent"}, "severity"),
    ({"category": "magic", "rule_id": "ZZZ-001"}, "category"),
    ({"file": "nope.py"}, "not a tracked file"),
    ({"file": "app.py"}, "not inspected"),
    ({"line": 99}, "outside"),
    ({"evidence": ""}, "required"),
])
def test_invalid_findings_are_rejected_with_reason(tools, override, fragment):
    violation, error = normalise_violation(finding(**override), tools)
    assert violation is None and fragment in error


def test_missing_file_findings_use_null_and_drop_line(tools):
    violation, error = normalise_violation(finding(file=None, line=5, rule_id="REPO-002",
                                                   category="repository_hygiene"), tools)
    assert error is None and violation["file"] is None and violation["line"] is None


def test_unknown_rule_id_falls_back_to_category_default(tools):
    violation, _ = normalise_violation(finding(rule_id="WHATEVER"), tools)
    assert violation["rule_id"] == "SEC-099"


def test_category_alias_and_case(tools):
    violation, _ = normalise_violation(finding(category="CI/CD", rule_id="CI-001", severity="HIGH"), tools)
    assert violation["category"] == "ci" and violation["severity"] == "high"


def test_deduplicate_keeps_highest_severity_and_orders():
    a = {"rule_id": "SEC-001", "category": "security", "severity": "low", "file": "x", "line": 1,
         "description": "Hardcoded API key found", "evidence": "e", "recommendation": "r"}
    b = {**a, "severity": "critical", "line": 1, "description": "Hardcoded credential"}
    c = {**a, "rule_id": "CI-002", "category": "ci", "severity": "medium", "file": "y", "description": "gate"}
    out = deduplicate([a, c, b])
    assert [v["severity"] for v in out] == ["critical", "medium"]


def test_different_lines_same_rule_are_not_duplicates():
    a = {"rule_id": "SEC-001", "category": "security", "severity": "high", "file": "x", "line": 1,
         "description": "key one", "evidence": "e", "recommendation": "r"}
    b = {**a, "line": 9, "description": "totally other secret in file"}
    assert len(deduplicate([a, b])) == 2
