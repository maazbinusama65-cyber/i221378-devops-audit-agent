from conftest import BAD_WORKFLOW, HEALTHY_WORKFLOW

from workflows import parse_workflow, summarize_workflow


def test_bad_workflow_facts_are_surfaced_with_lines():
    out = summarize_workflow("ci.yml", BAD_WORKFLOW)
    assert "pull_request_target" in out
    assert "write-all" in out
    assert "script injection" in out
    assert "swallows failure" in out
    assert "continue-on-error: true" in out
    assert "if: false" in out
    assert "moving ref @main" in out
    assert "untrusted PR ref" in out
    assert "L16: step is disabled" in out  # line-aware


def test_healthy_workflow_has_no_observations():
    out = summarize_workflow("ci.yml", HEALTHY_WORKFLOW)
    assert "tests=True lint=True" in out
    assert out.rstrip().endswith("- none")


def test_missing_permissions_is_reported():
    out = summarize_workflow("w.yml", "on: push\njobs:\n  a:\n    runs-on: x\n    steps:\n      - run: pytest\n")
    assert "no `permissions:` block" in out


def test_invalid_yaml_does_not_crash():
    assert "could not be parsed" in summarize_workflow("w.yml", "a: [unclosed")


def test_on_key_parsed_as_boolean_is_handled():
    assert parse_workflow("on: push\njobs: {}\n")[True] == "push"
    assert "triggers (line 1): push" in summarize_workflow("w.yml", "on: push\njobs: {}\n")
