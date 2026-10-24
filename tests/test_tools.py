import os

import pytest

from tools import RepoTools


@pytest.fixture
def tools(sample_repo):
    return RepoTools(sample_repo)


def test_list_files_shows_tracked_files(tools):
    out = tools.dispatch("list_files", {})
    assert "README.md" in out and "requirements.txt" in out
    assert ".git/" not in out


def test_list_files_subdirectory_and_missing(tools):
    assert "ci.yml" in tools.dispatch("list_files", {"path": ".github/workflows"})
    assert "ERROR" in tools.dispatch("list_files", {"path": "nope"})


def test_read_file_numbers_lines_and_redacts(tools):
    out = tools.dispatch("read_file", {"path": "config.py"})
    assert "1: API_KEY" in out
    assert "aB3dE5gH7jK9" not in out
    assert "config.py" in tools.inspected


def test_read_file_range_and_truncation_hint(make_repo):
    root = make_repo({"big.txt": "\n".join(f"line{i}" for i in range(1000))})
    out = RepoTools(root).dispatch("read_file", {"path": "big.txt", "start_line": 10, "end_line": 12})
    assert "10: line9" in out and "12: line11" in out and "13:" not in out
    capped = RepoTools(root).dispatch("read_file", {"path": "big.txt"})
    assert "call again with start_line=401" in capped


@pytest.mark.parametrize("bad", ["../etc/passwd", "/etc/passwd", "C:/Windows/win.ini", "a/../../x"])
def test_read_file_rejects_path_escape(tools, bad):
    assert tools.dispatch("read_file", {"path": bad}).startswith("ERROR")


def test_read_file_missing_suggests_location(tools):
    out = tools.dispatch("read_file", {"path": "ci.yml"})
    assert "ERROR" in out and ".github/workflows/ci.yml" in out


@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
def test_symlinks_are_not_followed(make_repo, tmp_path):
    root = make_repo({"a.txt": "x"})
    outside = tmp_path / "secret.txt"
    outside.write_text("top secret")
    (root / "link.txt").symlink_to(outside)
    assert RepoTools(root).dispatch("read_file", {"path": "link.txt"}).startswith("ERROR")


def test_search_code_finds_and_reports_location(tools):
    out = tools.dispatch("search_code", {"pattern": r"/home/\w+"})
    assert "app.py:4" in out
    assert "no matches" in tools.dispatch("search_code", {"pattern": "zzzzqqq"})
    assert "invalid regular expression" in tools.dispatch("search_code", {"pattern": "("})


def test_git_log_and_status(tools):
    assert "initial commit" in tools.dispatch("git_log", {})
    assert "working tree clean" in tools.dispatch("git_status", {})


def test_scan_secrets_hides_values(tools):
    out = tools.dispatch("scan_secrets", {})
    assert "config.py:1" in out
    assert "sk-live" not in out and "value hidden" in out
    assert ".env.example" not in out


def test_find_tracked_artifacts(tools):
    out = tools.dispatch("find_tracked_artifacts", {})
    assert "__pycache__" in out and ".env  <-" in out
    assert ".env.example" not in out


def test_analyze_workflow_all_and_single(tools):
    both = tools.dispatch("analyze_workflow", {})
    assert "ci.yml" in both and "good.yml" in both
    assert "pull_request_target" in tools.dispatch("analyze_workflow", {"path": ".github/workflows/ci.yml"})
    assert "ERROR" in tools.dispatch("analyze_workflow", {"path": "x.yml"})


def test_run_ruff_reports_or_declares_unavailable(make_repo):
    root = make_repo({"m.py": "import os\n"})
    out = RepoTools(root).dispatch("run_ruff", {})
    assert "F401" in out or out.startswith("UNAVAILABLE")


def test_run_ruff_without_python_files(make_repo):
    assert "no Python files" in RepoTools(make_repo({"a.txt": "x"})).dispatch("run_ruff", {})


def test_unknown_tool_and_bad_args_never_raise(tools):
    assert tools.dispatch("rm_rf", {}).startswith("ERROR: unknown tool")
    assert "unexpected argument" in tools.dispatch("read_file", {"path": "README.md", "bogus": 1})
    assert tools.dispatch("read_file", {"path": "README.md", "start_line": "abc"}).startswith("ERROR")


def test_tools_leave_repository_unchanged(tools):
    before = tools.git_status()
    for name in tools.registry:
        tools.dispatch(name, {})
    assert tools.git_status() == before
