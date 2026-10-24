"""End-to-end CLI behaviour with a scripted LLM (no network)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import agent
from llm import LLMError, resolve_config
from repo import RepoError, parse_github_url

REQUIRED = {"rule_id", "category", "severity", "file", "line", "description", "evidence", "recommendation"}
ROOT = Path(__file__).resolve().parent.parent


class FakeLLM:
    calls = 0

    def __init__(self, replies):
        self.replies = list(replies)

    def complete(self, messages, max_tokens=0):
        self.calls += 1
        return json.dumps(self.replies.pop(0) if self.replies else {"done": True})


def scripted(*replies):
    return lambda: FakeLLM(replies)


FINDING = {"rule_id": "SEC-004", "category": "security", "severity": "high", "file": ".env", "line": 1,
           "description": "Environment file with credentials is committed",
           "evidence": ".env is tracked and holds a SECRET_KEY value", "recommendation": "Remove and rotate"}


def test_writes_valid_report_and_trace(sample_repo, tmp_path):
    out = tmp_path / "report.json"
    code = agent.main(["--repo", str(sample_repo), "--output", str(out), "--no-verify", "--quiet"],
                      llm_factory=scripted(
                          {"tool_calls": [{"tool": "read_file", "args": {"path": ".env"}}]},
                          {"record_findings": [FINDING], "done": True}))
    assert code == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert set(report) == {"repository", "violations"}
    assert report["repository"] == str(sample_repo)
    assert len(report["violations"]) == 1 and set(report["violations"][0]) == REQUIRED
    assert "abcdef123456789" not in out.read_text(encoding="utf-8")
    trace = [json.loads(line) for line in (tmp_path / "report.json.trace.jsonl").read_text().splitlines()]
    assert any(e.get("tool") == "read_file" for e in trace)


def test_empty_violations_array_when_nothing_found(sample_repo, tmp_path):
    out = tmp_path / "r.json"
    assert agent.main(["--repo", str(sample_repo), "--output", str(out), "--quiet"],
                      llm_factory=scripted({"done": True})) == 0
    assert json.loads(out.read_text(encoding="utf-8")) == {"repository": str(sample_repo), "violations": []}


def test_target_repository_is_untouched(sample_repo, tmp_path):
    before = subprocess.run(["git", "-C", str(sample_repo), "status", "--porcelain"], capture_output=True,
                            text=True).stdout
    agent.main(["--repo", str(sample_repo), "--output", str(tmp_path / "r.json"), "--quiet"],
               llm_factory=scripted({"tool_calls": [{"tool": "run_ruff", "args": {}}]}, {"done": True}))
    after = subprocess.run(["git", "-C", str(sample_repo), "status", "--porcelain"], capture_output=True,
                           text=True).stdout
    assert before == after == ""


def test_invalid_url_exits_nonzero_with_message(tmp_path, capsys):
    code = agent.main(["--repo", "https://example.com/x/y", "--output", str(tmp_path / "r.json"), "--quiet"],
                      llm_factory=scripted({"done": True}))
    assert code == 2
    assert "not a public GitHub repository URL" in capsys.readouterr().err
    assert not (tmp_path / "r.json").exists()


def test_missing_credentials_exit_code(tmp_path, monkeypatch, capsys):
    for var in ("LLM_PROVIDER", "LLM_API_KEY", "LLM_BASE_URL", "GROQ_API_KEY", "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(agent, "_load_dotenv", lambda: None)
    code = agent.main(["--repo", "https://github.com/a/b", "--output", str(tmp_path / "r.json")])
    assert code == 3 and "No LLM credentials" in capsys.readouterr().err


def test_llm_outage_on_first_call_is_nonzero(sample_repo, tmp_path, capsys):
    class Dead:
        calls = 0

        def complete(self, *_a, **_k):
            raise LLMError("boom")

    code = agent.main(["--repo", str(sample_repo), "--output", str(tmp_path / "r.json"), "--quiet"],
                      llm_factory=lambda: Dead())
    assert code == 3 and "LLM unavailable" in capsys.readouterr().err


def test_cli_requires_arguments():
    proc = subprocess.run([sys.executable, str(ROOT / "agent.py")], capture_output=True, text=True)
    assert proc.returncode == 2 and "--repo" in proc.stderr


@pytest.mark.parametrize("url,expected", [
    ("https://github.com/owner/repo", ("owner", "repo")),
    ("https://github.com/owner/repo.git", ("owner", "repo")),
    ("https://github.com/owner/repo/", ("owner", "repo")),
    ("https://github.com/owner/my.repo-x/tree/main/src", ("owner", "my.repo-x")),
])
def test_parse_github_url(url, expected):
    assert parse_github_url(url) == expected


@pytest.mark.parametrize("url", ["http://github.com/a/b", "https://gitlab.com/a/b", "git@github.com:a/b.git",
                                 "https://github.com/a", "https://github.com/a/b;rm -rf", "https://github.com/../b"])
def test_parse_github_url_rejects(url):
    with pytest.raises(RepoError):
        parse_github_url(url)


def test_resolve_config_autodetect_and_overrides():
    cfg = resolve_config({"GROQ_API_KEY": "k"})
    assert cfg["provider"] == "groq" and "groq.com" in cfg["base_url"]
    cfg = resolve_config({"LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "k", "LLM_MODEL": "m"})
    assert (cfg["provider"], cfg["model"]) == ("anthropic", "m")
    cfg = resolve_config({"LLM_API_KEY": "k", "LLM_BASE_URL": "http://localhost:11434/v1", "LLM_MODEL": "llama"})
    assert cfg["base_url"].startswith("http://localhost")
    with pytest.raises(LLMError):
        resolve_config({})
    with pytest.raises(LLMError):
        resolve_config({"LLM_PROVIDER": "openai"})


def test_llm_client_classifies_provider_errors():
    from llm import ContextTooLarge, LLMClient

    class RateLimit(Exception):
        status_code = 429

    client = LLMClient("groq", "k", "m", "https://example.invalid/v1")
    client._retryable = (RateLimit,)

    def boom(message):
        def _once(*_a, **_k):
            raise RateLimit(message)
        return _once

    client._once = boom("Request too large for model x on tokens per minute (TPM): Limit 8000")
    with pytest.raises(ContextTooLarge):
        client.complete([{"role": "system", "content": "s"}])
    client._once = boom("Rate limit reached ... on tokens per day (TPD): Limit 200000")
    with pytest.raises(LLMError, match="quota exhausted"):
        client.complete([{"role": "system", "content": "s"}])
