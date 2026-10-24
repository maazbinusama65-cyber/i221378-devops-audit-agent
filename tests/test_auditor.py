"""Agent-loop behaviour, driven by a scripted fake LLM that reacts to tool output."""

from __future__ import annotations

import json

import pytest

from auditor import AuditAgent, AuditConfig, extract_json
from llm import LLMError
from repo import build_overview
from tools import RepoTools


class ScriptedLLM:
    """Replies via a policy function of (turn index, last user message)."""

    def __init__(self, policy):
        self.policy = policy
        self.calls = 0
        self.transcripts: list[list[dict]] = []

    def complete(self, messages, max_tokens=0):
        self.calls += 1
        self.transcripts.append([dict(m) for m in messages])
        reply = self.policy(self.calls, messages[-1]["content"], messages)
        return reply if isinstance(reply, str) else json.dumps(reply)


def run_agent(repo, policy, **cfg):
    tools = RepoTools(repo)
    llm = ScriptedLLM(policy)
    agent = AuditAgent(llm, tools, build_overview(tools, "local"), AuditConfig(verify=False, **cfg),
                       log=lambda _m: None)
    return agent, llm, agent.run()


def adaptive_policy(turn, last, _messages):
    """Next tool depends on what the previous observation contained."""
    if turn == 1:
        return {"thought": "orient", "tool_calls": [{"tool": "list_files", "args": {}}]}
    if turn == 2:
        assert ".github/workflows/ci.yml" in last  # it really saw the tool output
        return {"thought": "workflow present, analyse it",
                "tool_calls": [{"tool": "analyze_workflow", "args": {"path": ".github/workflows/ci.yml"}}]}
    if turn == 3:
        assert "pull_request_target" in last
        return {"thought": "flagged, record it", "record_findings": [{
            "rule_id": "SEC-003", "category": "security", "severity": "critical",
            "file": ".github/workflows/ci.yml", "line": 3,
            "description": "pull_request_target checks out untrusted PR code",
            "evidence": "trigger pull_request_target plus checkout of PR head sha",
            "recommendation": "Use pull_request or avoid checking out PR code"}],
            "tool_calls": [{"tool": "scan_secrets", "args": {}}]}
    return {"thought": "done", "done": True}


def test_loop_selects_tools_from_observations(sample_repo):
    agent, llm, result = run_agent(sample_repo, adaptive_policy)
    tools_used = [e["tool"] for e in result.trace if e["event"] == "tool_call"]
    assert tools_used == ["list_files", "analyze_workflow", "scan_secrets"]
    assert [v["rule_id"] for v in result.violations] == ["SEC-003"]
    assert result.steps == 5 and result.tool_calls == 3  # step 4 "done" was challenged
    assert all(m["role"] in {"system", "user", "assistant"} for m in llm.transcripts[-1])


def test_secret_values_never_reach_the_llm(sample_repo):
    def policy(turn, last, messages):
        if turn == 1:
            return {"tool_calls": [{"tool": "read_file", "args": {"path": "config.py"}},
                                   {"tool": "search_code", "args": {"pattern": "API_KEY"}}]}
        return {"done": True}

    _, llm, _ = run_agent(sample_repo, policy)
    everything = json.dumps(llm.transcripts)
    assert "aB3dE5gH7jK9" not in everything


def test_rejected_findings_are_fed_back_and_not_reported(sample_repo):
    def policy(turn, last, _m):
        if turn == 1:
            return {"record_findings": [{
                "rule_id": "SEC-001", "category": "security", "severity": "high", "file": "ghost.py",
                "line": 1, "description": "d", "evidence": "e", "recommendation": "r"}]}
        if turn == 2:
            assert "REJECTED" in last and "ghost.py" in last
        return {"done": True}

    _, _, result = run_agent(sample_repo, policy)
    assert result.violations == []


def test_invalid_json_is_retried_then_aborts(sample_repo):
    calls = []

    def policy(turn, last, _m):
        calls.append(last[:40])
        return "I will now inspect the repo"

    _, _, result = run_agent(sample_repo, policy, max_steps=10)
    assert result.violations == [] and any("invalid JSON" in w for w in result.warnings)


def test_json_in_code_fence_is_accepted(sample_repo):
    def policy(turn, last, _m):
        return "```json\n" + json.dumps({"done": True}) + "\n```"

    _, _, result = run_agent(sample_repo, policy)
    assert result.steps == 3  # fenced JSON parsed; early done challenged twice, then accepted


def test_budget_exhaustion_triggers_final_report_call(sample_repo):
    def policy(turn, last, _m):
        if "budget is exhausted" in last:
            return {"record_findings": [{
                "rule_id": "CI-002", "category": "ci", "severity": "high", "file": None, "line": None,
                "description": "late finding", "evidence": "seen earlier", "recommendation": "fix"}],
                "done": True}
        return {"tool_calls": [{"tool": "list_files", "args": {"path": f"x{turn}"}}]}

    _, _, result = run_agent(sample_repo, policy, max_steps=3)
    assert [v["description"] for v in result.violations] == ["late finding"]


def test_repeated_identical_calls_are_blocked(sample_repo):
    def policy(turn, last, _m):
        if turn <= 4:
            return {"tool_calls": [{"tool": "git_log", "args": {}}]}
        return {"done": True}

    _, _, result = run_agent(sample_repo, policy)
    assert result.tool_calls == 3  # the 4th identical call is blocked


def test_tool_call_cap_per_turn(sample_repo):
    def policy(turn, last, _m):
        if turn == 1:
            return {"tool_calls": [{"tool": "list_files", "args": {"path": str(i)}} for i in range(9)]}
        return {"done": True}

    _, _, result = run_agent(sample_repo, policy, max_calls_per_turn=4)
    assert result.tool_calls == 4


def test_time_budget(sample_repo):
    ticks = iter(range(0, 10_000, 500))
    tools = RepoTools(sample_repo)
    llm = ScriptedLLM(lambda *_: {"tool_calls": [{"tool": "git_log", "args": {"max_count": next(ticks)}}]})
    agent = AuditAgent(llm, tools, "x", AuditConfig(verify=False, time_budget_s=1000, max_steps=50),
                       log=lambda _m: None, clock=lambda: next(ticks))
    result = agent.run()
    assert any("time budget" in w for w in result.warnings)


def test_llm_failure_on_first_call_is_fatal(sample_repo):
    class Dead:
        calls = 0

        def complete(self, *_a, **_k):
            raise LLMError("no network")

    tools = RepoTools(sample_repo)
    with pytest.raises(LLMError):
        AuditAgent(Dead(), tools, "x", AuditConfig(verify=False), log=lambda _m: None).run()


def test_llm_failure_mid_run_keeps_verified_findings(sample_repo):
    state = {"n": 0}

    class Flaky:
        calls = 0

        def complete(self, *_a, **_k):
            state["n"] += 1
            if state["n"] == 1:
                return json.dumps({"tool_calls": [{"tool": "read_file", "args": {"path": "config.py"}}],
                                   "record_findings": []})
            if state["n"] == 2:
                return json.dumps({"record_findings": [{
                    "rule_id": "SEC-001", "category": "security", "severity": "high", "file": "config.py",
                    "line": 1, "description": "Hardcoded key", "evidence": "literal", "recommendation": "env"}],
                    "tool_calls": [{"tool": "git_log", "args": {}}]})
            raise LLMError("rate limited")

    tools = RepoTools(sample_repo)
    result = AuditAgent(Flaky(), tools, "x", AuditConfig(verify=True), log=lambda _m: None).run()
    assert len(result.violations) == 1 and any("LLM failed mid-run" in w for w in result.warnings)


def test_review_pass_drops_contradicted_findings(sample_repo):
    def policy(turn, last, messages):
        if messages[0]["content"].startswith("You are the second-opinion"):
            ids = [f["id"] for f in json.loads(last)["findings"]]
            return {"verdicts": [{"id": i, "keep": False, "reason": "placeholder"} for i in ids]}
        if turn == 1:
            return {"tool_calls": [{"tool": "read_file", "args": {"path": "config.py"}}],
                    "record_findings": []}
        if turn == 2:
            return {"record_findings": [{
                "rule_id": "SEC-001", "category": "security", "severity": "high", "file": "config.py",
                "line": 1, "description": "Hardcoded key", "evidence": "literal", "recommendation": "env"}],
                "done": True}
        return {"done": True}

    tools = RepoTools(sample_repo)
    llm = ScriptedLLM(policy)
    result = AuditAgent(llm, tools, "x", AuditConfig(verify=True), log=lambda _m: None).run()
    assert result.violations == []
    assert any(e["event"] == "finding_dropped_by_review" for e in result.trace)


def test_context_compression_keeps_recent_messages(sample_repo):
    def policy(turn, last, _m):
        if turn < 6:
            return {"tool_calls": [{"tool": "read_file", "args": {"path": "app.py", "start_line": turn}}]}
        return {"done": True}

    tools = RepoTools(sample_repo)
    llm = ScriptedLLM(policy)
    agent = AuditAgent(llm, tools, "x" * 100, AuditConfig(verify=False, context_chars=3000), log=lambda _m: None)
    big = "TOOL RESULTS:\n" + "z" * 5000 + "\n\nFINDINGS LEDGER:\n(empty)"
    agent.messages += [{"role": "assistant", "content": "{}"}, {"role": "user", "content": big}]
    agent._compress_context()
    assert "trimmed" in agent.messages[3]["content"] or len(agent.messages[3]["content"]) < 6000


@pytest.mark.parametrize("text,expected", [
    ('{"a": 1}', {"a": 1}),
    ('noise {"a": {"b": "}"}} trailing', {"a": {"b": "}"}}),
    ("```json\n{\"a\": 2}\n```", {"a": 2}),
    ("no json here", None),
    ("[1, 2]", None),
])
def test_extract_json(text, expected):
    assert extract_json(text) == expected


def test_context_too_large_shrinks_history_and_retries(sample_repo):
    from llm import ContextTooLarge

    class Tight:
        calls = 0

        def __init__(self):
            self.sizes = []

        def complete(self, messages, max_tokens=0):
            self.calls += 1
            size = sum(len(m["content"]) for m in messages)
            self.sizes.append(size)
            if size > 12_000:
                raise ContextTooLarge("413")
            return json.dumps({"done": True})

    tools = RepoTools(sample_repo)
    llm = Tight()
    agent = AuditAgent(llm, tools, "x", AuditConfig(verify=False), log=lambda _m: None)
    agent.messages.append({"role": "assistant", "content": "{}"})
    agent.messages.append({"role": "user", "content": "TOOL RESULTS:\n" + "z" * 15_000 + "\n\nFINDINGS LEDGER:\n-"})
    result = agent.run()
    assert llm.sizes[-1] <= 12_000
    assert any("shrinking history" in w for w in result.warnings)


def test_notes_are_kept_as_memory_and_echoed_back(sample_repo):
    seen = {}

    def policy(turn, last, _m):
        if turn == 1:
            return {"notes": "python app; CI has pull_request_target", "tool_calls": [
                {"tool": "read_file", "args": {"path": "README.md"}}]}
        seen["last"] = last
        return {"done": True}

    run_agent(sample_repo, policy)
    assert "YOUR NOTES" in seen["last"] and "pull_request_target" in seen["last"]
    assert "FILES ALREADY READ OR SEARCHED: README.md" in seen["last"]


def test_premature_done_is_challenged_then_model_can_insist(sample_repo):
    messages_seen = []

    def policy(turn, last, _m):
        messages_seen.append(last)
        return {"thought": "nothing applies", "done": True}

    _, _, result = run_agent(sample_repo, policy)
    assert result.steps == 3
    assert "no tool has yet examined" in messages_seen[1] and "scan_secrets" not in messages_seen[1]
    assert "credentials and secrets" in messages_seen[1]
    assert sum(e["event"] == "done_challenged" for e in result.trace) == 2


def test_done_accepted_without_challenge_after_coverage(sample_repo):
    def policy(turn, last, _m):
        if turn == 1:
            return {"tool_calls": [{"tool": t, "args": {}} for t in
                                   ("find_tracked_artifacts", "scan_secrets", "analyze_workflow", "git_log")]}
        return {"done": True}

    _, _, result = run_agent(sample_repo, policy)
    assert result.steps == 2


def test_stalled_loop_of_blocked_repeats_stops_early(sample_repo):
    def policy(turn, last, _m):
        return {"tool_calls": [{"tool": "git_log", "args": {}}]}

    _, _, result = run_agent(sample_repo, policy, max_steps=24)
    assert result.steps < 10 and any("stalled" in w for w in result.warnings)
    assert result.tool_calls == 3
