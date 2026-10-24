# DevOps Repository Auditing Agent

Assignment 2 · MLOps & Production AI Engineering · Roll no. **i221378**

An autonomous agent that inspects a **public GitHub repository**, decides *for itself* which
inspection tools to run next, and writes an evidence-based JSON report of DevOps best-practice
violations. The agent loop, tool dispatch and control flow are plain Python: **no LangChain,
LangGraph, n8n or any other agent framework**. Only the basic provider SDKs (`openai`,
`anthropic`) are used to talk to the model.

```
python agent.py --repo https://github.com/<owner>/<repo> --output report.json
```

---

## 1. Installation

Requirements: Python 3.10+, `git` on `PATH`, outbound HTTPS access to GitHub and the LLM provider.

```bash
git clone https://github.com/maazbinusama65-cyber/i221378-devops-audit-agent.git
cd i221378-devops-audit-agent

python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Configuration

Copy `.env.example` to `.env` and set **one** provider key (the file is git-ignored). The agent is
provider-agnostic; the model choice does not affect the design.

```bash
cp .env.example .env
```

| Variable | Meaning |
|---|---|
| `LLM_PROVIDER` | `groq`, `openai`, `anthropic`, `gemini`, `openrouter` (autodetected from whichever key is set if omitted) |
| `GROQ_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` / `OPENROUTER_API_KEY` | key for the chosen provider |
| `LLM_API_KEY` + `LLM_BASE_URL` | any other OpenAI-compatible endpoint (Ollama, vLLM, Azure, ...) |
| `LLM_MODEL` | optional model override (defaults: Groq `llama-3.3-70b-versatile`, OpenAI `gpt-4o-mini`, Anthropic `claude-sonnet-4-5`, Gemini `gemini-2.0-flash`) |

Real environment variables take precedence over `.env`, so the evaluator can simply export them.

## 3. Running

```bash
python agent.py --repo https://github.com/example/demo-project --output report.json
```

| Option | Default | Purpose |
|---|---|---|
| `--repo` | required | public GitHub URL (`https://github.com/<owner>/<repo>`) |
| `--output` | required | where the JSON report is written |
| `--trace` | `<output>.trace.jsonl` | tool-call trace (one JSON object per step) |
| `--max-steps` | 24 | LLM reasoning-step budget |
| `--max-tool-calls` | 70 | tool-execution budget |
| `--time-budget` | 900 | wall-clock budget in seconds |
| `--no-verify` | off | skip the second-opinion review pass |
| `--quiet` | off | no progress output on stderr |

The run is unattended (no prompts). Exit codes: `0` report written · `2` bad input or repository
unreachable · `3` LLM misconfigured/unavailable · `1` internal error. On a non-zero exit **no report is
written** and a one-line reason is printed to stderr. Budget exhaustion or a mid-run LLM outage is
*not* fatal: the agent reports what it has already verified and prints a warning.

### Output contract

```json
{
  "repository": "https://github.com/example/demo-project",
  "violations": [
    {
      "rule_id": "SEC-001",
      "category": "security",
      "severity": "critical",
      "file": "config.py",
      "line": 17,
      "description": "Hardcoded API credential",
      "evidence": "Credential assigned directly in source code",
      "recommendation": "Use a secrets manager or environment variable"
    }
  ]
}
```

`severity` is one of `critical|high|medium|low`; `file`/`line` are JSON `null` for findings about
something that is absent (e.g. no tests). `violations` is `[]` when nothing defensible is found.
Secret values are never written: everything is passed through a redactor first.
See [`sample_report.json`](sample_report.json) for real output.

---

## 4. Architecture

```
                    ┌──────────────────────── agent.py (CLI) ────────────────────────┐
                    │ parse args · load .env · clone · run agent · write JSON/trace   │
                    └───────┬───────────────────────────────────────────┬────────────┘
                            │                                           │
                     repo.py (safe clone)                        llm.py (LLM client)
                            │                                           │
                            ▼                                           ▼
        ┌──────────────────────────────────────  auditor.py  ──────────────────────────────────┐
        │  OBSERVE      initial overview (file tree summary, git metadata)                     │
        │  REASON       LLM returns JSON: thought + tool_calls + record_findings + done       │
        │  SELECT TOOL  the LLM names the next tool(s) and arguments                           │
        │  EXECUTE      tools.py runs them read-only; output redacted and size-capped          │
        │  INTERPRET    results go back to the LLM, which records / refines findings           │
        │  REPORT       validate → de-duplicate → second-opinion review → JSON                  │
        └──────────────────────────────────────────────────────────────────────────────────────┘
```

| Module | Responsibility |
|---|---|
| `agent.py` | CLI entry point, exit codes, atomic JSON/trace writing |
| `auditor.py` | The agent loop: transcript, tool dispatch, ledger, budgets, context compression, review pass |
| `tools.py` | The inspection tools (below) bound to one checkout |
| `workflows.py` | Line-aware GitHub Actions reader that emits *neutral facts* for the LLM to judge |
| `llm.py` | OpenAI-compatible and Anthropic backends, retry/back-off, provider presets |
| `prompts.py` | System prompt: protocol, inspection areas, evidence rules, rule catalog |
| `schema.py` | Rule catalog, validation of findings against the real repo, de-duplication |
| `redact.py` | Credential detection and masking used on every output path |
| `repo.py` | Hardened shallow clone and the initial observation |

### The LLM drives the investigation

There is **no fixed checklist and no static scanner followed by a summary**. Each turn the model
receives the results of its previous tool calls and replies with a JSON object:

```json
{"thought": "...", "tool_calls": [{"tool": "read_file", "args": {"path": ".github/workflows/ci.yml"}}],
 "record_findings": [ ... ], "done": false}
```

It chooses what to look at next from what it has just seen: a flagged workflow leads to reading the
files it references, a dependency manifest leads to comparing it against the lockfile, a secret
scan hit leads to reading the surrounding lines to tell a real key from a fixture. The harness only
executes the requested calls, enforces budgets (steps, tool calls, wall-clock, context size,
identical-call limit) and feeds back validation errors. `tests/test_auditor.py` asserts this
behaviour with a scripted LLM whose next action depends on the previous tool output.

### Tools

| Tool | Purpose |
|---|---|
| `list_files(path, depth)` | Discover tracked files and structure (large trees are collapsed) |
| `read_file(path, start_line, end_line)` | Numbered, range-limited, redacted file contents |
| `search_code(pattern, path, max_results)` | Regex search with `file:line` hits |
| `git_log(max_count, path, show_files)` | Commit history (shallow clone, depth 100) |
| `git_status()` | Working-tree state (used to prove the run changed nothing) |
| `analyze_workflow(path)` | Parsed triggers, permissions, jobs, steps plus neutral observations (`pull_request_target`, `\|\| true`, `continue-on-error`, `if: false`, unpinned actions, script injection patterns, ...) |
| `scan_secrets(path)` | Locations of credential-like values; **values are never shown** |
| `find_tracked_artifacts()` | Tracked files matching temp/generated/local-only patterns |
| `run_ruff(path)` | Static lint via ruff (honours the repo's ruff config, else default rules) |

Every tool result is interpreted by the agent (it is the only consumer), and a tool that cannot run
returns `UNAVAILABLE: ...` or `ERROR: ...` rather than a made-up result.

`run_pytest` is intentionally **not** provided: running a stranger's test suite means executing
untrusted code, which the assignment forbids without isolation. The agent instead reasons about
whether CI runs tests and whether the gate can actually fail.

### Precision: how false positives are limited

1. **Grounding at record time.** A finding must name a tracked file that the agent has actually
   inspected, and the line must exist; otherwise it is rejected and the reason is sent back to the
   model. Missing-file findings use `file: null`.
2. **Evidence rules in the prompt.** Equivalent solutions count (pyproject vs requirements, ruff in
   pre-commit/tox/CI, Poetry/uv lockfiles, libraries without lockfiles), example/template/placeholder
   values are acceptable, branch-protection settings are never inferred, unavailable checks are never
   invented, and a healthy repository may legitimately yield `[]`.
3. **De-duplication.** One root cause, one finding (same rule/file/line or near-identical text).
4. **Second-opinion review.** Before output, a separate LLM call sees each claim next to the real
   source excerpt and drops findings the excerpt contradicts (`--no-verify` disables it).

### Safety

* The target is cloned with hooks disabled, symlinks materialised as plain files, LFS smudge off, no
  credentials, shallow depth, a size limit and a timeout, into a private temp dir that is deleted
  afterwards. Nothing is ever pushed or written back; `git status` is compared before/after the run.
* **No repository code is executed.** All tools only read text; `ruff` is a static analyser.
* `read_file` rejects absolute paths, `..`, and symlinks; files over 1 MB or binary are refused.
* Secrets: `redact.py` masks known token formats (AWS, GitHub, Slack, Google, Stripe, OpenAI-style,
  Groq, JWTs, URL passwords, private keys) and `name = "value"` credential assignments in **tool
  output, prompts, traces, logs and the report**. The runtime LLM API key is scrubbed as well.
* Errors (missing files, invalid regex, LLM rate limits/timeouts, unparsable YAML, invalid JSON from
  the model) are handled and surfaced to the model or as warnings, never as crashes.
* Nothing is hard-coded about particular repositories; the prompt teaches the method and generic
  rules only.

---

## 5. Tests

```bash
pip install -r requirements-dev.txt
pytest                    # ~80 tests, no network and no API key needed
pytest --cov=. --cov-report=term-missing
```

* `test_tools.py` - every tool, path-escape and symlink rejection, redaction, read-only guarantee
* `test_workflows.py` - GitHub Actions analysis on bad and healthy workflows
* `test_schema.py` - finding validation, evidence grounding, de-duplication
* `test_auditor.py` - agent behaviour with a scripted LLM: observation-dependent tool choice, budgets,
  retries on invalid JSON, mid-run LLM failure, review pass, secret non-leakage
* `test_cli.py` - JSON contract, exit codes, URL validation, untouched target, provider config

CI (`.github/workflows/ci.yml`) runs ruff and the test suite on every push and pull request.

## 6. Limitations

* Only public GitHub URLs are fetched; GitHub-side settings (branch protection, secrets, required
  reviews) are invisible from files and are deliberately never reported.
* Findings quality depends on the model; weaker models may investigate less deeply. Raise
  `--max-steps` for large repositories.
* Free-tier provider rate limits can slow a run; the client honours `Retry-After` and backs off.
