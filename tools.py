"""Inspection tools the audit agent can call.

All tools are *read-only* and operate on a local clone. Nothing from the
inspected repository is ever executed: ``run_ruff`` is a static analyser, and
the workflow/secret tools only parse text. Every output is redacted and
size-capped before it is shown to the LLM.
"""

from __future__ import annotations

import fnmatch
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from redact import find_secrets, redact
from workflows import summarize_workflow

MAX_READ_BYTES = 1_000_000
MAX_READ_LINES = 400
MAX_OUTPUT_CHARS = 9_000
GIT_TIMEOUT = 30
RUFF_TIMEOUT = 60

SKIP_CONTENT_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".zip", ".gz", ".tar", ".whl", ".jar", ".class",
    ".pyc", ".so", ".dll", ".exe", ".woff", ".woff2", ".ttf", ".eot", ".mp4", ".mp3", ".bin", ".pkl",
    ".parquet", ".onnx", ".pt", ".h5", ".sqlite", ".db", ".lock",
}

# Patterns that usually indicate a tracked temporary / generated / local-only file.
ARTIFACT_PATTERNS: list[tuple[str, str]] = [
    ("**/__pycache__/**", "Python bytecode cache"),
    ("*.pyc", "Python bytecode"),
    ("*.pyo", "Python bytecode"),
    ("**/node_modules/**", "installed JS dependencies"),
    ("**/.venv/**", "virtual environment"),
    ("**/venv/**", "virtual environment"),
    ("**/.idea/**", "JetBrains IDE settings"),
    ("**/.vscode/**", "VS Code settings"),
    ("**/.DS_Store", "macOS metadata"),
    ("**/Thumbs.db", "Windows metadata"),
    ("*.log", "log file"),
    ("*.tmp", "temporary file"),
    ("*.swp", "editor swap file"),
    ("*~", "editor backup file"),
    ("*.bak", "backup file"),
    ("*.orig", "merge leftover"),
    ("**/dist/**", "build output"),
    ("**/build/**", "build output"),
    ("**/*.egg-info/**", "packaging metadata"),
    ("**/.pytest_cache/**", "pytest cache"),
    ("**/.mypy_cache/**", "mypy cache"),
    ("**/.ruff_cache/**", "ruff cache"),
    ("**/.coverage", "coverage data"),
    ("**/coverage/**", "coverage output"),
    ("**/htmlcov/**", "coverage report"),
    (".env", "local environment file (may hold secrets)"),
    (".env.*", "local environment file (check if it is only an example/template)"),
    ("**/*.sqlite3", "local database"),
    ("**/*.db", "local database"),
    ("**/*.pem", "key/certificate material"),
    ("**/*.key", "key material"),
    ("**/id_rsa*", "SSH key material"),
    ("**/terraform.tfstate*", "Terraform state"),
    ("**/.terraform/**", "Terraform cache"),
    ("**/*.tfvars", "Terraform variables (may hold secrets)"),
]
TEMPLATE_ENV_NAMES = {".env.example", ".env.sample", ".env.template", ".env.dist", ".env.defaults"}


class ToolError(Exception):
    """A tool was called incorrectly or cannot run; message is shown to the LLM."""


def _cap(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [output truncated at {limit} chars; narrow the request]"


def _run_git(root: Path, *args: str, timeout: int = GIT_TIMEOUT) -> str:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_PAGER": "cat"}
    try:
        proc = subprocess.run(
            ["git", "-c", "core.hooksPath=", "-c", "core.fsmonitor=false", "-C", str(root), *args],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout, env=env, check=False,
        )
    except FileNotFoundError as exc:
        raise ToolError("git executable not found") from exc
    except subprocess.TimeoutExpired as exc:
        raise ToolError(f"git {args[0]} timed out after {timeout}s") from exc
    if proc.returncode != 0:
        raise ToolError(f"git {args[0]} failed: {proc.stderr.strip()[:300]}")
    return proc.stdout


class RepoTools:
    """Read-only inspection tools bound to one repository checkout."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.is_git = (self.root / ".git").exists()
        self._files = self._discover_files()
        self._file_set = set(self._files)
        self.inspected: set[str] = set()  # files the agent has actually looked at
        self.registry: dict[str, tuple[Callable[..., str], dict[str, str]]] = {
            "list_files": (self.list_files, {
                "path": "optional sub-directory to list (default: repository root)",
                "depth": "optional int; collapse directories deeper than this many levels",
            }),
            "read_file": (self.read_file, {
                "path": "repository-relative file path (required)",
                "start_line": "optional 1-based first line (default 1)",
                "end_line": "optional last line (max 400 lines per call)",
            }),
            "search_code": (self.search_code, {
                "pattern": "Python regular expression (required, case-insensitive)",
                "path": "optional directory or file prefix to restrict the search",
                "max_results": "optional int, default 40",
            }),
            "git_log": (self.git_log, {
                "max_count": "optional int, default 20",
                "path": "optional file path to restrict history to",
                "show_files": "optional bool: include changed files per commit",
            }),
            "git_status": (self.git_status, {}),
            "analyze_workflow": (self.analyze_workflow, {
                "path": "optional workflow path under .github/workflows (default: all workflows)",
            }),
            "scan_secrets": (self.scan_secrets, {
                "path": "optional directory or file prefix (default: whole repository)",
            }),
            "find_tracked_artifacts": (self.find_tracked_artifacts, {}),
            "run_ruff": (self.run_ruff, {
                "path": "optional directory or file to lint (default: whole repository)",
            }),
        }

    # ------------------------------------------------------------------ plumbing
    def _discover_files(self) -> list[str]:
        if self.is_git:
            try:
                out = _run_git(self.root, "ls-files", "-z")
                files = [f for f in out.split("\0") if f]
                if files:
                    return sorted(files)
            except ToolError:
                pass
        found: list[str] = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in dirnames if d != ".git"]
            for name in filenames:
                found.append(Path(dirpath, name).relative_to(self.root).as_posix())
        return sorted(found)

    @property
    def files(self) -> list[str]:
        return list(self._files)

    def describe(self) -> dict[str, dict[str, str]]:
        return {name: params for name, (_fn, params) in self.registry.items()}

    def dispatch(self, name: str, args: dict[str, Any] | None) -> str:
        """Run a tool by name. Never raises: errors come back as text for the LLM."""
        if name not in self.registry:
            return f"ERROR: unknown tool {name!r}. Available: {', '.join(self.registry)}"
        fn, params = self.registry[name]
        args = args if isinstance(args, dict) else {}
        unknown = set(args) - set(params)
        if unknown:
            return f"ERROR: {name} got unexpected argument(s) {sorted(unknown)}. Valid: {sorted(params)}"
        try:
            return redact(_cap(fn(**args)))
        except ToolError as exc:
            return f"ERROR: {exc}"
        except TypeError as exc:
            return f"ERROR: bad arguments for {name}: {exc}"
        except Exception as exc:  # noqa: BLE001 - a tool bug must not kill the audit
            return f"ERROR: {name} failed unexpectedly ({type(exc).__name__}: {str(exc)[:200]})"

    def _norm(self, path: str | None) -> str:
        if path is None:
            return ""
        path = str(path).replace("\\", "/").strip()
        while path.startswith("./"):
            path = path[2:]
        return path.strip("/")

    def _safe_path(self, rel: str) -> Path:
        if not rel:
            raise ToolError("path is required")
        if rel.startswith("/") or re.match(r"^[A-Za-z]:", rel) or ".." in Path(rel).parts:
            raise ToolError("path must be repository-relative and stay inside the repository")
        full = (self.root / rel)
        if full.is_symlink():
            raise ToolError("symbolic links are not followed")
        resolved = full.resolve()
        if self.root != resolved and self.root not in resolved.parents:
            raise ToolError("path escapes the repository")
        return resolved

    def _in_scope(self, rel_file: str, prefix: str) -> bool:
        return not prefix or rel_file == prefix or rel_file.startswith(prefix + "/")

    def _read_text(self, rel: str) -> str | None:
        """Return decoded text, or None if the file is binary/oversized/unreadable."""
        try:
            full = self._safe_path(rel)
            if not full.is_file() or full.stat().st_size > MAX_READ_BYTES:
                return None
            data = full.read_bytes()
        except (ToolError, OSError):
            return None
        if b"\0" in data[:4096]:
            return None
        return data.decode("utf-8", errors="replace")

    def file_line_count(self, rel: str) -> int | None:
        text = self._read_text(rel)
        return None if text is None else len(text.splitlines())

    def has_file(self, rel: str) -> bool:
        return rel in self._file_set

    # --------------------------------------------------------------------- tools
    def list_files(self, path: str = "", depth: Any = None) -> str:
        prefix = self._norm(path)
        scoped = [f for f in self._files if self._in_scope(f, prefix)]
        if not scoped:
            raise ToolError(f"no tracked files under {prefix or '.'!r}")
        try:
            limit_depth = int(depth) if depth not in (None, "") else (None if len(scoped) <= 250 else 2)
        except (TypeError, ValueError) as exc:
            raise ToolError("depth must be an integer") from exc
        base = prefix.count("/") + 1 if prefix else 0
        lines: list[str] = []
        collapsed: dict[str, int] = {}
        for rel in scoped:
            parts = rel.split("/")
            if limit_depth is not None and len(parts) - base > limit_depth:
                key = "/".join(parts[: base + limit_depth]) + "/"
                collapsed[key] = collapsed.get(key, 0) + 1
                continue
            try:
                size = (self.root / rel).stat().st_size
            except OSError:
                size = -1
            lines.append(f"{rel}  ({size} B)")
        lines.extend(f"{key}  ({count} more files, collapsed)" for key, count in sorted(collapsed.items()))
        header = f"{len(scoped)} tracked file(s) under {prefix or '.'}"
        return header + "\n" + "\n".join(sorted(lines))

    def read_file(self, path: str = "", start_line: Any = 1, end_line: Any = None) -> str:
        rel = self._norm(path)
        full = self._safe_path(rel)
        if not full.exists():
            hint = self._similar(rel)
            raise ToolError(f"file not found: {rel}" + (f" (did you mean {hint}?)" if hint else ""))
        if not full.is_file():
            raise ToolError(f"{rel} is a directory; use list_files")
        if full.stat().st_size > MAX_READ_BYTES:
            raise ToolError(f"{rel} is larger than {MAX_READ_BYTES} bytes; use search_code instead")
        text = self._read_text(rel)
        if text is None:
            raise ToolError(f"{rel} is binary or unreadable")
        try:
            start = max(1, int(start_line or 1))
            end = int(end_line) if end_line not in (None, "") else None
        except (TypeError, ValueError) as exc:
            raise ToolError("start_line/end_line must be integers") from exc
        all_lines = text.splitlines()
        total = len(all_lines)
        last = min(end if end is not None else total, start + MAX_READ_LINES - 1, total)
        if start > total and total:
            raise ToolError(f"{rel} has only {total} lines")
        self.inspected.add(rel)
        body = "\n".join(f"{n}: {all_lines[n - 1][:400]}" for n in range(start, last + 1))
        note = f"\n[showing lines {start}-{last} of {total}; call again with start_line={last + 1}]" if last < total else ""
        return f"FILE {rel} ({total} lines)\n{body}{note}" if total else f"FILE {rel} is empty"

    def _similar(self, rel: str) -> str | None:
        name = rel.rsplit("/", 1)[-1].lower()
        for candidate in self._files:
            if candidate.rsplit("/", 1)[-1].lower() == name:
                return candidate
        return None

    def search_code(self, pattern: str = "", path: str = "", max_results: Any = 40) -> str:
        if not pattern:
            raise ToolError("pattern is required")
        if len(pattern) > 200:
            raise ToolError("pattern too long (max 200 chars)")
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as exc:
            raise ToolError(f"invalid regular expression: {exc}") from exc
        try:
            limit = max(1, min(int(max_results), 100))
        except (TypeError, ValueError) as exc:
            raise ToolError("max_results must be an integer") from exc
        prefix = self._norm(path)
        hits: list[str] = []
        files_with_hits: set[str] = set()
        for rel in self._files:
            if not self._in_scope(rel, prefix) or Path(rel).suffix.lower() in SKIP_CONTENT_SUFFIXES:
                continue
            text = self._read_text(rel)
            if text is None:
                continue
            for number, line in enumerate(text.splitlines(), start=1):
                if regex.search(line[:500]):
                    hits.append(f"{rel}:{number}: {line.strip()[:200]}")
                    files_with_hits.add(rel)
                    if len(hits) >= limit:
                        break
            if len(hits) >= limit:
                break
        self.inspected.update(files_with_hits)
        if not hits:
            return f"no matches for /{pattern}/ in {prefix or 'repository'}"
        suffix = f"\n[stopped at {limit} results]" if len(hits) >= limit else ""
        return f"{len(hits)} match(es) for /{pattern}/\n" + "\n".join(hits) + suffix

    def git_log(self, max_count: Any = 20, path: str | None = None, show_files: Any = False) -> str:
        if not self.is_git:
            raise ToolError("not a git repository; history is unavailable")
        try:
            count = max(1, min(int(max_count), 100))
        except (TypeError, ValueError) as exc:
            raise ToolError("max_count must be an integer") from exc
        args = ["log", f"-n{count}", "--date=short", "--pretty=format:%h | %ad | %an | %s"]
        if str(show_files).lower() in {"true", "1", "yes"}:
            args.append("--name-status")
        rel = self._norm(path)
        if rel:
            self._safe_path(rel)
            args += ["--", rel]
        out = _run_git(self.root, *args)
        shallow = (self.root / ".git" / "shallow").exists()
        note = "\n[history is shallow: older commits are not available]" if shallow else ""
        return (out or "no commits") + note

    def git_status(self) -> str:
        if not self.is_git:
            raise ToolError("not a git repository")
        status = _run_git(self.root, "status", "--porcelain").strip()
        head = _run_git(self.root, "rev-parse", "--short", "HEAD").strip()
        branch = _run_git(self.root, "rev-parse", "--abbrev-ref", "HEAD").strip()
        return f"branch={branch} HEAD={head}\n" + (status or "working tree clean")

    def analyze_workflow(self, path: str = "") -> str:
        rel = self._norm(path)
        workflows = [f for f in self._files
                     if f.startswith(".github/workflows/") and f.endswith((".yml", ".yaml"))]
        if rel:
            if rel not in self._file_set:
                raise ToolError(f"workflow not found: {rel}")
            workflows = [rel]
        if not workflows:
            return ("no workflow files under .github/workflows/ (other CI systems may exist: check for "
                    "other CI configuration such as .gitlab-ci.yml, Jenkinsfile, .circleci/, azure-pipelines.yml)")
        parts = []
        for wf in workflows[:12]:
            text = self._read_text(wf)
            if text is None:
                parts.append(f"{wf}: unreadable")
                continue
            self.inspected.add(wf)
            parts.append(summarize_workflow(wf, text))
        if len(workflows) > 12:
            parts.append(f"... {len(workflows) - 12} more workflow(s) not shown; pass path=")
        return "\n\n".join(parts)

    def scan_secrets(self, path: str = "") -> str:
        prefix = self._norm(path)
        findings: list[str] = []
        for rel in self._files:
            if not self._in_scope(rel, prefix) or Path(rel).suffix.lower() in SKIP_CONTENT_SUFFIXES:
                continue
            text = self._read_text(rel)
            if text is None:
                continue
            for hit in find_secrets(text):
                findings.append(f"{rel}:{hit.line}: {hit.kind} (value hidden)")
                self.inspected.add(rel)
            if len(findings) >= 60:
                break
        if not findings:
            return "no credential-like values detected by the pattern scanner (it is heuristic)"
        return (f"{len(findings)} credential-like value(s) detected; values are never shown. "
                "Read the surrounding lines with read_file to judge whether they are real, "
                "test fixtures or placeholders.\n" + "\n".join(findings))

    def find_tracked_artifacts(self) -> str:
        matches: list[str] = []
        for rel in self._files:
            name = rel.rsplit("/", 1)[-1]
            if name in TEMPLATE_ENV_NAMES:
                continue
            for pattern, why in ARTIFACT_PATTERNS:
                if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(name, pattern):
                    matches.append(f"{rel}  <- {why}")
                    self.inspected.add(rel)
                    break
        gitignore = ".gitignore" in self._file_set
        head = f".gitignore present: {gitignore}\n"
        if not matches:
            return head + "no tracked files match common temporary/generated/local-only patterns"
        shown = matches[:80]
        extra = f"\n... {len(matches) - 80} more" if len(matches) > 80 else ""
        return head + f"{len(matches)} tracked file(s) match common temporary/generated patterns:\n" \
            + "\n".join(shown) + extra

    def run_ruff(self, path: str = "") -> str:
        ruff = shutil.which("ruff")
        cmd = [ruff] if ruff else [sys.executable, "-m", "ruff"]
        rel = self._norm(path)
        target = self._safe_path(rel) if rel else self.root
        py_files = [f for f in self._files if f.endswith(".py") and self._in_scope(f, rel)]
        if not py_files:
            return "no Python files in scope; ruff not run"
        has_config = any(f in self._file_set for f in ("ruff.toml", ".ruff.toml")) or (
            "pyproject.toml" in self._file_set and "[tool.ruff" in (self._read_text("pyproject.toml") or ""))
        args = [*cmd, "check", "--no-cache", "--exit-zero", "--output-format", "concise", str(target)]
        if not has_config:
            args.insert(len(cmd) + 1, "--isolated")
        try:
            proc = subprocess.run(args, cwd=self.root, capture_output=True, text=True, encoding="utf-8",
                                  errors="replace", timeout=RUFF_TIMEOUT, check=False)
        except FileNotFoundError:
            return "UNAVAILABLE: ruff is not installed in the auditor environment; no lint result"
        except subprocess.TimeoutExpired:
            return f"UNAVAILABLE: ruff timed out after {RUFF_TIMEOUT}s"
        if "No module named ruff" in proc.stderr:
            return "UNAVAILABLE: ruff is not installed in the auditor environment; no lint result"
        out = proc.stdout.strip()
        mode = "repository ruff config" if has_config else "default rules (--isolated; repo has no ruff config)"
        lines = out.splitlines()
        shown = "\n".join(lines[:60])
        more = f"\n... {len(lines) - 60} more diagnostics" if len(lines) > 60 else ""
        return f"ruff over {len(py_files)} python file(s) using {mode}:\n{shown or 'no diagnostics'}{more}"
