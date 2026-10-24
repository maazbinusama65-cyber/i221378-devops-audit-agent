"""Safe acquisition of the target repository.

The target is cloned (shallow, hooks disabled, symlinks materialised as plain files,
LFS smudge off) into a private temp directory and only ever read. The original
remote is never written to.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from tools import RepoTools, _run_git

GITHUB_URL = re.compile(
    r"^https://(?:www\.)?github\.com/(?P<owner>[A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))/"
    r"(?P<repo>[A-Za-z0-9._-]{1,100}?)(?:\.git)?(?:/(?:tree|blob)/.*)?/?$"
)
CLONE_TIMEOUT = 240
CLONE_DEPTH = 100
MAX_CHECKOUT_MB = 400


class RepoError(Exception):
    """The repository could not be fetched or is unusable."""


def parse_github_url(url: str) -> tuple[str, str]:
    match = GITHUB_URL.match(url.strip())
    if not match:
        raise RepoError(f"not a public GitHub repository URL: {url!r} "
                        "(expected https://github.com/<owner>/<repo>)")
    repo = match.group("repo")
    if repo in {".", ".."}:
        raise RepoError(f"invalid repository name in {url!r}")
    return match.group("owner"), repo


def _force_remove(func, path, _exc) -> None:  # noqa: ANN001 - shutil callback signature
    os.chmod(path, stat.S_IWRITE)
    func(path)


def remove_tree(path: Path) -> None:
    """Delete a directory tree, including read-only git objects on Windows."""
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=_force_remove)
    else:
        shutil.rmtree(path, onerror=_force_remove)


@contextmanager
def checkout(target: str) -> Iterator[Path]:
    """Yield a read-only working copy of ``target`` (GitHub URL, or an existing local path).

    Local paths exist for development/tests and are used in place, untouched.
    """
    local = Path(target)
    if not target.startswith(("http://", "https://")) and local.is_dir():
        yield local.resolve()
        return
    if target.startswith("http://"):
        raise RepoError("only https GitHub URLs are supported")
    owner, repo = parse_github_url(target)
    workdir = Path(tempfile.mkdtemp(prefix="devops-audit-"))
    dest = workdir / repo
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_LFS_SKIP_SMUDGE": "1"}
    cmd = [
        "git", "-c", "core.hooksPath=", "-c", "core.symlinks=false", "-c", "protocol.file.allow=never",
        "-c", "credential.helper=", "-c", "filter.lfs.smudge=", "-c", "filter.lfs.process=",
        "-c", "filter.lfs.required=false",
        "clone", "--quiet", "--depth", str(CLONE_DEPTH), "--no-tags", "--single-branch",
        f"https://github.com/{owner}/{repo}.git", str(dest),
    ]
    try:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                  timeout=CLONE_TIMEOUT, env=env, check=False)
        except FileNotFoundError as exc:
            raise RepoError("git is not installed or not on PATH") from exc
        except subprocess.TimeoutExpired as exc:
            raise RepoError(f"cloning {owner}/{repo} timed out after {CLONE_TIMEOUT}s") from exc
        if proc.returncode != 0:
            detail = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "unknown error"
            raise RepoError(f"could not clone {owner}/{repo} (private, missing or network error): {detail[:200]}")
        size_mb = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file()) / 1_048_576
        if size_mb > MAX_CHECKOUT_MB:
            raise RepoError(f"repository checkout is {size_mb:.0f} MB, above the {MAX_CHECKOUT_MB} MB limit")
        yield dest
    finally:
        remove_tree(workdir)


def build_overview(tools: RepoTools, url: str) -> str:
    """The agent's first observation: facts only, no conclusions."""
    files = tools.files
    suffixes = Counter(Path(f).suffix.lower() or Path(f).name for f in files)
    top = sorted({f.split("/")[0] + ("/" if "/" in f else "") for f in files})
    lines = [f"repository: {url}", f"tracked files: {len(files)}"]
    if tools.is_git:
        try:
            lines.append("git: " + _run_git(tools.root, "rev-parse", "--abbrev-ref", "HEAD").strip()
                         + " @ " + _run_git(tools.root, "rev-parse", "--short", "HEAD").strip()
                         + ", commits visible: " + _run_git(tools.root, "rev-list", "--count", "HEAD").strip()
                         + (" (shallow clone)" if (tools.root / ".git" / "shallow").exists() else ""))
        except Exception:  # noqa: BLE001 - overview is best effort
            lines.append("git: metadata unavailable")
    lines.append("most common file types: " + ", ".join(f"{k}({v})" for k, v in suffixes.most_common(8)))
    lines.append("top-level entries: " + ", ".join(top[:80]))
    return "\n".join(lines)
