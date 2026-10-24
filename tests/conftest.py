"""Shared fixtures: tiny throw-away git repositories with known content."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable

import pytest

GIT = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false"]

HEALTHY_WORKFLOW = """\
name: CI
on:
  push:
    branches: [main]
  pull_request:
permissions:
  contents: read
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install -r requirements.txt
      - run: ruff check .
      - run: pytest
"""

BAD_WORKFLOW = """\
name: CI
on:
  pull_request_target:
permissions: write-all
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          ref: ${{ github.event.pull_request.head.sha }}
      - uses: some-org/some-action@main
      - run: echo "${{ github.event.pull_request.title }}"
      - run: pytest || true
        continue-on-error: true
      - run: lint
        if: false
"""


def _git(root: Path, *args: str) -> None:
    subprocess.run([*GIT, "-C", str(root), *args], check=True, capture_output=True)


@pytest.fixture
def make_repo(tmp_path: Path) -> Callable[[dict[str, str]], Path]:
    def _make(files: dict[str, str]) -> Path:
        root = tmp_path / "repo"
        root.mkdir()
        _git(root, "init", "-q", "-b", "main")
        for rel, content in files.items():
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        _git(root, "add", "-A")
        _git(root, "commit", "-q", "-m", "initial commit")
        return root

    return _make


@pytest.fixture
def sample_repo(make_repo) -> Path:
    return make_repo({
        "README.md": "# demo\n",
        "app.py": "import os\n\nDB = os.environ['DB_URL']\nPATH = '/home/alice/data'\n",
        "config.py": 'API_KEY = "sk-live-aB3dE5gH7jK9mN1pQ3sT5vX7zA9cE1"\n',
        "requirements.txt": "flask\nrequests>=2\n",
        ".env": "SECRET_KEY=abcdef123456789xyz\n",
        ".env.example": "SECRET_KEY=your-key-here\n",
        "src/__pycache__/x.pyc": "junk",
        ".github/workflows/ci.yml": BAD_WORKFLOW,
        ".github/workflows/good.yml": HEALTHY_WORKFLOW,
    })
