"""Secret detection and redaction.

Everything that leaves the process (LLM prompts, traces, the JSON report) goes
through :func:`redact`, so a credential found in an audited repository is never
copied into any output. :func:`find_secrets` exposes the same patterns to the
``scan_secrets`` tool, which reports *where* a secret is without showing it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

REDACTED = "[REDACTED]"

# (kind, compiled pattern). The whole match is replaced.
_TOKEN_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("private key block", re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(?:-----END [A-Z ]*PRIVATE KEY-----|\Z)")),
    ("AWS access key id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("GitHub token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b")),
    ("Slack token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("Google API key (new format)", re.compile(r"\bAQ\.[A-Za-z0-9_-]{20,}")),
    ("Stripe key", re.compile(r"\b[sr]k_live_[0-9A-Za-z]{16,}\b")),
    ("OpenAI-style key", re.compile(r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}\b")),
    ("Groq key", re.compile(r"\bgsk_[A-Za-z0-9]{20,}\b")),
    ("JSON web token", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")),
    ("partial key prefix", re.compile(
        r"\b(?:AIza|AKIA|ASIA|gh[pousr]_|github_pat_|xox[abprs]-|sk-|gsk_|[sr]k_live_)[A-Za-z0-9_\-]{0,6}(?:\.{2,}|…)")),
    ("URL with embedded password", re.compile(r"(?<=://)[^/\s:@]+:[^/\s:@]{3,}(?=@[\w.-]+)")),
]

# name = "value" style assignments where the name suggests a credential.
_ASSIGNMENT = re.compile(
    r"""(?ix)
    (?P<prefix>
        [\w.\-]*
        (?:password|passwd|pwd|secret|token|api[_\-]?key|apikey|access[_\-]?key|
           private[_\-]?key|client[_\-]?secret|credential)
        [\w.\-]*
        ["']?\s*[:=]\s*
    )
    (?P<quote>["']?)
    (?P<value>[^\s"'#,;`]{6,})
    """
)

# Literal fallback in an env lookup: os.getenv("API_KEY", "literal"), process.env.TOKEN || "literal".
_ENV_DEFAULT = re.compile(
    r"""(?ix)
    (?P<prefix>
        (?:getenv|environ\.get|env\.get|config\.get|\.get)\(\s*["'][^"']*
        (?:key|secret|token|password|passwd|credential)[^"']*["']\s*,\s*
      |
        process\.env\.\w*(?:key|secret|token|password|credential)\w*\s*(?:\|\||\?\?)\s*
    )
    (?P<quote>["'])
    (?P<value>[^"']{6,})
    (?P=quote)
    """
)

_PLACEHOLDER_HINTS = (
    "changeme", "change-me", "change_me", "example", "placeholder", "your-", "your_", "yourkey",
    "xxxx", "****", "dummy", "sample", "<", "todo", "none", "null", "true", "false", "redacted",
)
_REFERENCE_PREFIXES = ("${", "$(", "$", "{{", "%", "os.", "os[", "process.env", "env.", "env(",
                       "getenv", "secrets.", "vars.", "settings.", "config.", "config[", "self.",
                       "args.", "request.", "input(", "getpass", "environ", "std::getenv")


@dataclass(frozen=True)
class SecretHit:
    line: int
    kind: str


def _is_placeholder(value: str) -> bool:
    lowered = value.lower()
    if lowered.startswith(_REFERENCE_PREFIXES):
        return True
    if any(hint in lowered for hint in _PLACEHOLDER_HINTS):
        return True
    # Function calls / attribute lookups are code, not literals.
    if "(" in value or value.endswith(")") or re.fullmatch(r"[A-Za-z_][\w.]*\[?", value) and "." in value:
        return True
    # A real secret has some entropy: require at least two character classes or length >= 12.
    classes = sum(bool(re.search(p, value)) for p in (r"[a-z]", r"[A-Z]", r"\d", r"[^\w]"))
    return classes < 2 and len(value) < 12


_known_literals: set[str] = set()


def register_secret(value: str | None) -> None:
    """Make sure a runtime secret (the LLM API key) is scrubbed from every output."""
    if value and len(value) >= 6:
        _known_literals.add(value)


def redact(text: str) -> str:
    """Return ``text`` with credential-looking values masked."""
    if not text:
        return text
    for literal in _known_literals:
        text = text.replace(literal, REDACTED)
    for _kind, pattern in _TOKEN_PATTERNS:
        text = pattern.sub(REDACTED, text)

    def _mask_assignment(match: re.Match[str]) -> str:
        value = match.group("value")
        if _is_placeholder(value):
            return match.group(0)
        return f"{match.group('prefix')}{match.group('quote')}{REDACTED}"

    def _mask_env_default(match: re.Match[str]) -> str:
        if _is_placeholder(match.group("value")):
            return match.group(0)
        return f"{match.group('prefix')}{match.group('quote')}{REDACTED}{match.group('quote')}"

    return _ASSIGNMENT.sub(_mask_assignment, _ENV_DEFAULT.sub(_mask_env_default, text))


def find_secrets(text: str) -> list[SecretHit]:
    """Locate credential-looking values; never returns the values themselves."""
    hits: list[SecretHit] = []
    # Multi-line private key blocks first (anchor to the BEGIN line).
    for match in _TOKEN_PATTERNS[0][1].finditer(text):
        hits.append(SecretHit(text.count("\n", 0, match.start()) + 1, _TOKEN_PATTERNS[0][0]))
    for number, line in enumerate(text.splitlines(), start=1):
        if len(line) > 2000:
            continue
        matched = False
        for kind, pattern in _TOKEN_PATTERNS[1:]:
            if pattern.search(line):
                hits.append(SecretHit(number, kind))
                matched = True
                break
        if matched:
            continue
        for pattern in (_ENV_DEFAULT, _ASSIGNMENT):
            if any(not _is_placeholder(m.group("value")) for m in pattern.finditer(line)):
                hits.append(SecretHit(number, "credential-like assignment"))
                break
    return hits
