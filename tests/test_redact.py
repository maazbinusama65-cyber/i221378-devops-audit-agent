from redact import find_secrets, redact, register_secret


def test_masks_known_token_formats():
    text = "aws=AKIAABCDEFGHIJKLMNOP gh=ghp_" + "a" * 36
    out = redact(text)
    assert "AKIAABCDEFGHIJKLMNOP" not in out
    assert "ghp_" not in out
    assert out.count("[REDACTED]") == 2


def test_masks_credential_assignment_but_keeps_name():
    out = redact('password = "Tr0ub4dor&3xyz"')
    assert "Tr0ub4dor" not in out
    assert out.startswith("password")


def test_placeholders_and_references_are_not_masked():
    for line in ('API_KEY = os.environ["API_KEY"]', "token: ${{ secrets.GH_TOKEN }}",
                 "SECRET_KEY=your-key-here", 'password = "changeme"'):
        assert redact(line) == line


def test_private_key_block_is_masked():
    block = "-----BEGIN RSA PRIVATE KEY-----\nMIIEabc\n-----END RSA PRIVATE KEY-----"
    assert "MIIE" not in redact(block)


def test_runtime_secret_is_scrubbed_everywhere():
    register_secret("super-secret-runtime-key")
    assert "super-secret" not in redact("header: super-secret-runtime-key")


def test_find_secrets_reports_lines_not_values():
    hits = find_secrets('x = 1\nTOKEN = "ghp_' + "b" * 36 + '"\n')
    assert [(h.line, h.kind) for h in hits] == [(2, "GitHub token")]
    assert find_secrets("def f(token):\n    return token\n") == []


def test_partial_key_prefixes_with_ellipsis_are_masked():
    assert "AIza" not in redact('key is "AIzaSyC..." in config')
    assert "ghp_" not in redact("starts with ghp_ab…")


def test_env_lookup_with_literal_default_is_masked():
    fake = "AQ." + "Zx9" * 14
    out = redact(f'GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "{fake}")')
    assert fake not in out and 'os.getenv("GEMINI_API_KEY", "[REDACTED]")' in out
    assert "Zx9Zx9" not in redact('token = os.environ.get("API_TOKEN", "Qw3rty9Uiop8Asdf")')
    assert "Qw3rty" not in redact('const k = process.env.API_KEY || "Qw3rty9Uiop8Asdf"')


def test_env_lookup_without_literal_or_with_placeholder_is_kept():
    for line in ('k = os.getenv("API_KEY")', 'k = os.getenv("API_KEY", "")', 'k = os.getenv("API_KEY", "changeme")',
                 'k = os.getenv("API_KEY", default_value)'):
        assert redact(line) == line


def test_new_format_google_key_is_masked_and_found():
    fake = "AQ." + "Zx9" * 14
    assert fake not in redact(f"see {fake} here")
    assert [h.line for h in find_secrets(f"a\nKEY={fake}\n")] == [2]
    assert find_secrets('k = os.getenv("API_KEY", "Qw3rty9Uiop8Asdf")')
