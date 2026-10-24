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
