"""Publication redaction must remove credentials without destroying trace evidence."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('publication_redaction', Path(__file__).resolve().parents[2] / 'scripts/redaction.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


@pytest.fixture(autouse=True)
def synthetic_secrets(monkeypatch):
    monkeypatch.setattr(r, 'SECRET_VALUES', ['synthetic-secret-only-for-testing'])


@pytest.mark.parametrize('text', [
    'export OPENAI_API_KEY="short secret with spaces"',
    "api_key = 'lowercase-secret'",
    'authorization: Bearer abcdefghijklmnopqrstuvwxyz',
    'https://user:password@example.org/path?api_key=secret-value&x=2',
    '-----BEGIN RSA PRIVATE KEY-----\nsecret material\n-----END RSA PRIVATE KEY-----',
    'eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjMifQ.signature',
    'alice@example.org /Users/alice/projects/solution.py 10.12.13.14 2001:db8::1',
    'synthetic-secret-only-for-testing',
    '# AGENTS.md instructions\nprivate personal instructions',
    'cat ~/.codex/auth.json\nprivate configuration',
])
def test_secret_and_private_patterns(text):
    assert r.scan_text(text)
    clean = r.redact_text(text)
    assert not r.scan_text(clean)
    assert clean != text


def test_recursive_json_preserves_numbers_and_kernel_code():
    source = {'apiKey': 'short-secret', 'timeout': 300000, 'input_tokens': 18000,
              'ok': True, 'n': None, 'speedup': 1.5,
              'nested': [{'password': 123456, 'refreshToken': 'other-secret'}],
              'code': 'def kernel(x):\n    return x * 2\n',
              'synthetic-secret-only-for-testing': 'key also private'}
    clean = r.redact_jsonable(source)
    assert clean['timeout'] == 300000
    assert clean['input_tokens'] == 18000
    assert clean['ok'] is True and clean['n'] is None
    assert clean['speedup'] == 1.5 and clean['code'] == source['code']
    assert clean['nested'][0]['password'] == 0
    assert clean['apiKey'] == r.REDACTED
    assert not r.scan_jsonable(clean)
    assert 'synthetic-secret-only-for-testing' not in json.dumps(clean)
    assert source['apiKey'] == 'short-secret'


def test_embedded_json_stays_valid():
    source = '{"api_key": "short-secret", "timeout": 300000, "env": {"TOKEN": "abcdef"}}'
    clean = json.loads(r.redact_text(source))
    assert clean['timeout'] == 300000
    assert clean['api_key'] == r.REDACTED
    assert not r.scan_jsonable(clean)


def test_scan_file_does_not_mutate_or_disclose(tmp_path):
    path = tmp_path / 'trace.jsonl'
    path.write_text(json.dumps({'api_key': 'abcdef', 'content': 'synthetic-secret-only-for-testing'}) + '\n')
    original = path.read_bytes()
    counts = r.scan_file(path)
    assert counts['credential_field'] == 1 and counts['known_secret'] == 1
    assert 'abcdef' not in repr(counts)
    assert path.read_bytes() == original
    r.redact_jsonl_file(path)
    assert r.scan_file(path) == {}
    assert json.loads(path.read_text())['api_key'] == r.REDACTED


def test_known_values_only_read_sensitive_names(tmp_path, monkeypatch):
    (tmp_path / '.env_vars').write_text('export API_KEY="a long secret"\nCONTEXT_WINDOW=300000\nMODEL_NAME=technical-evidence\n')
    (tmp_path / '.codex').mkdir()
    (tmp_path / '.codex/auth.json').write_text(json.dumps({'tokens': {'access_token': 'opaque-access-value'}}))
    monkeypatch.setattr(r.Path, 'home', classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(r.os, 'environ', {'lower_api_key': 'environment-secret', 'TOKENS': '300000'})
    values = r._candidate_secret_values()
    assert 'a long secret' in values and 'opaque-access-value' in values
    assert 'environment-secret' in values
    assert '300000' not in values and 'technical-evidence' not in values


def test_media_assignment_gate_is_clean():
    import re
    text = r.redact_text('GOG_KEYRING_PASSWORD="abcdef" OPENAI_API_KEY=$OPENAI_API_KEY')
    assert not re.search(r'[A-Z0-9_]*(API_KEY|TOKEN|SECRET|PASSWORD)=', text)


def test_ordinary_tool_output_kept():
    text = 'nvcc 13.2\ncheck passed\nspeedup=2.8\ninput_tokens=10000\nblock_size=256\n'
    assert r.redact_text(text) == text
    assert r.scan_text(text) == {}


@pytest.mark.parametrize('command', ['cat ~/.env_vars', '# AGENTS.md instructions\nprivate prose'])
def test_json_tool_args_containing_private_material_stay_parseable(command):
    args = {'command': command, 'timeout': 300000}
    clean = json.loads(r.redact_text(json.dumps(args)))
    assert clean['timeout'] == 300000
    assert clean['command'].startswith('[REDACTED')
    assert not r.scan_jsonable(clean)


def test_nested_encoded_tool_args_are_finite_and_preserve_types():
    args = {'body': json.dumps({'apiKey': 'credential', 'count': 64}), 'args': {'command': 'nvcc solution.cu'}}
    clean = json.loads(r.redact_text(json.dumps(args)))
    assert json.loads(clean['body']) == {'apiKey': r.REDACTED, 'count': 64}
    assert clean['args']['command'] == 'nvcc solution.cu'



def test_embedded_json_triton_decorator_is_not_email():
    code = 'import triton\n\n@triton.jit\ndef kernel(): pass'
    encoded = json.dumps({'output': code})
    assert r.scan_text(encoded) == {}
    assert json.loads(r.redact_text(encoded))['output'] == code
    assert r.scan_text(json.dumps({'output': 'contact alice@example.org'})) == {'email': 1}


@pytest.mark.parametrize('text', [
    '-L/home/infatoshi/kernelbench.com/repo/.venv/lib',   # linker -L path
    '-I/Users/bob/include',
    'export LD_LIBRARY_PATH=/home/bob/lib',
    '"cat /home/bob/.env"',
    '-L/root/x',
])
def test_home_path_after_a_flag_is_redacted(text):
    """A home path preceded by a short flag letter must still be redacted.

    The old word-char lookbehind let `-L/home/user/...` through because `L`
    is a word character; one such string was live in a published viewer.
    """
    assert r.HOME_RE.search(text), f'{text!r} should match a home path'
    red = r.redact_text(text)
    assert '/home/' not in red, f'{text!r} left a home path: {red!r}'


@pytest.mark.parametrize('text', [
    'foo/home/x',        # path segment, not a home dir
    'see:homepage',
    'x/home/y',
])
def test_relative_segments_are_not_treated_as_home_paths(text):
    assert not r.HOME_RE.search(text), f'{text!r} should not match'
