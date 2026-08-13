import json
import os
import time

TOKEN_MAX_AGE = 15 * 86400  # 15 days, matches chainlit user_session_timeout

_STATE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ".admin_tokens.json",
)


def _load() -> dict:
    try:
        with open(_STATE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save() -> None:
    tmp = _STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(admin_auth_tokens, f)
    os.replace(tmp, _STATE_FILE)


# Shared token store between app.py and admin.py
admin_auth_tokens: dict = _load()

# Drop expired tokens on import so the state file doesn't grow unbounded
_now = time.time()
_expired = [t for t, ts in admin_auth_tokens.items() if _now - ts > TOKEN_MAX_AGE]
for _t in _expired:
    admin_auth_tokens.pop(_t, None)
if _expired:
    _save()


def add_token(token):
    admin_auth_tokens[token] = time.time()
    _save()


def remove_token(token):
    if admin_auth_tokens.pop(token, None) is not None:
        _save()


def is_valid(token):
    if token not in admin_auth_tokens:
        return False
    if time.time() - admin_auth_tokens[token] > TOKEN_MAX_AGE:
        admin_auth_tokens.pop(token, None)
        _save()
        return False
    return True
