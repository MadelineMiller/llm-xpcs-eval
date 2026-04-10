import time

# Shared token store between app.py and admin.py
admin_auth_tokens = {}
TOKEN_MAX_AGE = 86400  # 24 hours

def add_token(token):
    admin_auth_tokens[token] = time.time()

def remove_token(token):
    admin_auth_tokens.pop(token, None)

def is_valid(token):
    if token not in admin_auth_tokens:
        return False
    if time.time() - admin_auth_tokens[token] > TOKEN_MAX_AGE:
        admin_auth_tokens.pop(token, None)
        return False
    return True
