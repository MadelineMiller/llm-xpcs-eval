"""
Centralized logging for the XPCS LLM app.

Three log files under logs/:
  access.log  — logins, session starts/ends
  queries.log — per-question record (JSONL): user, question, retrieval stats, sources, answer
  errors.log  — reranker failures, API errors, exceptions
"""

import json
import logging
import logging.handlers
import os
from datetime import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)


def _make_logger(name: str, filename: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = logging.handlers.RotatingFileHandler(
            os.path.join(LOG_DIR, filename),
            maxBytes=10 * 1024 * 1024,  # 10 MB per file
            backupCount=5,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    return logger


_access  = _make_logger("xpcs.access",  "access.log")
_queries = _make_logger("xpcs.queries", "queries.log")
_errors  = _make_logger("xpcs.errors",  "errors.log")


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ── Access log ────────────────────────────────────────────────────────────────

def log_login_success(username: str, full_name: str, email: str):
    _access.info(f"{_ts()} | LOGIN_OK    | user={username} | name={full_name} | email={email}")


def log_login_failure(username: str, reason: str):
    _access.info(f"{_ts()} | LOGIN_FAIL  | user={username} | reason={reason}")
    _errors.error(f"{_ts()} | LOGIN_FAIL  | user={username} | reason={reason}")


def log_session_start(username: str):
    _access.info(f"{_ts()} | SESSION_START | user={username}")


def log_session_end(username: str):
    _access.info(f"{_ts()} | SESSION_END   | user={username}")


# ── Query log (JSONL) ─────────────────────────────────────────────────────────

def log_query(
    username: str,
    question: str,
    keywords: list,
    vec_count: int,
    kw_added: int,
    adj_added: int,
    kept: int,
    sources: list,   # list of "Title (page N)" strings
    context: str,
    answer: str,
):
    record = {
        "timestamp": _ts(),
        "user": username,
        "question": question,
        "keywords": keywords,
        "retrieval": {
            "vector_candidates": vec_count,
            "keyword_added": kw_added,
            "adjacent_added": adj_added,
            "total": vec_count + kw_added + adj_added,
            "kept_after_rerank": kept,
        },
        "sources_used": sources,
        "context": context,
        "answer": answer,
    }
    _queries.info(json.dumps(record, ensure_ascii=False))


# ── Error log ─────────────────────────────────────────────────────────────────

def log_reranker_empty(raw_response: str):
    _errors.warning(f"{_ts()} | RERANKER_EMPTY | raw={raw_response[:300]}")

def log_reranker_fallback(raw_response: str):
    _errors.error(f"{_ts()} | RERANKER_PARSE_FAIL | raw={raw_response[:300]}")


def log_reranker_error(exc: Exception):
    _errors.error(f"{_ts()} | RERANKER_ERROR | {exc}")


def log_api_error(label: str, status_code, detail: str):
    _errors.error(f"{_ts()} | API_ERROR | {label} | status={status_code} | {detail[:300]}")


def log_api_network_error(label: str, exc: Exception):
    _errors.error(f"{_ts()} | API_NETWORK_ERROR | {label} | {exc}")


def log_error(context: str, detail: str):
    _errors.error(f"{_ts()} | ERROR | {context} | {detail}")
