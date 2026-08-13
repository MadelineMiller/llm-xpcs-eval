"""
Periodic XPCS paper monitor.

Polls arXiv + CrossRef for new papers matching a keyword (default "XPCS"),
uses the Argo LLM to score each candidate for relevance, and emails a digest
of the relevant ones to a configured address via Gmail SMTP.

Runs either as a background daemon thread inside the Chainlit process
(via start_background(), called from app.py) or as a standalone CLI
(python -m agent.paper_monitor --once --dry-run).
"""

from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import os
import re
import smtplib
import sys
import threading
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from html import escape
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import call_argo_llm  # noqa: E402
from config import LLM_CONFIG  # noqa: E402
import logger as applog  # noqa: E402


# ── Config (env-driven) ────────────────────────────────────────────────────────

KEYWORDS         = [k.strip() for k in os.getenv("PAPER_MONITOR_KEYWORDS", "XPCS").split(",") if k.strip()]
TO_ADDRS         = [a.strip() for a in os.getenv("PAPER_MONITOR_TO", "momiller@anl.gov").split(",") if a.strip()]
RUN_AT_HHMM      = os.getenv("PAPER_MONITOR_RUN_AT", "06:00")  # 24-hour HH:MM in RUN_TZ
RUN_TZ_NAME      = os.getenv("PAPER_MONITOR_TZ", "America/Chicago")
MIN_SCORE        = float(os.getenv("PAPER_MONITOR_MIN_SCORE", "6"))
GMAIL_ADDRESS    = os.getenv("GMAIL_ADDRESS", "")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")


def _parse_hhmm(s: str) -> tuple[int, int]:
    hh, mm = s.strip().split(":")
    return int(hh), int(mm)


def _next_run_at(now_utc: datetime) -> datetime:
    """Next occurrence of RUN_AT_HHMM in RUN_TZ_NAME, strictly after now_utc."""
    tz = ZoneInfo(RUN_TZ_NAME)
    now_local = now_utc.astimezone(tz)
    hh, mm = _parse_hhmm(RUN_AT_HHMM)
    target = now_local.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if target <= now_local:
        target += timedelta(days=1)
    return target

STATE_FILE = Path(__file__).parent / ".paper_monitor_state.json"
CROSSREF_UA = "XPCS-Harvester-Bot/1.0 (mailto:momiller@anl.gov)"
MAX_SEEN = 5000

# ── Per-tick JSONL log (persistent audit trail) ────────────────────────────────
_run_log = logging.getLogger("paper_monitor.runs")
if not _run_log.handlers:
    os.makedirs("logs", exist_ok=True)
    _h = logging.handlers.RotatingFileHandler(
        "logs/paper_monitor.log",
        maxBytes=10 * 1024 * 1024,   # 10 MB
        backupCount=10,               # keep ~100 MB history
        encoding="utf-8",
    )
    _h.setFormatter(logging.Formatter("%(message)s"))
    _run_log.addHandler(_h)
    _run_log.setLevel(logging.INFO)
    _run_log.propagate = False


def _log_run(meta: dict, included: list[dict], candidates_total: int, delivered: dict) -> None:
    """Append one JSONL record per real (non-dry-run) tick."""
    record = {
        "timestamp":         datetime.now(timezone.utc).isoformat(),
        "candidates_total":  candidates_total,
        "included_count":    len(included),
        "delivered":         delivered,       # {"email": bool, "slack": bool}
        "meta":              meta,
        "papers": [
            {
                "source":  p.get("source"),
                "id":      p.get("id"),
                "doi":     p.get("doi", ""),
                "title":   p.get("title"),
                "authors": p.get("authors", []),
                "venue":   p.get("venue", ""),
                "year":    p.get("year", ""),
                "url":     p.get("url"),
                "score":   p.get("score"),
                "reason":  p.get("reason", ""),
                "summary": p.get("summary", ""),
            }
            for p in included
        ],
    }
    _run_log.info(json.dumps(record, ensure_ascii=False))


# ── State ──────────────────────────────────────────────────────────────────────

def _load_state() -> dict:
    try:
        with open(STATE_FILE) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}
    data.setdefault("last_run", None)
    data.setdefault("seen_dois", [])
    data.setdefault("seen_arxiv_ids", [])
    data.setdefault("seen_other_ids", [])
    return data


def _save_state(state: dict) -> None:
    state["seen_dois"]       = state["seen_dois"][-MAX_SEEN:]
    state["seen_arxiv_ids"]  = state["seen_arxiv_ids"][-MAX_SEEN:]
    state["seen_other_ids"]  = state["seen_other_ids"][-MAX_SEEN:]
    tmp = str(STATE_FILE) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_FILE)


# ── Sources ────────────────────────────────────────────────────────────────────

ARXIV_NS = {
    "atom":  "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


def fetch_arxiv(keyword: str, max_results: int = 50) -> list[dict]:
    """Fetch recent arXiv submissions matching `keyword`, sorted newest first."""
    url = (
        "http://export.arxiv.org/api/query"
        f"?search_query=all:{requests.utils.quote(keyword)}"
        f"&sortBy=submittedDate&sortOrder=descending&max_results={max_results}"
    )
    try:
        resp = requests.get(url, timeout=30, headers={"User-Agent": CROSSREF_UA})
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        applog.log_api_network_error("paper_monitor.arxiv", e)
        return []

    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError as e:
        applog.log_error("paper_monitor.arxiv_parse", str(e))
        return []

    papers = []
    for entry in root.findall("atom:entry", ARXIV_NS):
        arxiv_id_full = (entry.findtext("atom:id", "", ARXIV_NS) or "").strip()
        arxiv_id = arxiv_id_full.rsplit("/", 1)[-1] if arxiv_id_full else ""
        if not arxiv_id:
            continue
        title = _clean_ws(entry.findtext("atom:title", "", ARXIV_NS) or "")
        abstract = _clean_ws(entry.findtext("atom:summary", "", ARXIV_NS) or "")
        published = (entry.findtext("atom:published", "", ARXIV_NS) or "").strip()
        doi = _clean_ws(entry.findtext("arxiv:doi", "", ARXIV_NS) or "").lower()
        authors = [
            (a.findtext("atom:name", "", ARXIV_NS) or "").strip()
            for a in entry.findall("atom:author", ARXIV_NS)
        ]
        papers.append({
            "source":    "arxiv",
            "id":        arxiv_id,
            "doi":       doi,
            "title":     title,
            "authors":   [a for a in authors if a],
            "abstract":  abstract,
            "venue":     "arXiv",
            "year":      published[:4] if published else "",
            "url":       f"https://arxiv.org/abs/{arxiv_id}",
        })
    return papers


def fetch_crossref(keyword: str, since_date: str, rows: int = 100) -> list[dict]:
    """Fetch works indexed by CrossRef on/after `since_date` (YYYY-MM-DD)
    matching `keyword`, sorted by most recently indexed."""
    url = "https://api.crossref.org/works"
    params = {
        "query":  keyword,
        "filter": f"from-index-date:{since_date}",
        "rows":   rows,
        "sort":   "indexed",
        "order":  "desc",
    }
    try:
        resp = requests.get(url, params=params, timeout=30, headers={"User-Agent": CROSSREF_UA})
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        applog.log_api_network_error("paper_monitor.crossref", e)
        return []

    try:
        items = resp.json().get("message", {}).get("items", [])
    except ValueError as e:
        applog.log_error("paper_monitor.crossref_parse", str(e))
        return []

    papers = []
    for it in items:
        doi = (it.get("DOI") or "").lower()
        if not doi:
            continue
        title_list = it.get("title") or []
        title = _clean_ws(title_list[0]) if title_list else ""
        if not title:
            continue
        authors = [
            _clean_ws(f"{a.get('given', '')} {a.get('family', '')}".strip())
            for a in it.get("author") or []
        ]
        venue_list = it.get("container-title") or []
        venue = _clean_ws(venue_list[0]) if venue_list else ""
        year_parts = (it.get("issued") or {}).get("date-parts") or [[]]
        year = str(year_parts[0][0]) if year_parts and year_parts[0] else ""
        abstract = _strip_tags(it.get("abstract", "") or "")
        papers.append({
            "source":   "crossref",
            "id":       doi,
            "doi":      doi,
            "title":    title,
            "authors":  [a for a in authors if a],
            "abstract": abstract,
            "venue":    venue,
            "year":     year,
            "url":      f"https://doi.org/{doi}",
        })
    return papers


def fetch_openalex(keyword: str, since_date: str, per_page: int = 100) -> list[dict]:
    """Fetch OpenAlex works matching `keyword` published on/after `since_date`."""
    url = "https://api.openalex.org/works"
    params = {
        "search":   keyword,
        "filter":   f"from_publication_date:{since_date}",
        "sort":     "publication_date:desc",
        "per-page": per_page,
        "mailto":   "momiller@anl.gov",
    }
    try:
        resp = requests.get(url, params=params, timeout=30, headers={"User-Agent": CROSSREF_UA})
        resp.raise_for_status()
        items = resp.json().get("results", [])
    except (requests.exceptions.RequestException, ValueError) as e:
        applog.log_api_network_error("paper_monitor.openalex", e)
        return []

    papers = []
    for it in items:
        oa_url = it.get("id", "") or ""
        openalex_id = oa_url.rsplit("/", 1)[-1] if oa_url else ""
        doi = (it.get("doi") or "").replace("https://doi.org/", "").lower()
        title = _clean_ws(it.get("title") or "")
        if not title or not openalex_id:
            continue
        authors = [
            _clean_ws((a.get("author") or {}).get("display_name", ""))
            for a in it.get("authorships") or []
        ]
        primary_location = it.get("primary_location") or {}
        venue_src = primary_location.get("source") or {}
        venue = _clean_ws(venue_src.get("display_name") or "")
        year = str(it.get("publication_year") or "")
        abstract = _reconstruct_inverted(it.get("abstract_inverted_index"))
        papers.append({
            "source":   "openalex",
            "id":       doi or openalex_id,
            "doi":      doi,
            "title":    title,
            "authors":  [a for a in authors if a],
            "abstract": abstract,
            "venue":    venue,
            "year":     year,
            "url":      (f"https://doi.org/{doi}" if doi else oa_url),
        })
    return papers


def fetch_semantic_scholar(keyword: str, since_year: str, limit: int = 100) -> list[dict]:
    """Fetch Semantic Scholar search hits for `keyword`, filtered to since_year onwards."""
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query":  keyword,
        "limit":  limit,
        "year":   f"{since_year}-",
        "fields": "title,authors,abstract,externalIds,venue,year,publicationDate,url",
    }
    try:
        resp = requests.get(url, params=params, timeout=30, headers={"User-Agent": CROSSREF_UA})
        resp.raise_for_status()
        items = resp.json().get("data", [])
    except (requests.exceptions.RequestException, ValueError) as e:
        applog.log_api_network_error("paper_monitor.semantic_scholar", e)
        return []

    papers = []
    for it in items:
        paper_id = it.get("paperId") or ""
        ext = it.get("externalIds") or {}
        doi = (ext.get("DOI") or "").lower()
        arxiv_id = ext.get("ArXiv") or ""
        title = _clean_ws(it.get("title") or "")
        if not title or not paper_id:
            continue
        authors = [_clean_ws(a.get("name") or "") for a in it.get("authors") or []]
        venue = _clean_ws(it.get("venue") or "")
        year = str(it.get("year") or "")
        abstract = _clean_ws(it.get("abstract") or "")
        link = (
            it.get("url")
            or (f"https://doi.org/{doi}" if doi else "")
            or (f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "")
            or f"https://www.semanticscholar.org/paper/{paper_id}"
        )
        papers.append({
            "source":   "semantic_scholar",
            "id":       doi or arxiv_id or paper_id,
            "doi":      doi,
            "title":    title,
            "authors":  [a for a in authors if a],
            "abstract": abstract,
            "venue":    venue,
            "year":     year,
            "url":      link,
        })
    return papers


def _reconstruct_inverted(index) -> str:
    """OpenAlex stores abstracts as {word: [positions]}. Reconstruct into text."""
    if not index:
        return ""
    positions = []
    for word, idxs in index.items():
        for i in idxs:
            positions.append((i, word))
    positions.sort()
    return " ".join(w for _, w in positions)


def _clean_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _strip_tags(s: str) -> str:
    return _clean_ws(re.sub(r"<[^>]+>", "", s or ""))


# ── LLM screening ──────────────────────────────────────────────────────────────

_SCREEN_PROMPT_TMPL = (
    "You are triaging scientific literature for the beamline scientists at APS 8-ID (XPCS).\n"
    "Given the title and abstract below, do two things:\n"
    "  1. Rate 0-10 how relevant this paper is to X-ray Photon Correlation Spectroscopy.\n"
    "     10 = paper is explicitly about XPCS methodology or uses XPCS as a primary technique\n"
    "     6-8 = adjacent (dynamic light scattering, coherent diffraction, speckle statistics, "
    "colloidal nanoparticle dynamics)\n"
    "     below 6 = tangential mention or off-topic\n"
    "  2. Write a one-sentence plain-English summary of what the paper is about.\n\n"
    "TITLE: {title}\n\n"
    "ABSTRACT: {abstract}\n\n"
    "Respond with ONLY a raw JSON object, no markdown or code fences. Example:\n"
    '{{"score": 8, "reason": "Uses XPCS to study nanoparticle diffusion.", '
    '"summary": "The authors combine XPCS and rheology to probe the yielding of a colloidal gel."}}'
)


def screen(paper: dict) -> tuple[float, str, str]:
    """Ask the LLM to rate a paper's XPCS relevance and summarize it in one sentence.
    Returns (score, reason, summary). Fail-open with score=0 on any error."""
    prompt = _SCREEN_PROMPT_TMPL.format(
        title=paper.get("title", ""),
        abstract=(paper.get("abstract") or "")[:2000],
    )
    raw = call_argo_llm([{"role": "user", "content": prompt}])
    if not isinstance(raw, str):
        applog.log_error("paper_monitor.screen", f"non-string LLM response: {type(raw)}")
        return (0.0, "", "")

    text = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text.strip())
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        applog.log_error("paper_monitor.screen_parse", f"no JSON in: {text[:200]}")
        return (0.0, "", "")
    try:
        obj = json.loads(match.group(0))
        score = float(obj.get("score", 0))
        reason = str(obj.get("reason", "")).strip()
        summary = str(obj.get("summary", "")).strip()
        return (score, reason, summary)
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        applog.log_error("paper_monitor.screen_json", f"{e}: {text[:200]}")
        return (0.0, "", "")


# ── Email ──────────────────────────────────────────────────────────────────────

def _render_metadata_html(meta: dict) -> str:
    row = "<div style='margin:2px 0'><b>{label}:</b> {value}</div>"
    lines = "".join(row.format(label=escape(k), value=escape(str(v))) for k, v in meta.items())
    return (
        "<div style='margin-top:2em; padding-top:1em; border-top:2px solid #888; "
        "font-size:0.9em; color:#555; font-family:monospace'>"
        "<div style='font-weight:bold; margin-bottom:6px; color:#333'>About this digest</div>"
        f"{lines}"
        "</div>"
    )


def _render_metadata_plain(meta: dict) -> str:
    width = max(len(k) for k in meta.keys())
    lines = "\n".join(f"  {k.ljust(width)}  {v}" for k, v in meta.items())
    return "\n\n----------------------------------------\nAbout this digest\n\n" + lines


def _render_html(papers: list[dict], date_str: str, meta: dict) -> str:
    row = "<div style='margin:2px 0'><b>{label}:</b> {value}</div>"

    entries = []
    for i, p in enumerate(papers, 1):
        authors = ", ".join(p["authors"][:6]) + (" et al." if len(p["authors"]) > 6 else "")
        venue_year = escape(p.get("venue", "") or "?")
        if p.get("year"):
            venue_year += f", {escape(p['year'])}"
        abstract = escape(p.get("abstract", "")) or "<i>(no abstract available)</i>"
        link_html = f"<a href='{escape(p['url'])}'>{escape(p.get('id', p['url']))}</a>"

        entries.append(
            f"<div style='margin:0 0 1.5em 0; padding-bottom:1em; border-bottom:1px solid #ddd'>"
            f"<div style='font-size:1.05em; margin-bottom:6px'><b>#{i}</b></div>"
            + row.format(label="Title",     value=f"<b>{escape(p['title'])}</b>")
            + row.format(label="Authors",   value=escape(authors) or "<i>(unknown)</i>")
            + row.format(label="Venue",     value=venue_year)
            + row.format(label="Summary",   value=escape(p.get('summary') or '(no summary generated)'))
            + row.format(label="Relevance", value=f"<b>{p['score']:.1f}/10</b> &mdash; {escape(p['reason'] or '(no reason given)')}")
            + row.format(label="Link",      value=link_html)
            + f"<div style='margin-top:10px'><b>Abstract:</b></div>"
            + f"<div style='margin-top:4px'>{abstract}</div>"
            + "</div>"
        )
    return (
        f"<h2 style='margin-bottom:0.75em'>{len(papers)} new XPCS-relevant papers &mdash; {escape(date_str)}</h2>"
        + "".join(entries)
        + _render_metadata_html(meta)
    )


def send_digest(papers: list[dict], date_str: str, meta: dict) -> None:
    if not (GMAIL_ADDRESS and GMAIL_APP_PASSWORD):
        applog.log_error("paper_monitor.email", "GMAIL_ADDRESS/GMAIL_APP_PASSWORD not set; digest not sent")
        return

    msg = EmailMessage()
    msg["Subject"] = f"XPCS Paper Digest — {date_str} — {len(papers)} papers"
    msg["From"] = GMAIL_ADDRESS
    msg["To"] = ", ".join(TO_ADDRS)
    def _plain_entry(i, p):
        authors = ", ".join(p["authors"][:6]) + (" et al." if len(p["authors"]) > 6 else "")
        venue = (p.get("venue") or "?") + (f", {p['year']}" if p.get("year") else "")
        abstract = p.get("abstract", "") or "(no abstract available)"
        return (
            f"#{i}\n"
            f"Title:     {p['title']}\n"
            f"Authors:   {authors or '(unknown)'}\n"
            f"Venue:     {venue}\n"
            f"Summary:   {p.get('summary') or '(no summary generated)'}\n"
            f"Relevance: {p['score']:.1f}/10 — {p.get('reason') or '(no reason given)'}\n"
            f"Link:      {p['url']}\n"
            f"\nAbstract:\n{abstract}\n"
        )

    plain = "\n----------------------------------------\n\n".join(
        _plain_entry(i, p) for i, p in enumerate(papers, 1)
    )
    plain = (plain or "(no papers)") + _render_metadata_plain(meta)
    msg.set_content(plain)
    msg.add_alternative(_render_html(papers, date_str, meta), subtype="html")

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
        s.send_message(msg)


# ── Slack ──────────────────────────────────────────────────────────────────────

_SLACK_MAX_PAPERS = 20  # Slack messages cap at ~50 blocks; leave headroom


def _slack_paper_block(i: int, p: dict) -> dict:
    authors = ", ".join(p["authors"][:6]) + (" et al." if len(p["authors"]) > 6 else "")
    venue = (p.get("venue") or "?") + (f", {p['year']}" if p.get("year") else "")
    title_link = f"<{p['url']}|{_slack_escape(p['title'])}>"
    lines = [
        f"*#{i} — {p['score']:.1f}/10*  {title_link}",
        f"_{_slack_escape(authors) or 'unknown'}_ • _{_slack_escape(venue)}_",
        f"*Summary:* {_slack_escape(p.get('summary') or '(no summary generated)')}",
        f"*Why relevant:* {_slack_escape(p.get('reason') or '(no reason given)')}",
    ]
    return {"type": "section", "text": {"type": "mrkdwn", "text": "\n".join(lines)}}


def _slack_escape(s: str) -> str:
    """Slack mrkdwn: escape only the three chars that break parsing."""
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def send_slack(papers: list[dict], date_str: str, meta: dict) -> None:
    if not SLACK_WEBHOOK_URL:
        return

    shown = papers[:_SLACK_MAX_PAPERS]
    truncated = len(papers) - len(shown)

    blocks = [
        {"type": "header", "text": {"type": "plain_text",
                                     "text": f"{len(papers)} new XPCS-relevant papers — {date_str}"}},
    ]
    for i, p in enumerate(shown, 1):
        blocks.append(_slack_paper_block(i, p))
        blocks.append({"type": "divider"})

    if truncated > 0:
        blocks.append({"type": "section", "text": {"type": "mrkdwn",
                                                    "text": f"_…and {truncated} more (see email for full list)_"}})
        blocks.append({"type": "divider"})

    footer_text = (
        "*About this digest*\n"
        + "\n".join(f"*{_slack_escape(k)}:* {_slack_escape(str(v))}" for k, v in meta.items())
    )
    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": footer_text},
    })

    payload = {
        "text": f"{len(papers)} new XPCS-relevant papers — {date_str}",
        "blocks": blocks,
    }
    try:
        resp = requests.post(SLACK_WEBHOOK_URL, json=payload, timeout=15)
        if not resp.ok:
            applog.log_api_error("paper_monitor.slack", resp.status_code, resp.text)
    except requests.exceptions.RequestException as e:
        applog.log_api_network_error("paper_monitor.slack", e)


# ── Main tick ──────────────────────────────────────────────────────────────────

def run_once(dry_run: bool = False, limit: Optional[int] = None) -> list[dict]:
    """Run one monitor tick. Returns the list of scored papers included in the digest.
    limit caps the per-source fetch size (useful for quick smoke tests)."""
    try:
        started_at = datetime.now()
        state = _load_state()
        since_dt = (
            datetime.fromisoformat(state["last_run"]) if state["last_run"]
            else datetime.now(timezone.utc) - timedelta(days=1)
        )
        since_date = (since_dt - timedelta(days=1)).strftime("%Y-%m-%d")

        seen_dois     = set(state["seen_dois"])
        seen_arxiv    = set(state["seen_arxiv_ids"])
        seen_other    = set(state["seen_other_ids"])
        this_tick     = set()  # cross-source dedup within one run

        def _key(p):
            doi = (p.get("doi") or "").lower()
            if doi:
                return ("doi", doi)
            if p["source"] == "arxiv":
                return ("arxiv", p["id"])
            return ("other", f"{p['source']}:{p['id']}")

        def _is_seen(p):
            k = _key(p)
            if k in this_tick:
                return True
            return (
                (k[0] == "doi"   and k[1] in seen_dois) or
                (k[0] == "arxiv" and k[1] in seen_arxiv) or
                (k[0] == "other" and k[1] in seen_other)
            )

        def _mark_seen(p):
            k = _key(p)
            this_tick.add(k)
            if k[0] == "doi":     seen_dois.add(k[1])
            elif k[0] == "arxiv": seen_arxiv.add(k[1])
            else:                 seen_other.add(k[1])

        arxiv_n      = limit if limit is not None else 50
        crossref_n   = limit if limit is not None else 100
        openalex_n   = limit if limit is not None else 100
        semscholar_n = limit if limit is not None else 100

        candidates = []
        for kw in KEYWORDS:
            for p in fetch_arxiv(kw, max_results=arxiv_n):
                if not _is_seen(p):
                    candidates.append(p); _mark_seen(p)
            for p in fetch_crossref(kw, since_date, rows=crossref_n):
                if not _is_seen(p):
                    candidates.append(p); _mark_seen(p)
            for p in fetch_openalex(kw, since_date, per_page=openalex_n):
                if not _is_seen(p):
                    candidates.append(p); _mark_seen(p)
            for p in fetch_semantic_scholar(kw, since_year=since_date[:4], limit=semscholar_n):
                if not _is_seen(p):
                    candidates.append(p); _mark_seen(p)

        by_source = {}
        for p in candidates:
            by_source[p["source"]] = by_source.get(p["source"], 0) + 1
        print(
            f"[paper_monitor] {len(candidates)} new candidates from {len(KEYWORDS)} keyword(s) "
            f"({', '.join(f'{k}={v}' for k, v in by_source.items()) or 'none'})"
        )

        included = []
        for p in candidates:
            score, reason, summary = screen(p)
            print(f"  [{score:>4.1f}] {p['source']:>8} {p['id']} — {p['title'][:80]}")
            if score >= MIN_SCORE:
                p["score"], p["reason"], p["summary"] = score, reason, summary
                included.append(p)
        included.sort(key=lambda x: x["score"], reverse=True)

        date_str = started_at.strftime("%Y-%m-%d")
        print(f"[paper_monitor] {len(included)} above threshold ({MIN_SCORE})")

        is_background = threading.current_thread().name == "paper_monitor"
        next_run_dt   = _next_run_at(datetime.now(timezone.utc))
        schedule_str  = f"daily at {RUN_AT_HHMM} {RUN_TZ_NAME}"
        next_run_str  = (
            next_run_dt.strftime("%Y-%m-%d %H:%M %Z")
            if is_background
            else f"manual run — background thread runs {schedule_str} (next: {next_run_dt.strftime('%Y-%m-%d %H:%M %Z')})"
        )

        source_counts = ", ".join(f"{k}={v}" for k, v in sorted(by_source.items())) or "none"
        included_by_source = {}
        for p in included:
            included_by_source[p["source"]] = included_by_source.get(p["source"], 0) + 1
        included_counts = ", ".join(f"{k}={v}" for k, v in sorted(included_by_source.items())) or "none"

        meta = {
            "Sources scanned":       "arXiv, CrossRef, OpenAlex, Semantic Scholar",
            "Keywords":              ", ".join(KEYWORDS),
            "Date window":           f"arXiv: newest N submissions; CrossRef/OpenAlex: on/after {since_date}; Semantic Scholar: year {since_date[:4]}+",
            "Max papers per source": f"arXiv {arxiv_n}, CrossRef {crossref_n}, OpenAlex {openalex_n}, Semantic Scholar {semscholar_n}",
            "Candidates fetched":    f"{len(candidates)} ({source_counts})",
            "Above threshold":       f"{len(included)} (min score {MIN_SCORE}/10)",
            "Included per source":   included_counts,
            "LLM model":             LLM_CONFIG["model"],
            "Delivery channels":     ", ".join(
                ch for ch, on in [
                    ("email", bool(GMAIL_ADDRESS and GMAIL_APP_PASSWORD)),
                    ("slack", bool(SLACK_WEBHOOK_URL)),
                ] if on
            ) or "none configured",
            "Recipient":             ", ".join(TO_ADDRS),
            "Schedule":              schedule_str,
            "Ran at":                started_at.strftime("%Y-%m-%d %H:%M:%S %Z").strip() or started_at.strftime("%Y-%m-%d %H:%M:%S"),
            "Next run":              next_run_str,
        }

        if dry_run:
            print(_render_html(included, date_str, meta) if included else "(empty digest)")
            return included

        delivered = {"email": False, "slack": False}
        if included:
            if GMAIL_ADDRESS and GMAIL_APP_PASSWORD:
                send_digest(included, date_str, meta)
                delivered["email"] = True
            if SLACK_WEBHOOK_URL:
                send_slack(included, date_str, meta)
                delivered["slack"] = True

        _log_run(meta, included, len(candidates), delivered)

        state["seen_dois"]      = sorted(seen_dois)
        state["seen_arxiv_ids"] = sorted(seen_arxiv)
        state["seen_other_ids"] = sorted(seen_other)
        state["last_run"]       = datetime.now(timezone.utc).isoformat()
        _save_state(state)
        return included

    except Exception as e:
        applog.log_error("paper_monitor.run_once", str(e))
        print(f"[paper_monitor] tick failed: {e}")
        return []


# ── Background thread wiring ───────────────────────────────────────────────────

_started_lock = threading.Lock()
_started = False


def run_loop() -> None:
    print(f"[paper_monitor] loop started, daily at {RUN_AT_HHMM} {RUN_TZ_NAME}")
    while True:
        next_run = _next_run_at(datetime.now(timezone.utc))
        sleep_s  = (next_run - datetime.now(next_run.tzinfo)).total_seconds()
        print(f"[paper_monitor] sleeping {sleep_s/3600:.1f}h until {next_run.strftime('%Y-%m-%d %H:%M %Z')}")
        time.sleep(max(60.0, sleep_s))
        run_once()


def start_background() -> None:
    """Idempotent. Called from app.py at chainlit startup.
    Self-disables if GMAIL_ADDRESS/GMAIL_APP_PASSWORD are missing."""
    global _started
    with _started_lock:
        if _started:
            return
        if not (GMAIL_ADDRESS and GMAIL_APP_PASSWORD):
            print("[paper_monitor] disabled: GMAIL_ADDRESS/GMAIL_APP_PASSWORD not set in .env")
            return
        _started = True
        t = threading.Thread(target=run_loop, daemon=True, name="paper_monitor")
        t.start()


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="XPCS paper monitor")
    ap.add_argument("--once", action="store_true", help="run one tick and exit (default: loop forever)")
    ap.add_argument("--dry-run", action="store_true", help="print digest, do not send email or mutate state")
    ap.add_argument("--reset-state", action="store_true", help="delete the persistent state file and exit")
    ap.add_argument("--limit", type=int, default=None, help="cap fetch size per source (smoke test)")
    args = ap.parse_args()

    if args.reset_state:
        if STATE_FILE.exists():
            STATE_FILE.unlink()
            print(f"[paper_monitor] deleted {STATE_FILE}")
        else:
            print(f"[paper_monitor] no state file at {STATE_FILE}")
        sys.exit(0)

    if args.once or args.dry_run:
        run_once(dry_run=args.dry_run, limit=args.limit)
    else:
        run_loop()
