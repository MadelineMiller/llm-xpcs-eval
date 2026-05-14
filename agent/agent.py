import re
import json
import time
import sys
import tempfile
import requests
import importlib.util
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# forge_graph (als_knowledge_agent) — literature search and paper reading tools
_FORGE_GRAPH_SRC = Path("/home/beams/MOMILLER/Desktop/forge_graph/als_knowledge_agent/src")
if str(_FORGE_GRAPH_SRC) not in sys.path:
    sys.path.insert(0, str(_FORGE_GRAPH_SRC))

try:
    from als_knowledge_agent.mcp_servers.literature_server import (
        search_papers_by_topic as _fg_search_papers_by_topic,
        search_papers_by_doi as _fg_search_papers_by_doi,
        find_citing_papers as _fg_find_citing_papers,
        find_related_papers_semantic_scholar as _fg_find_related_papers,
    )
    from als_knowledge_agent.mcp_servers.paper_reader_server import (
        read_paper_by_doi as _fg_read_paper_by_doi,
        read_paper_section as _fg_read_paper_section,
        read_paper_from_path as _fg_read_paper_from_path,
        extract_experimental_details as _fg_extract_experimental_details,
    )
    _FORGE_GRAPH_AVAILABLE = True
    print("[forge_graph] literature + paper reader tools loaded")
except ImportError as e:
    _FORGE_GRAPH_AVAILABLE = False
    print(f"[forge_graph] not available: {e}")

# ============================================================================
# CONFIGURATION
# ============================================================================

load_dotenv(Path(__file__).parent.parent / ".env")

config_path = Path(__file__).parent.parent / "config.py"
spec = importlib.util.spec_from_file_location("project_config", config_path)
project_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(project_config)
LLM_CONFIG = project_config.LLM_CONFIG

ARGO_API_URL = os.getenv("ARGO_API_URL")
ARGO_USER    = os.getenv("ARGO_USER")

REVIEW_QUEUE_FILE = Path(__file__).parent / "review_queue.json"

BEAMLINE_SOURCES = [
    "https://photon-science.desy.de/facilities/petra_iii/beamlines/p10_coherence_applications/publications_from_p10/2025/index_eng.html",
    "https://www.esrf.fr/files/live/sites/www/files/UsersAndScience/Experiments/SoftMatter/ID10/ID10EH2/Science/ID10EH2_publications2023.pdf",
]

# ============================================================================
# TOOL IMPLEMENTATIONS
# ============================================================================

def scrape_beamline_page(url: str) -> list:
    """Scrape a beamline publication page and return a list of papers."""
    print(f"\n[TOOL EXECUTING] scrape_beamline_page({url})")

    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )
    driver.get(url)
    time.sleep(5)
    page_text = driver.find_element("tag name", "body").text
    driver.quit()

    anchor_pos = page_text.find("Home / Facilities")
    if anchor_pos == -1:
        return []
    pub_text = page_text[anchor_pos:]

    lines_all = pub_text.split("\n")
    pub_lines = [
        l.strip() for l in lines_all
        if l.strip()
        and not l.strip().startswith("Home /")
        and l.strip() not in ("2026", "·")
    ]
    pub_text_clean = "\n".join(pub_lines)

    noise = ["files", "bibtex", "ris", "endnote", "data privacy", "cookies", "imprint"]
    doi_pattern = re.compile(r'\[10\.\d{4,}/[^\]]+\]')
    parts = doi_pattern.split(pub_text_clean)
    doi_matches = doi_pattern.findall(pub_text_clean)

    papers = []
    for i, doi_raw in enumerate(doi_matches):
        doi = doi_raw.strip("[]")
        chunk = parts[i]
        lines = [l.strip() for l in chunk.split("\n") if l.strip()]
        lines = [l for l in lines if not any(n in l.lower() for n in noise)]

        if len(lines) < 3:
            continue

        journal     = lines[-1]
        title       = lines[-2]
        authors_raw = lines[-3]
        authors     = [a.strip() for a in authors_raw.split(";") if a.strip()]

        papers.append({
            "doi":        doi,
            "title":      title,
            "authors":    authors,
            "journal":    journal,
            "source_url": url,
        })

    print(f"  Scraped {len(papers)} papers")
    return papers


def fetch_abstract(doi: str) -> str:
    """Fetch abstract for a DOI from Crossref, falling back to Semantic Scholar."""
    print(f"\n[TOOL EXECUTING] fetch_abstract({doi})")

    headers = {"User-Agent": "XPCS-Harvester-Bot/1.0 (mailto:momiller@anl.gov)"}

    try:
        response = requests.get(
            f"https://api.crossref.org/works/{doi}",
            headers=headers,
            timeout=15
        )
        if response.status_code == 200:
            abstract = response.json()["message"].get("abstract", "")
            if abstract:
                abstract = re.sub(r'<[^>]+>', '', abstract)
                abstract = re.sub(r'\s+', ' ', abstract).strip()
                print(f"  Got abstract from Crossref ({len(abstract)} chars)")
                return abstract
    except Exception as e:
        print(f"  Crossref error: {e}")

    try:
        response = requests.get(
            f"https://api.semanticscholar.org/graph/v1/paper/{doi}",
            headers=headers,
            params={"fields": "abstract"},
            timeout=15
        )
        if response.status_code == 200:
            abstract = response.json().get("abstract", "")
            if abstract:
                print(f"  Got abstract from Semantic Scholar ({len(abstract)} chars)")
                return abstract
    except Exception as e:
        print(f"  Semantic Scholar error: {e}")

    print("  No abstract found")
    return ""


PDF_DIR = Path(__file__).parent / "pdfs"


_BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "application/pdf,text/html,application/xhtml+xml,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}


def _download_pdf(pdf_url: str, doi: str, headers: dict) -> dict:
    """Download a PDF from a URL and save it to PDF_DIR. Returns result dict."""
    PDF_DIR.mkdir(exist_ok=True)
    safe_name = re.sub(r"[^a-zA-Z0-9]", "_", doi) + ".pdf"
    pdf_path = PDF_DIR / safe_name
    dl = requests.get(pdf_url, headers=_BROWSER_HEADERS, timeout=60, stream=True, allow_redirects=True)
    dl.raise_for_status()
    content_type = dl.headers.get("Content-Type", "")
    if "pdf" not in content_type and "octet-stream" not in content_type:
        raise ValueError(f"Response is not a PDF (Content-Type: {content_type})")
    with open(pdf_path, "wb") as f:
        for chunk in dl.iter_content(chunk_size=8192):
            f.write(chunk)
    if pdf_path.stat().st_size < 4096:
        pdf_path.unlink()
        raise ValueError("Downloaded file is too small to be a valid PDF")
    print(f"  Downloaded PDF to {pdf_path} ({pdf_path.stat().st_size // 1024} KB)")
    return {"pdf_path": str(pdf_path), "pdf_url": pdf_url}


def _selenium_pdf_download(doi: str) -> dict:
    """Navigate to a DOI page with headless Chrome, find a PDF link, and download it.
    Uses ANL institutional IP for publisher access via session cookies."""
    print(f"  Trying Selenium browser fallback for {doi}")
    PDF_DIR.mkdir(exist_ok=True)
    safe_name = re.sub(r"[^a-zA-Z0-9]", "_", doi) + ".pdf"
    pdf_path = PDF_DIR / safe_name

    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")

    driver = None
    try:
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=options,
        )
        driver.get(f"https://doi.org/{doi}")
        time.sleep(5)

        pdf_url = None
        for selector in ["a[href*='.pdf']", "a[href*='pdf']", "a[data-title='PDF']", "a[title*='PDF']"]:
            try:
                for el in driver.find_elements("css selector", selector):
                    href = el.get_attribute("href") or ""
                    if href and "pdf" in href.lower():
                        pdf_url = href
                        break
            except Exception:
                pass
            if pdf_url:
                break

        if not pdf_url:
            print("  Selenium: no PDF link found on publisher page")
            return {"pdf_path": None, "pdf_url": None}

        print(f"  Selenium found PDF link: {pdf_url}")

        session = requests.Session()
        for cookie in driver.get_cookies():
            session.cookies.set(cookie["name"], cookie["value"])

        ua = driver.execute_script("return navigator.userAgent;")
        resp = session.get(
            pdf_url,
            headers={"User-Agent": ua, "Referer": driver.current_url, "Accept": "application/pdf,*/*"},
            timeout=60,
            stream=True,
            allow_redirects=True,
        )
        resp.raise_for_status()

        ct = resp.headers.get("Content-Type", "")
        if "pdf" not in ct and "octet-stream" not in ct:
            raise ValueError(f"Not a PDF: {ct}")

        with open(pdf_path, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)

        if pdf_path.stat().st_size < 4096:
            pdf_path.unlink()
            raise ValueError("File too small to be a valid PDF")

        print(f"  Selenium downloaded: {pdf_path} ({pdf_path.stat().st_size // 1024} KB)")
        return {"pdf_path": str(pdf_path), "pdf_url": pdf_url}

    except Exception as e:
        print(f"  Selenium fallback error: {e}")
        if pdf_path.exists():
            pdf_path.unlink()
        return {"pdf_path": None, "pdf_url": None}
    finally:
        if driver:
            driver.quit()


def fetch_pdf(doi: str, title: str = "") -> dict:
    """Try multiple sources to find and download an open-access PDF."""
    print(f"\n[TOOL EXECUTING] fetch_pdf({doi})")
    headers = {"User-Agent": "XPCS-Harvester-Bot/1.0 (mailto:momiller@anl.gov)"}

    # 1. Unpaywall — best source for verified open-access locations
    try:
        resp = requests.get(
            f"https://api.unpaywall.org/v2/{doi}",
            params={"email": "momiller@anl.gov"},
            headers=headers,
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            # Check every OA location, not just the best one
            locations = [data.get("best_oa_location")] + data.get("oa_locations", [])
            for loc in locations:
                if not loc:
                    continue
                pdf_url = loc.get("url_for_pdf")
                if pdf_url:
                    print(f"  Unpaywall found PDF: {pdf_url}")
                    return _download_pdf(pdf_url, doi, headers)
            print("  Unpaywall: no pdf URL in any OA location")
    except Exception as e:
        print(f"  Unpaywall error: {e}")

    # 2. Semantic Scholar — openAccessPdf field
    try:
        resp = requests.get(
            f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}",
            headers=headers,
            params={"fields": "openAccessPdf,externalIds"},
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            oa = data.get("openAccessPdf") or {}
            pdf_url = oa.get("url")
            if pdf_url:
                print(f"  Semantic Scholar found PDF: {pdf_url}")
                return _download_pdf(pdf_url, doi, headers)
            # Also check if it's on arXiv via externalIds
            arxiv_id = (data.get("externalIds") or {}).get("ArXiv")
            if arxiv_id:
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
                print(f"  Semantic Scholar found arXiv ID {arxiv_id}, trying PDF")
                return _download_pdf(pdf_url, doi, headers)
            print("  Semantic Scholar: no open-access PDF")
    except Exception as e:
        print(f"  Semantic Scholar error: {e}")

    # 3. arXiv title search — effective for physics/materials preprints
    if title:
        try:
            import urllib.parse
            query = urllib.parse.quote(f'ti:"{title}"')
            resp = requests.get(
                f"http://export.arxiv.org/api/query?search_query={query}&max_results=3",
                headers=headers,
                timeout=15,
            )
            if resp.status_code == 200:
                # Parse Atom XML — look for PDF link in each entry
                xml = resp.text
                entries = xml.split("<entry>")[1:]
                for entry in entries:
                    # Find PDF link: <link title="pdf" href="..."/>
                    pdf_match = re.search(r'<link[^>]+title="pdf"[^>]+href="([^"]+)"', entry)
                    if not pdf_match:
                        # Alternative format
                        pdf_match = re.search(r'href="(https://arxiv\.org/pdf/[^"]+)"', entry)
                    if pdf_match:
                        pdf_url = pdf_match.group(1)
                        if not pdf_url.endswith(".pdf"):
                            pdf_url += ".pdf"
                        print(f"  arXiv found PDF: {pdf_url}")
                        return _download_pdf(pdf_url, doi, headers)
            print("  arXiv: no matching paper found")
        except Exception as e:
            print(f"  arXiv error: {e}")

    # 4. Selenium browser fallback — uses ANL institutional IP for publisher sites
    result = _selenium_pdf_download(doi)
    if result["pdf_path"]:
        return result

    print("  No open-access PDF found via any source")
    return {"pdf_path": None, "pdf_url": None}


def add_to_review_queue(doi: str, title: str, authors: list,
                        journal: str, abstract: str, source_url: str,
                        relevant: bool, confidence: str, reason: str,
                        pdf_path: str = None, pdf_url: str = None) -> str:
    """Write a paper to the human review queue."""
    print(f"\n[TOOL EXECUTING] add_to_review_queue('{title[:60]}...')")

    if not relevant:
        msg = f"Paper '{title}' marked as not relevant, skipped."
        print(f"  {msg}")
        return msg

    queue = []
    if REVIEW_QUEUE_FILE.exists():
        with open(REVIEW_QUEUE_FILE) as f:
            queue = json.load(f)

    existing_dois = {p["doi"] for p in queue if p.get("status") in ("pending", "approved", "rejected")}
    if doi in existing_dois:
        return f"Paper '{title}' already in queue (pending/approved), skipped."

    entry = {
        "doi":        doi,
        "title":      title,
        "authors":    authors,
        "journal":    journal,
        "abstract":   abstract,
        "source_url": source_url,
        "pdf_path":   pdf_path,
        "pdf_url":    pdf_url,
        "agent_decision": {
            "relevant":   relevant,
            "confidence": confidence,
            "reason":     reason,
        },
        "status":    "pending",
        "queued_at": datetime.utcnow().isoformat(),
    }

    queue.append(entry)
    with open(REVIEW_QUEUE_FILE, "w") as f:
        json.dump(queue, f, indent=2)

    msg = f"Added '{title}' to review queue. Queue size: {len(queue)}"
    print(f"  {msg}")
    return msg


def _pdf_text(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes using PyMuPDF."""
    import fitz
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(pdf_bytes)
        tmp = f.name
    try:
        doc = fitz.open(tmp)
        pages = [page.get_text() for page in doc]
        doc.close()
        text = "\n".join(pages)
        print(f"  Extracted {len(text)} chars from {len(pages)} pages")
        return text
    finally:
        os.unlink(tmp)


def _selenium_fetch_pdf(url: str) -> bytes | None:
    """Fetch a PDF from within a real browser context to bypass bot detection.

    Navigates to the site root first (picks up session cookies), then uses
    the browser's own fetch() API so the request carries cookies and a real
    browser fingerprint. Works for sites that block plain Python requests.
    """
    import base64
    from urllib.parse import urlparse
    origin = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
    driver = None
    try:
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument(
            "--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36"
        )
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        driver.set_script_timeout(60)

        # Visit the site root to pick up session/consent cookies
        driver.get(origin)
        time.sleep(3)

        # Use the browser's fetch() to get the PDF — includes all cookies automatically
        b64 = driver.execute_async_script(f"""
            var done = arguments[arguments.length - 1];
            fetch({json.dumps(url)}, {{credentials: 'include'}})
                .then(function(r) {{
                    if (!r.ok) {{ done(null); return; }}
                    return r.arrayBuffer();
                }})
                .then(function(buf) {{
                    if (!buf) {{ done(null); return; }}
                    var bytes = new Uint8Array(buf);
                    var chunks = [];
                    for (var i = 0; i < bytes.length; i += 8192) {{
                        chunks.push(String.fromCharCode.apply(null, bytes.subarray(i, Math.min(i+8192, bytes.length))));
                    }}
                    done(btoa(chunks.join('')));
                }})
                .catch(function() {{ done(null); }});
        """)

        if b64:
            data = base64.b64decode(b64)
            print(f"  Browser fetch succeeded: {len(data) // 1024} KB")
            return data

        print("  Browser fetch returned null (server still rejected the request)")
        return None
    except Exception as e:
        print(f"  Selenium fetch error: {e}")
        return None
    finally:
        if driver:
            driver.quit()


def _parse_bibliography(pdf_bytes: bytes, source_url: str) -> list:
    """Extract papers from a bibliography PDF.

    Uses two strategies:
    1. PDF hyperlinks (get_links) — captures DOIs stored as clickable annotations,
       which is common in modern publication lists.
    2. Plain-text regex — catches DOIs written as visible text.
    Both are merged, de-duplicated, and enriched with surrounding text context.
    """
    import fitz
    doi_re = re.compile(r'(10\.\d{4,}/[^\s,;)\]\"\']+)')
    seen = set()
    doi_positions = {}   # doi -> (page_index, y position) for context lookup

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(pdf_bytes)
        tmp = f.name

    try:
        doc = fitz.open(tmp)
        page_texts = []

        for page_idx, page in enumerate(doc):
            text = page.get_text()
            page_texts.append(text)

            # Strategy 1: hyperlinks (DOIs stored as PDF annotations)
            for link in page.get_links():
                href = link.get("uri", "") or ""
                m = doi_re.search(href)
                if m:
                    doi = m.group(1).rstrip('.')
                    if doi not in seen:
                        seen.add(doi)
                        doi_positions[doi] = (page_idx, link.get("from", fitz.Rect()).y0)

            # Strategy 2: visible text regex
            for m in doi_re.finditer(text):
                doi = m.group(1).rstrip('.')
                if doi not in seen:
                    seen.add(doi)
                    doi_positions[doi] = (page_idx, -1)

        doc.close()
    finally:
        os.unlink(tmp)

    # Build paper entries: find context (title/authors) from surrounding text
    papers = []
    for doi, (page_idx, _) in doi_positions.items():
        page_text = page_texts[page_idx]
        lines = [l.strip() for l in page_text.split('\n') if l.strip()]

        # Find which line contains this DOI and grab surrounding context
        ctx_lines = []
        for i, line in enumerate(lines):
            if doi in line or doi.replace('/', '%2F') in line:
                ctx_lines = lines[max(0, i - 6):i + 1]
                break
        if not ctx_lines:
            ctx_lines = lines[:6]  # fallback to page start

        context = ' | '.join(ctx_lines)
        title_guess = max(ctx_lines[:-1], key=len) if len(ctx_lines) > 1 else ""

        papers.append({
            "doi":        doi,
            "title":      title_guess,
            "context":    context[:400],
            "source_url": source_url,
        })

    print(f"  Extracted {len(papers)} papers (hyperlinks + text) from bibliography")
    return papers


def scrape_publication_pdf(url: str) -> str:
    """Download a publication-list PDF and return a JSON list of papers with DOIs.

    Works like scrape_beamline_page but for PDF sources. Returns a JSON array
    where each entry has doi, title (guessed from context), context, source_url.
    """
    import fitz  # noqa: ensure import available
    print(f"\n[TOOL EXECUTING] scrape_publication_pdf({url})")

    # Try plain requests first (fast, works for most open URLs).
    from urllib.parse import urlparse
    origin = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
    session = requests.Session()
    session.headers.update({**_BROWSER_HEADERS, "Referer": origin + "/", "Accept": "application/pdf,*/*"})
    try:
        session.get(origin, timeout=10)
    except Exception:
        pass
    resp = session.get(url, timeout=30)

    if resp.ok:
        pdf_bytes = resp.content
    else:
        print(f"  HTTP {resp.status_code} — falling back to browser fetch...")
        pdf_bytes = _selenium_fetch_pdf(url)
        if not pdf_bytes:
            return json.dumps({"error": f"Could not download PDF (HTTP {resp.status_code}, browser fetch also failed)"})

    papers = _parse_bibliography(pdf_bytes, source_url=url)
    if not papers:
        # Fallback: no DOIs found — return truncated raw text so Claude can still try
        print("  No DOIs extracted; returning raw text as fallback")
        return _pdf_text(pdf_bytes)[:30000]
    return json.dumps(papers)


# --- forge_graph wrappers (relevance judgment only — no web discovery) ---

def lookup_papers_by_doi(dois: str) -> str:
    """Fetch abstract + concepts from OpenAlex for one or more DOIs (comma-separated)."""
    print(f"\n[TOOL EXECUTING] lookup_papers_by_doi({dois!r})")
    if not _FORGE_GRAPH_AVAILABLE:
        return json.dumps({"error": "forge_graph not available"})
    return _fg_search_papers_by_doi(dois)


def read_paper_content(doi: str) -> str:
    """Download and parse a paper by DOI into structured markdown with section list."""
    print(f"\n[TOOL EXECUTING] read_paper_content({doi!r})")
    if not _FORGE_GRAPH_AVAILABLE:
        return json.dumps({"error": "forge_graph not available"})
    return _fg_read_paper_by_doi(doi)


def read_paper_section(doi: str, section_name: str) -> str:
    """Extract one section from a paper already fetched by read_paper_content."""
    print(f"\n[TOOL EXECUTING] read_paper_section({doi!r}, {section_name!r})")
    if not _FORGE_GRAPH_AVAILABLE:
        return json.dumps({"error": "forge_graph not available"})
    return _fg_read_paper_section(doi, section_name)


def extract_experimental_details(doi: str) -> str:
    """Extract technique/material/instrument keywords from a paper's experimental section."""
    print(f"\n[TOOL EXECUTING] extract_experimental_details({doi!r})")
    if not _FORGE_GRAPH_AVAILABLE:
        return json.dumps({"error": "forge_graph not available"})
    return _fg_extract_experimental_details(doi)


# ============================================================================
# TOOL DEFINITIONS (sent to Claude)
# ============================================================================

TOOLS = [
    {
        "name": "fetch_pdf",
        "description": (
            "Try to find and download an open-access PDF for a paper. "
            "Searches Unpaywall, Semantic Scholar, and arXiv in order. "
            "Returns pdf_path (local file path) and pdf_url (source URL), "
            "or null for both if no open-access PDF is available. "
            "Always pass the paper title so arXiv search can be used as a fallback. "
            "Call this before add_to_review_queue for relevant papers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "doi": {
                    "type": "string",
                    "description": "The DOI of the paper"
                },
                "title": {
                    "type": "string",
                    "description": "The paper title, used to search arXiv as a fallback"
                }
            },
            "required": ["doi"]
        }
    },
    {
        "name": "scrape_beamline_page",
        "description": (
            "Scrape a beamline publication page and return a list of papers. "
            "Each paper has: doi, title, authors, journal, source_url."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL of the beamline publications page to scrape"
                }
            },
            "required": ["url"]
        }
    },
    {
        "name": "fetch_abstract",
        "description": (
            "Fetch the abstract of a paper given its DOI. "
            "Tries Crossref first, then Semantic Scholar. "
            "Returns the abstract text, or empty string if not found."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "doi": {
                    "type": "string",
                    "description": "The DOI of the paper"
                }
            },
            "required": ["doi"]
        }
    },
    {
        "name": "scrape_publication_pdf",
        "description": (
            "Download a publication-list PDF (e.g. from ESRF) and return a JSON list of papers. "
            "Use this for any source URL ending in .pdf. "
            "Each paper has: doi, title (guessed from context), context (surrounding text), source_url. "
            "Process the returned list the same way as scrape_beamline_page output."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL of the publication list PDF"}
            },
            "required": ["url"]
        }
    },
    {
        "name": "lookup_papers_by_doi",
        "description": (
            "Fetch abstract, concepts, and citation count from OpenAlex for one or more DOIs. "
            "Pass a comma-separated list of DOIs. "
            "Use this instead of fetch_abstract when you need richer metadata, "
            "or when fetch_abstract returns nothing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dois": {
                    "type": "string",
                    "description": "Comma-separated DOIs, e.g. '10.1000/xyz,10.1000/abc'"
                }
            },
            "required": ["dois"]
        }
    },
    {
        "name": "read_paper_content",
        "description": (
            "Download a paper by DOI and parse it into structured markdown. "
            "Returns a section list and a content preview (~8000 chars). "
            "Use only for papers where the abstract is insufficient to judge XPCS relevance. "
            "This downloads the PDF via open-access routes (Unpaywall, Semantic Scholar)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "doi": {"type": "string", "description": "DOI of the paper"}
            },
            "required": ["doi"]
        }
    },
    {
        "name": "read_paper_section",
        "description": (
            "Extract a specific section (e.g. 'experimental', 'methods') from a paper "
            "previously fetched by read_paper_content. "
            "Use when the content preview from read_paper_content is not enough."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "doi":          {"type": "string", "description": "DOI of the paper"},
                "section_name": {"type": "string", "description": "Section name substring, e.g. 'experimental' or 'methods'"}
            },
            "required": ["doi", "section_name"]
        }
    },
    {
        "name": "extract_experimental_details",
        "description": (
            "Extract detected techniques, materials, and instrument keywords from a paper's "
            "experimental section. Requires read_paper_content to have been called first. "
            "Useful for borderline papers to check for XPCS-relevant technique keywords."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "doi": {"type": "string", "description": "DOI of the paper"}
            },
            "required": ["doi"]
        }
    },
    {
        "name": "add_to_review_queue",
        "description": (
            "Add a paper to the human review queue after determining it is "
            "relevant to XPCS research. Include your reasoning."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "doi":        {"type": "string"},
                "title":      {"type": "string"},
                "authors":    {"type": "array", "items": {"type": "string"}},
                "journal":    {"type": "string"},
                "abstract":   {"type": "string"},
                "source_url": {"type": "string"},
                "relevant":   {"type": "boolean"},
                "confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"]
                },
                "reason": {
                    "type": "string",
                    "description": "One sentence explaining why this paper is XPCS relevant"
                },
                "pdf_path": {
                    "type": ["string", "null"],
                    "description": "Local file path from fetch_pdf, or null"
                },
                "pdf_url": {
                    "type": ["string", "null"],
                    "description": "Source URL from fetch_pdf, or null"
                }
            },
            "required": [
                "doi", "title", "authors", "journal", "abstract",
                "source_url", "relevant", "confidence", "reason"
            ]
        }
    }
]

# ============================================================================
# TOOL DISPATCHER
# ============================================================================

def dispatch_tool(tool_name: str, tool_input: dict) -> str:
    """Execute whichever tool Claude decided to call and return the result."""
    if tool_name == "scrape_beamline_page":
        papers = scrape_beamline_page(tool_input["url"])
        return json.dumps(papers)

    elif tool_name == "fetch_abstract":
        abstract = fetch_abstract(tool_input["doi"])
        return abstract if abstract else "No abstract available."

    elif tool_name == "fetch_pdf":
        result = fetch_pdf(tool_input["doi"], tool_input.get("title", ""))
        return json.dumps(result)

    elif tool_name == "scrape_publication_pdf":
        return scrape_publication_pdf(tool_input["url"])

    elif tool_name == "lookup_papers_by_doi":
        return lookup_papers_by_doi(tool_input["dois"])

    elif tool_name == "read_paper_content":
        return read_paper_content(tool_input["doi"])

    elif tool_name == "read_paper_section":
        return read_paper_section(tool_input["doi"], tool_input["section_name"])

    elif tool_name == "extract_experimental_details":
        return extract_experimental_details(tool_input["doi"])

    elif tool_name == "add_to_review_queue":
        return add_to_review_queue(
            doi        = tool_input["doi"],
            title      = tool_input["title"],
            authors    = tool_input.get("authors", []),
            journal    = tool_input.get("journal", ""),
            abstract   = tool_input.get("abstract", ""),
            source_url = tool_input.get("source_url", ""),
            relevant   = tool_input.get("relevant", True),
            confidence = tool_input.get("confidence", "medium"),
            reason     = tool_input.get("reason", ""),
            pdf_path   = tool_input.get("pdf_path"),
            pdf_url    = tool_input.get("pdf_url"),
        )

    else:
        return f"Unknown tool: {tool_name}"

# ============================================================================
# ARGO API CALL WITH TOOL SUPPORT
# ============================================================================

def call_argo_with_tools(messages: list) -> dict:
    """Call Argo API with tool definitions and return the full response."""
    payload = {
        "user":        ARGO_USER,
        "model":       LLM_CONFIG["model"],
        "messages":    messages,
        "tools":       TOOLS,
        "temperature": LLM_CONFIG["temperature"],
        "top_p":       LLM_CONFIG["top_p"],
        "max_tokens":  4096,
    }

    response = requests.post(ARGO_API_URL, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()

# ============================================================================
# AGENT LOOP — CLAUDE DRIVES THIS
# ============================================================================

def run_agent():
    print("=" * 60)
    print("XPCS LITERATURE HARVESTING AGENT")
    print("Claude is driving this loop via tool use")
    print("=" * 60)

    # Give Claude its goal and the list of sources to work through
    sources_list = "\n".join(f"- {url}" for url in BEAMLINE_SOURCES)

    system_prompt = """You are an autonomous XPCS literature harvesting agent.

Your goal is to find papers relevant to X-ray Photon Correlation Spectroscopy (XPCS)
from trusted beamline publication sources and add them to a human review queue.

XPCS-relevant topics include: X-ray photon correlation spectroscopy, speckle,
coherent X-rays, intensity fluctuations, correlation functions, dynamic light scattering
via X-ray, soft matter dynamics, nanoparticle dynamics, colloidal dynamics, phase
transitions, diffusion, relaxation times, SAXS dynamics, coherent diffraction,
heterogeneous dynamics, aging, jamming, gelation.

## Source types

You will receive two kinds of source URLs:

**HTML pages** (URL does not end in .pdf):
  Call scrape_beamline_page(url). This returns a structured list of papers with
  doi, title, authors, journal, source_url.

**PDF publication lists** (URL ends in .pdf):
  Call scrape_publication_pdf(url). This returns a JSON list of papers identical in
  structure to scrape_beamline_page output — each has doi, title, context, source_url.
  Process it the same way: evaluate each paper for XPCS relevance.

## Per-paper workflow

For each paper discovered from any source:

1. **Title check** — make an initial judgment from the title alone.
   - Clearly XPCS-related (speckle, photon correlation, coherent dynamics): proceed to step 2.
   - Clearly unrelated (protein crystallography, optics fabrication, detector engineering
     with no dynamics component): skip — do not fetch abstract.
   - Uncertain: proceed to step 2.

2. **Abstract check** — call lookup_papers_by_doi(doi). This returns the abstract,
   concepts, and citation count from OpenAlex. Decide if the paper is XPCS-relevant.
   Fall back to fetch_abstract(doi) if lookup_papers_by_doi returns no abstract.

3. **Full-text check (borderline papers only)** — if the abstract is ambiguous,
   call read_paper_content(doi) to download and parse the paper. Check the section
   list returned, then call read_paper_section(doi, "experimental") or
   extract_experimental_details(doi) to look for XPCS technique keywords in the methods.
   Use this sparingly — only when abstract-level judgment is genuinely uncertain.

4. **If relevant**:
   a. Call fetch_pdf(doi, title) to attempt open-access PDF download.
   b. Call add_to_review_queue with all metadata and the pdf_path/pdf_url from fetch_pdf.

5. **If not relevant**: move on.

## Notes
- Be selective. Not every paper at an XPCS beamline uses XPCS — some use SAXS, WAXS,
  CDI, or other techniques. Judge by whether the paper actually measures dynamics via
  photon correlation, not just whether it used the beamline.
- When done with all sources, give a concise summary: how many papers found per source,
  how many queued."""

    messages = [
        {
            "role": "user",
            "content": f"Please harvest XPCS relevant papers from these beamline sources:\n{sources_list}"
        }
    ]

    print(f"\nGoal given to Claude: harvest papers from {len(BEAMLINE_SOURCES)} source(s)")
    print("Starting agent loop...\n")

    step = 0
    max_steps = 50  # safety limit

    while step < max_steps:
        step += 1
        print(f"\n{'='*60}")
        print(f"AGENT STEP {step}")
        print(f"{'='*60}")

        # Ask Claude what to do next
        result = call_argo_with_tools(messages)
        response_content = result.get("response", {})

        text_content  = response_content.get("content", "")
        tool_calls    = response_content.get("tool_calls", [])

        if text_content:
            print(f"\nClaude: {text_content}")

        # If no tool calls, Claude is done
        if not tool_calls:
            print("\n Claude has finished. No more tool calls.")
            break

        # Add Claude's response to message history
        messages.append({
            "role": "assistant",
            "content": text_content or "",
            "tool_calls": tool_calls
        })

        # Execute each tool Claude asked for
        tool_results = []
        for tool_call in tool_calls:
            tool_name  = tool_call["name"]
            tool_input = tool_call["input"]
            tool_id    = tool_call.get("id", "")

            print(f"\nClaude decided to call: {tool_name}")
            print(f"With input: {json.dumps(tool_input, indent=2)}")

            # Execute the tool
            result_str = dispatch_tool(tool_name, tool_input)

            print(f"Tool returned: {result_str[:200]}{'...' if len(result_str) > 200 else ''}")

            tool_results.append({
                "tool_call_id": tool_id,
                "role":         "tool",
                "name":         tool_name,
                "content":      result_str,
            })

        # Send tool results back to Claude so it can decide what to do next
        messages.append({
            "role": "user",
            "content": json.dumps(tool_results)
        })

    if step >= max_steps:
        print(f"\nReached max steps ({max_steps}). Stopping.")

    print("\n" + "=" * 60)
    print("AGENT RUN COMPLETE")
    print(f"Review queue: {REVIEW_QUEUE_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    run_agent()

