import re
import json
import time
import sys
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
    "https://photon-science.desy.de/facilities/petra_iii/beamlines/p10_coherence_applications/publications_from_p10/2026/index_eng.html",
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

    existing_dois = {p["doi"] for p in queue if p.get("status") in ("pending", "approved")}
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
from beamline publication pages and add them to a review queue.

XPCS relevant topics include: speckle, coherence, photon correlation, dynamics,
scattering, synchrotron, soft matter dynamics, nanoparticle dynamics, phase transitions,
diffusion, relaxation, coherent X-ray imaging, BCDI, SAXS dynamics, correlation functions.

For each source URL you are given:
1. Call scrape_beamline_page to get the list of papers
2. For each paper, use the title to make an initial judgment
   - If the title is clearly XPCS relevant, fetch the abstract to confirm
   - If the title is clearly unrelated (e.g. protein crystallography, optics engineering),
     skip it without fetching the abstract
   - If you are unsure, fetch the abstract to decide
3. After reading the abstract, decide if the paper is XPCS relevant
4. If relevant:
   a. Call fetch_pdf with the DOI and the paper title. The tool tries Unpaywall,
      Semantic Scholar, and arXiv in sequence. Always pass title so arXiv search works.
      This returns pdf_path and pdf_url (either a path/URL or null if unavailable).
   b. Call add_to_review_queue, passing pdf_path and pdf_url from the fetch_pdf result.
5. If not relevant, move on to the next paper

Be selective but thorough. When you are done with all sources,
give a brief summary of what you found."""

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

