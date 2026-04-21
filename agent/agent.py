import re
import json
import time
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


def add_to_review_queue(doi: str, title: str, authors: list,
                        journal: str, abstract: str, source_url: str,
                        relevant: bool, confidence: str, reason: str) -> str:
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

    existing_dois = {p["doi"] for p in queue}
    if doi in existing_dois:
        return f"Paper '{title}' already in queue, skipped."

    entry = {
        "doi":        doi,
        "title":      title,
        "authors":    authors,
        "journal":    journal,
        "abstract":   abstract,
        "source_url": source_url,
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
4. If relevant, call add_to_review_queue with your reasoning
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

