import re
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

URL = "https://photon-science.desy.de/facilities/petra_iii/beamlines/p10_coherence_applications/publications_from_p10/2026/index_eng.html"

NOISE = [
    "files", "bibtex", "ris", "endnote", "xml, text",
    "data privacy", "cookies", "imprint", "accessibility"
]


def clean_lines(lines):
    return [
        l for l in lines
        if l.strip() and not any(n in l.lower() for n in NOISE)
    ]


def parse_publications(page_text):
    publications = []

    # Anchor to just after the breadcrumb line
    anchor = "Home / Facilities"
    anchor_pos = page_text.find(anchor)
    if anchor_pos == -1:
        print("ERROR: Could not find anchor in page text")
        return publications

    # Trim everything before the publications
    pub_text = page_text[anchor_pos:]

    # Skip the breadcrumb line and the "2026" header line
    lines_all = pub_text.split("\n")
    pub_lines = []
    for line in lines_all:
        stripped = line.strip()
        if stripped.startswith("Home /") or stripped == "2026" or stripped == "·":
            continue
        pub_lines.append(stripped)

    pub_text_clean = "\n".join(pub_lines)

    doi_pattern = re.compile(r'\[10\.\d{4,}/[^\]]+\]')

    parts = doi_pattern.split(pub_text_clean)
    doi_matches = doi_pattern.findall(pub_text_clean)

    for i, doi_raw in enumerate(doi_matches):
        doi = doi_raw.strip("[]")
        chunk = parts[i]

        lines = [l.strip() for l in chunk.split("\n") if l.strip()]
        lines = clean_lines(lines)

        if len(lines) < 3:
            print(f"  WARNING: Not enough lines for DOI {doi}, got: {lines}")
            continue

        # Structure is always:
        # AUTHORS (line with semicolons)
        # TITLE
        # JOURNAL  <-- last line before DOI
        journal     = lines[-1]
        title       = lines[-2]
        authors_raw = lines[-3]

        authors = [a.strip() for a in authors_raw.split(";") if a.strip()]

        publications.append({
            "doi":     doi,
            "title":   title,
            "authors": authors,
            "journal": journal,
        })

    return publications


def scrape_desy_p10():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )

    print(f"Fetching: {URL}\n")
    driver.get(URL)
    time.sleep(5)

    page_text = driver.find_element("tag name", "body").text
    driver.quit()

    publications = parse_publications(page_text)

    print("=" * 60)
    print(f"PUBLICATIONS FOUND: {len(publications)}")
    print("=" * 60)

    for i, pub in enumerate(publications):
        print(f"\n[{i+1}]")
        print(f"  Title:   {pub['title']}")
        if len(pub['authors']) > 1:
            print(f"  Authors: {pub['authors'][0]} et al.")
        else:
            print(f"  Authors: {pub['authors']}")
        print(f"  Journal: {pub['journal']}")
        print(f"  DOI:     {pub['doi']}")

    return publications


if __name__ == "__main__":
    scrape_desy_p10()

