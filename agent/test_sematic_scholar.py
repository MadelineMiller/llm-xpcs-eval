import requests

def fetch_semantic_scholar(doi):
    url = f"https://api.semanticscholar.org/graph/v1/paper/{doi}"
    params = {"fields": "title,abstract,authors,year,journal"}
    headers = {"User-Agent": "XPCS-Harvester-Bot/1.0 (mailto:momiller@anl.gov)"}

    print(f"Trying Semantic Scholar for DOI: {doi}\n")
    response = requests.get(url, headers=headers, params=params, timeout=15)

    if response.status_code != 200:
        print(f"  ERROR: Got status {response.status_code}")
        return None

    data = response.json()

    title    = data.get("title", "N/A")
    abstract = data.get("abstract", "No abstract available")
    year     = data.get("year", "N/A")
    authors  = [a.get("name", "") for a in data.get("authors", [])]
    journal  = data.get("journal", {})
    journal_name = journal.get("name", "N/A") if journal else "N/A"

    print(f"  Title:    {title}")
    print(f"  Journal:  {journal_name}")
    print(f"  Year:     {year}")
    print(f"  Authors:  {', '.join(authors[:3])} et al.")
    print(f"  Abstract: {abstract[:400] if abstract else 'No abstract available'}")

    return {
        "doi":      doi,
        "title":    title,
        "abstract": abstract,
        "journal":  journal_name,
        "year":     year,
        "authors":  authors,
    }


if __name__ == "__main__":
    fetch_semantic_scholar("10.1016/j.newton.2025.100269")

