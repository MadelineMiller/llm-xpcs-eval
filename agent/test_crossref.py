import re
import requests


def strip_jats(text):
    """Remove JATS XML tags and normalize whitespace."""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def fetch_crossref(doi):
    url = f"https://api.crossref.org/works/{doi}"
    headers = {"User-Agent": "XPCS-Harvester-Bot/1.0 (mailto:momiller@anl.gov)"}

    print(f"Fetching metadata for DOI: {doi}\n")
    response = requests.get(url, headers=headers, timeout=15)

    if response.status_code != 200:
        print(f"ERROR: Got status {response.status_code}")
        return None

    data = response.json()["message"]

    title    = strip_jats(data.get("title", ["N/A"])[0])
    abstract = strip_jats(data.get("abstract", "No abstract available"))
    journal  = data.get("container-title", ["N/A"])[0]
    year     = data.get("published", {}).get("date-parts", [[None]])[0][0]
    authors  = data.get("author", [])
    author_list = [
        f"{a.get('given', '')} {a.get('family', '')}".strip()
        for a in authors
    ]

    print(f"  Title:    {title}")
    print(f"  Journal:  {journal}")
    print(f"  Year:     {year}")
    print(f"  Authors:  {', '.join(author_list[:3])} et al.")
    print(f"  Abstract: {abstract[:400]}")
    print()

    return {
        "doi":      doi,
        "title":    title,
        "abstract": abstract,
        "journal":  journal,
        "year":     year,
        "authors":  author_list,
    }


if __name__ == "__main__":
    test_dois = [
        "10.1039/D5NR05321H",
        "10.1063/5.0305153",
        "10.1111/bpa.70044",
        "10.1016/j.newton.2025.100269",
    ]
    for doi in test_dois:
        fetch_crossref(doi)

