import threading
import uvicorn
import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv
from weights_manager import load_weights, save_weights, get_all_docs
from qdrant_client import QdrantClient

import html as html_lib

import re

def clean_title(title):
    """Strip MathML/XML tags and extract readable text from CrossRef titles."""
    if not title:
        return title
    # Remove full MathML blocks but keep inner text content
    title = re.sub(r'<[^>]+>', '', title)
    # Collapse extra whitespace
    title = re.sub(r'\s+', ' ', title).strip()
    return title

load_dotenv()

admin_app = FastAPI()

client = QdrantClient(
    host=os.getenv('QDRANT_HOST', 'localhost'),
    port=int(os.getenv('QDRANT_PORT', 6333))
)


def get_bar(weight: int) -> str:
    filled = round(weight / 10)
    empty = 10 - filled
    return (
        '<i class="fa-solid fa-square" style="color:#4caf50;"></i>' * filled
        + '<i class="fa-regular fa-square" style="color:#555;"></i>' * empty
    )


@admin_app.get("/", response_class=HTMLResponse)
async def admin_page():
    docs = get_all_docs(client, os.getenv('QDRANT_COLLECTION_NAME', 'xpcs_documents'))
    weights = load_weights()

    rows = ""
    for doc in docs:
        source = doc["source"]
        title = clean_title(doc["title"])
        authors = doc["authors"]
        year = doc.get("year", "")
        journal = doc.get("journal", "")
        doi = doc.get("doi", "")
        url = doc.get("url", "")
        weight = weights.get(source, 50)

        if len(authors) > 2:
            author_str = f"{authors[0]} et al."
        elif authors:
            author_str = ", ".join(authors)
        else:
            author_str = "Unknown"

        safe_source = source.replace("'", "\\'")

        # Escape HTML entities in display strings
        safe_title      = html_lib.escape(title)
        safe_author_str = html_lib.escape(author_str)
        safe_journal    = html_lib.escape(journal)

        # Build metadata lines
        meta_lines = f'<span class="meta-label">Author:</span> {safe_author_str}'
        if journal:
            meta_lines += f'<br><span class="meta-label">Journal:</span> {safe_journal}'
        if year:
            meta_lines += f'<br><span class="meta-label">Year:</span> {year}'
        if doi and url:
            meta_lines += f'<br><span class="meta-label">DOI:</span> <a href="{url}" target="_blank" class="doi-link">{html_lib.escape(doi)}</a>'
        elif doi:
            meta_lines += f'<br><span class="meta-label">DOI:</span> {html_lib.escape(doi)}'

        # Also escape for data attributes
        data_title  = html_lib.escape(title.lower())
        data_author = html_lib.escape(author_str.lower())
        data_source = html_lib.escape(source.lower())

        rows += f"""
        <tr class="doc-row" data-title="{data_title}" data-author="{data_author}" data-source="{data_source}">
            <td>
                <strong class="doc-title">{safe_title}</strong>
                <div class="doc-meta">{meta_lines}</div>
            </td>
            <td>
                <div style="display:flex; align-items:center; gap:8px; white-space:nowrap;">
                    <button onclick="adjust('{safe_source}', -10)" title="Decrease by 10">
                        <i class="fa-solid fa-minus"></i>
                    </button>
                    <span id="bar-{source}" style="font-size:14px; letter-spacing:2px;">{get_bar(weight)}</span>
                    <button onclick="adjust('{safe_source}', 10)" title="Increase by 10">
                        <i class="fa-solid fa-plus"></i>
                    </button>
                    <span id="val-{source}" style="font-weight:bold; min-width:55px;">{weight}/100</span>
                    <input type="hidden" id="weight-{source}" value="{weight}">
                </div>
            </td>
        </tr>
        """


    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>XPCS Document Manager</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
    <style>
        body {{
            font-family: Arial, sans-serif;
            padding: 30px;
            max-width: 1100px;
            margin: 0 auto;
            background: #1a1a1a;
            color: #e0e0e0;
        }}
        h1 {{ color: #ffffff; }}
        .search-box {{
            width: 100%;
            padding: 10px 14px 10px 36px;
            font-size: 15px;
            border: 1px solid #555;
            border-radius: 6px;
            background: #2a2a2a;
            color: #e0e0e0;
            margin-bottom: 16px;
            box-sizing: border-box;
        }}
        .search-box::placeholder {{ color: #777; }}
        .search-box:focus {{ outline: none; border-color: #4caf50; }}
        .search-wrapper {{
            position: relative;
        }}
        .search-icon {{
            position: absolute;
            left: 12px;
            top: 50%;
            transform: translateY(-50%);
            color: #777;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: #2a2a2a;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 1px 4px rgba(0,0,0,0.4);
        }}
        th {{
            background: #333333;
            padding: 12px 16px;
            text-align: left;
            border-bottom: 2px solid #444;
            color: #ffffff;
        }}
        td {{
            padding: 12px 16px;
            border-bottom: 1px solid #333;
            vertical-align: middle;
        }}
        tr:hover {{ background: #333333; }}
        .doc-title {{
            color: #ffffff;
            font-size: 14px;
            display: block;
            margin-bottom: 6px;
        }}
        .doc-meta {{
            font-size: 12px;
            color: #999;
            line-height: 1.6;
        }}
        .meta-label {{
            color: #bbb;
            font-weight: bold;
        }}
        .doi-link {{
            color: #4caf50;
            text-decoration: none;
        }}
        .doi-link:hover {{
            text-decoration: underline;
            color: #66bb6a;
        }}
        button {{
            padding: 6px 10px;
            cursor: pointer;
            border: 1px solid #555;
            border-radius: 4px;
            background: #333;
            color: #e0e0e0;
            font-size: 14px;
            transition: background 0.15s;
        }}
        button:hover {{ background: #4caf50; color: white; border-color: #4caf50; }}
        .toast {{
            position: fixed;
            bottom: 24px;
            right: 24px;
            background: #4caf50;
            color: white;
            padding: 10px 20px;
            border-radius: 6px;
            display: none;
            font-size: 14px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.4);
        }}
        .toast i {{ margin-right: 6px; }}
        .back {{ color: #aaa; text-decoration: none; font-size: 14px; }}
        .back:hover {{ color: #fff; }}
        .back i {{ margin-right: 4px; }}
        p.subtitle {{ color: #aaa; margin-bottom: 20px; }}
        .no-results {{
            text-align: center;
            padding: 20px;
            color: #666;
            display: none;
        }}
    </style>
</head>
<body>
    <a class="back" href="http://localhost:8000"><i class="fa-solid fa-arrow-left"></i> Back to Chat</a>
    <h1><i class="fa-solid fa-book"></i> XPCS Document Manager</h1>
    <p class="subtitle">
        Adjust each document's relevance weight.<br>
        <strong>Higher weight = prioritized more when answering questions.</strong> Default is 50/100.
    </p>

    <div class="search-wrapper">
        <i class="fa-solid fa-magnifying-glass search-icon"></i>
        <input type="text" class="search-box" id="searchBox" placeholder="Search by title, author, or filename..." oninput="filterDocs()">
    </div>

    <table>
        <thead>
            <tr>
                <th><i class="fa-solid fa-file-lines"></i> Document</th>
                <th><i class="fa-solid fa-sliders"></i> Weight</th>
            </tr>
        </thead>
        <tbody id="docTableBody">
            {rows}
        </tbody>
    </table>

    <p class="no-results" id="noResults"><i class="fa-solid fa-circle-exclamation"></i> No documents match your search.</p>

    <div class="toast" id="toast"><i class="fa-solid fa-check"></i> Saved!</div>

    <script>
        async function adjust(source, delta) {{
            const hidden = document.getElementById('weight-' + source);
            let current  = parseInt(hidden.value);
            let newVal   = Math.max(0, Math.min(100, current + delta));

            hidden.value = newVal;
            document.getElementById('val-' + source).textContent = newVal + '/100';
            document.getElementById('bar-' + source).innerHTML  = getBar(newVal);

            await fetch('/set-weight', {{
                method: 'POST',
                headers: {{'Content-Type': 'application/json'}},
                body: JSON.stringify({{filename: source, weight: newVal}})
            }});

            showToast();
        }}

        function getBar(val) {{
            const filled = Math.round(val / 10);
            const empty  = 10 - filled;
            let bar = '';
            for (let i = 0; i < filled; i++) {{
                bar += '<i class="fa-solid fa-square" style="color:#4caf50;"></i>';
            }}
            for (let i = 0; i < empty; i++) {{
                bar += '<i class="fa-regular fa-square" style="color:#555;"></i>';
            }}
            return bar;
        }}

        function showToast() {{
            const toast = document.getElementById('toast');
            toast.style.display = 'block';
            setTimeout(() => toast.style.display = 'none', 1500);
        }}

        function filterDocs() {{
            const query = document.getElementById('searchBox').value.toLowerCase();
            const rows  = document.querySelectorAll('.doc-row');
            let visible = 0;

            rows.forEach(row => {{
                const title  = row.getAttribute('data-title')  || '';
                const author = row.getAttribute('data-author') || '';
                const source = row.getAttribute('data-source') || '';

                if (title.includes(query) || author.includes(query) || source.includes(query)) {{
                    row.style.display = '';
                    visible++;
                }} else {{
                    row.style.display = 'none';
                }}
            }});

            document.getElementById('noResults').style.display = visible === 0 ? 'block' : 'none';
        }}
    </script>
</body>
</html>"""
    return HTMLResponse(content=html)


@admin_app.post("/set-weight")
async def set_weight(request: Request):
    data = await request.json()
    weights = load_weights()
    weights[data["filename"]] = data["weight"]
    save_weights(weights)
    return {"ok": True}


def start_admin_server():
    uvicorn.run(admin_app, host="0.0.0.0", port=8001, log_level="warning")


def launch_admin():
    thread = threading.Thread(target=start_admin_server, daemon=True)
    thread.start()
    print("Admin page running at http://localhost:8001")
