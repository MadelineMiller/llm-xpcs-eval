import threading
import uvicorn
import os
import re
import html as html_lib
import uuid
import tempfile
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from dotenv import load_dotenv
from weights_manager import load_weights, save_weights, get_all_docs
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader

import requests as http_requests  # rename to avoid conflict with FastAPI Request

def lookup_crossref_metadata(title: str) -> dict:
    """Try to find paper metadata from CrossRef using the title."""
    try:
        resp = http_requests.get(
            "https://api.crossref.org/works",
            params={"query.title": title, "rows": 1},
            timeout=10
        )
        if resp.status_code != 200:
            return {}

        items = resp.json().get("message", {}).get("items", [])
        if not items:
            return {}

        item = items[0]

        # Extract authors
        authors = []
        for author in item.get("author", []):
            name = f"{author.get('given', '')} {author.get('family', '')}".strip()
            if name:
                authors.append(name)

        # Extract journal
        journal = ""
        containers = item.get("container-title", [])
        if containers:
            journal = containers[0]

        # Extract year
        year = ""
        date_parts = item.get("published-print", item.get("published-online", {})).get("date-parts", [[]])
        if date_parts and date_parts[0]:
            year = str(date_parts[0][0])

        # Extract DOI and URL
        doi = item.get("DOI", "")
        url = f"https://doi.org/{doi}" if doi else ""

        # Extract title
        titles = item.get("title", [])
        found_title = titles[0] if titles else ""

        return {
            "title": found_title,
            "authors": authors,
            "journal": journal,
            "year": year,
            "doi": doi,
            "url": url,
        }

    except Exception as e:
        print(f"[CROSSREF] Lookup failed: {e}")
        return {}


def clean_title(title):
    """Strip MathML/XML tags and extract readable text from CrossRef titles."""
    if not title:
        return title
    title = re.sub(r'<[^>]+>', '', title)
    title = re.sub(r'\s+', ' ', title).strip()
    return title


load_dotenv()

admin_app = FastAPI()

client = QdrantClient(
    host=os.getenv('QDRANT_HOST', 'localhost'),
    port=int(os.getenv('QDRANT_PORT', 6333))
)

embeddings = HuggingFaceEmbeddings(
    model_name="allenai/scibert_scivocab_uncased",
    model_kwargs={'device': 'cpu'}
)

COLLECTION_NAME = os.getenv('QDRANT_COLLECTION_NAME', 'xpcs_documents')


def get_bar(weight: int) -> str:
    filled = round(weight / 10)
    empty = 10 - filled
    return (
        '<i class="fa-solid fa-square" style="color:#2196f3;"></i>' * filled
        + '<i class="fa-regular fa-square" style="color:#555;"></i>' * empty
    )


@admin_app.get("/", response_class=HTMLResponse)
async def admin_page():
    docs = get_all_docs(client, COLLECTION_NAME)
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
        safe_title = html_lib.escape(title or source)
        safe_author_str = html_lib.escape(author_str)
        safe_journal = html_lib.escape(journal)

        meta_lines = f'<span class="meta-label">Author:</span> {safe_author_str}'
        if journal:
            meta_lines += f'<br><span class="meta-label">Journal:</span> {safe_journal}'
        if year:
            meta_lines += f'<br><span class="meta-label">Year:</span> {year}'
        if doi and url:
            meta_lines += f'<br><span class="meta-label">DOI:</span> <a href="{url}" target="_blank" class="doi-link">{html_lib.escape(doi)}</a>'
        elif doi:
            meta_lines += f'<br><span class="meta-label">DOI:</span> {html_lib.escape(doi)}'

        data_title = html_lib.escape((title or "").lower())
        data_author = html_lib.escape(author_str.lower())
        data_source = html_lib.escape(source.lower())

        rows += f"""
        <tr class="doc-row" id="row-{source}" data-title="{data_title}" data-author="{data_author}" data-source="{data_source}">
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
                    <button onclick="deleteDoc('{safe_source}')" title="Delete document" class="delete-btn">
                        <i class="fa-solid fa-trash"></i>
                    </button>
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
        .search-wrapper {{
            display: flex;
            align-items: center;
            gap: 10px;
            background: #2a2a2a;
            border: 1px solid #555;
            border-radius: 6px;
            padding: 10px 14px;
            margin-bottom: 16px;
        }}
        .search-wrapper:focus-within {{
            border-color: #2196f3;
        }}
        .search-wrapper i {{
            color: #777;
            font-size: 14px;
        }}
        .search-box {{
            flex: 1;
            background: none;
            border: none;
            outline: none;
            color: #e0e0e0;
            font-size: 15px;
        }}
        .search-box::placeholder {{ color: #777; }}
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
            color: #2196f3;
            text-decoration: none;
        }}
        .doi-link:hover {{
            text-decoration: underline;
            color: #64b5f6;
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
        button:hover {{ background: #2196f3; color: white; border-color: #2196f3; }}
        .delete-btn:hover {{ background: #e53935; border-color: #e53935; }}
        .toast {{
            position: fixed;
            bottom: 24px;
            right: 24px;
            background: #2196f3;
            color: white;
            padding: 10px 20px;
            border-radius: 6px;
            display: none;
            font-size: 14px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.4);
        }}
        .toast.error {{ background: #e53935; }}
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
        .upload-section {{
            background: #2a2a2a;
            border: 2px dashed #555;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            text-align: center;
            transition: border-color 0.2s;
        }}
        .upload-section:hover {{ border-color: #2196f3; }}
        .upload-section.dragover {{ border-color: #2196f3; background: #333; }}
        .upload-btn {{
            padding: 10px 24px;
            background: #2196f3;
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 15px;
            cursor: pointer;
            margin-top: 10px;
        }}
        .upload-btn:hover {{ background: #1e88e5; }}
        .upload-btn:disabled {{ background: #555; cursor: not-allowed; }}
        .file-input-label {{
            display: inline-block;
            padding: 8px 18px;
            background: #333;
            border: 1px solid #555;
            border-radius: 4px;
            cursor: pointer;
            color: #e0e0e0;
            font-size: 14px;
        }}
        .file-input-label:hover {{ background: #444; }}
        #fileInput {{ display: none; }}
        .selected-file {{ color: #2196f3; margin-top: 8px; font-size: 13px; }}
        .progress-bar {{
            width: 100%;
            height: 6px;
            background: #333;
            border-radius: 3px;
            margin-top: 10px;
            display: none;
        }}
        .progress-fill {{
            height: 100%;
            background: #2196f3;
            border-radius: 3px;
            width: 0%;
            transition: width 0.3s;
        }}
        .confirm-overlay {{
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.7);
            display: none;
            align-items: center;
            justify-content: center;
            z-index: 100;
        }}
        .confirm-box {{
            background: #2a2a2a;
            border: 1px solid #555;
            border-radius: 8px;
            padding: 24px;
            max-width: 400px;
            text-align: center;
        }}
        .confirm-box h3 {{ color: #e53935; margin-bottom: 12px; }}
        .confirm-box p {{ color: #ccc; margin-bottom: 20px; font-size: 14px; }}
        .confirm-btns {{ display: flex; gap: 10px; justify-content: center; }}
        .confirm-btns button {{ padding: 8px 20px; font-size: 14px; }}
        .btn-cancel {{ background: #333; }}
        .btn-delete {{ background: #e53935; border-color: #e53935; color: white; }}
        .btn-delete:hover {{ background: #c62828; }}
        .doc-count {{ color: #888; font-size: 13px; margin-bottom: 16px; }}
        .reset-btn {{
            padding: 8px 18px;
            background: #333;
            color: #e0e0e0;
            border: 1px solid #555;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            margin-bottom: 16px;
            transition: background 0.15s;
        }}
        .reset-btn:hover {{ background: #e53935; border-color: #e53935; color: white; }}
        .title-input {{
            width: 80%;
            padding: 8px 12px;
            font-size: 14px;
            border: 1px solid #555;
            border-radius: 4px;
            background: #333;
            color: #e0e0e0;
            outline: none;
        }}
        .title-input:focus {{
            border-color: #2196f3;
        }}
                .edit-field {{
            margin-bottom: 10px;
        }}
        .edit-field label {{
            display: block;
            color: #aaa;
            font-size: 12px;
            margin-bottom: 4px;
            font-weight: bold;
        }}
        .edit-field input {{
            width: 100%;
            padding: 8px 12px;
            font-size: 14px;
            border: 1px solid #555;
            border-radius: 4px;
            background: #333;
            color: #e0e0e0;
            outline: none;
            box-sizing: border-box;
        }}
        .edit-field input:focus {{ border-color: #2196f3; }}
        .btn-save {{ background: #2196f3; border-color: #2196f3; color: white; }}
        .btn-save:hover {{ background: #1e88e5; }}
    </style>
</head>
<body>
    <a class="back" href="http://localhost:8000"><i class="fa-solid fa-arrow-left"></i> Back to Chat</a>
    <h1><i class="fa-solid fa-book"></i> XPCS Document Manager</h1>
    <p class="subtitle">
        Adjust each document's relevance weight. (Default is 50/100) <br>
        <strong>Higher weight = prioritized more when answering questions.</strong><br>
    </p>

    <button class="reset-btn" onclick="resetAllWeights()">
        <i class="fa-solid fa-rotate-left"></i> Reset All Document Weights
    </button>

    <div class="upload-section" id="uploadSection">
        <i class="fa-solid fa-cloud-arrow-up" style="font-size:28px; color:#2196f3; margin-bottom:8px;"></i>
        <p style="margin:4px 0; color:#ccc;">Upload a PDF to add to the knowledge base</p>
        <label class="file-input-label" for="fileInput">
            <i class="fa-solid fa-file-pdf"></i> Choose PDF
        </label>
        <input type="file" id="fileInput" accept=".pdf" onchange="fileSelected()">
        <div class="selected-file" id="selectedFile"></div>
        <div id="titleConfirmSection" style="display:none; margin-top:12px;">
            <p style="color:#aaa; font-size:13px; margin-bottom:6px;">
                <i class="fa-solid fa-pen"></i> Confirm or edit the publication title:
            </p>
            <input type="text" id="titleInput" class="title-input" placeholder="Publication title...">
        </div>
        <div class="progress-bar" id="progressBar"><div class="progress-fill" id="progressFill"></div></div>
        <br>
        <button class="upload-btn" id="uploadBtn" onclick="uploadFile()" style="display:none">
            <i class="fa-solid fa-upload"></i> Upload PDF
        </button>
    </div>


    <div class="search-wrapper">
        <i class="fa-solid fa-magnifying-glass"></i>
        <input type="text" class="search-box" id="searchBox" placeholder="Search by title or author..." oninput="filterDocs()">
    </div>

    <p class="doc-count" id="docCount">{len(docs)} documents in database</p>

    <table>
        <thead>
            <tr>
                <th><i class="fa-solid fa-file-lines"></i> Document</th>
                <th><i class="fa-solid fa-sliders"></i> Weight & Actions</th>
            </tr>
        </thead>
        <tbody id="docTableBody">
            {rows}
        </tbody>
    </table>

    <p class="no-results" id="noResults"><i class="fa-solid fa-circle-exclamation"></i> No documents match your search.</p>

    <div class="toast" id="toast"><i class="fa-solid fa-check"></i> <span id="toastMsg">Saved!</span></div>

    <div class="confirm-overlay" id="confirmOverlay">
        <div class="confirm-box">
            <h3><i class="fa-solid fa-triangle-exclamation"></i> Delete Document</h3>
            <p>Are you sure you want to permanently delete<br><strong id="confirmFilename"></strong><br>and all its chunks from the knowledge base?</p>
            <div class="confirm-btns">
                <button class="btn-cancel" onclick="cancelDelete()">Cancel</button>
                <button class="btn-delete" onclick="confirmDelete()"><i class="fa-solid fa-trash"></i> Delete</button>
            </div>
        </div>
    </div>

    <div class="confirm-overlay" id="resetOverlay">
        <div class="confirm-box">
            <h3><i class="fa-solid fa-triangle-exclamation"></i> Reset All Weights</h3>
            <p>Are you sure you want to reset all document weights back to the default <strong>50/100</strong>?</p>
            <div class="confirm-btns">
                <button class="btn-cancel" onclick="cancelReset()">Cancel</button>
                <button class="btn-delete" onclick="confirmReset()"><i class="fa-solid fa-rotate-left"></i> Reset All</button>
            </div>
        </div>
    </div>

        <div class="confirm-overlay" id="metadataOverlay">
        <div class="confirm-box" style="max-width:500px; text-align:left;">
            <h3 style="color:#2196f3;"><i class="fa-solid fa-circle-info"></i> Metadata Not Found</h3>
            <p style="color:#aaa; font-size:13px;">CrossRef couldn't find this paper. Please fill in the metadata manually.</p>
            <div class="edit-field">
                <label>Title</label>
                <input type="text" id="metaTitle">
            </div>
            <div class="edit-field">
                <label>Authors (comma separated)</label>
                <input type="text" id="metaAuthors" placeholder="Jane Smith, John Doe">
            </div>
            <div class="edit-field">
                <label>Journal</label>
                <input type="text" id="metaJournal">
            </div>
            <div class="edit-field">
                <label>Year</label>
                <input type="text" id="metaYear">
            </div>
            <div class="edit-field">
                <label>DOI</label>
                <input type="text" id="metaDoi" placeholder="10.xxxx/xxxxx">
            </div>
            <div class="edit-field">
                <label>URL</label>
                <input type="text" id="metaUrl" placeholder="https://doi.org/...">
            </div>
            <div class="confirm-btns" style="margin-top:16px;">
                <button class="btn-cancel" onclick="cancelMetadata()">Cancel</button>
                <button class="btn-save" onclick="submitWithMetadata()"><i class="fa-solid fa-upload"></i> Upload</button>
            </div>
        </div>
    </div>


    <script>
        let pendingDeleteSource = null;

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

            showToast('Saved!', false);
        }}

        function getBar(val) {{
            const filled = Math.round(val / 10);
            const empty  = 10 - filled;
            let bar = '';
            for (let i = 0; i < filled; i++) {{
                bar += '<i class="fa-solid fa-square" style="color:#2196f3;"></i>';
            }}
            for (let i = 0; i < empty; i++) {{
                bar += '<i class="fa-regular fa-square" style="color:#555;"></i>';
            }}
            return bar;
        }}

        function showToast(msg, isError) {{
            const toast = document.getElementById('toast');
            document.getElementById('toastMsg').textContent = msg;
            toast.className = isError ? 'toast error' : 'toast';
            toast.style.display = 'block';
            setTimeout(() => toast.style.display = 'none', 4000);
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

        function deleteDoc(source) {{
            pendingDeleteSource = source;
            document.getElementById('confirmFilename').textContent = source;
            document.getElementById('confirmOverlay').style.display = 'flex';
        }}

        function cancelDelete() {{
            pendingDeleteSource = null;
            document.getElementById('confirmOverlay').style.display = 'none';
        }}

        async function confirmDelete() {{
            const source = pendingDeleteSource;
            document.getElementById('confirmOverlay').style.display = 'none';

            const resp = await fetch('/delete-doc', {{
                method: 'POST',
                headers: {{'Content-Type': 'application/json'}},
                body: JSON.stringify({{filename: source}})
            }});

            const data = await resp.json();
            if (data.ok) {{
                const row = document.getElementById('row-' + source);
                if (row) row.remove();
                const count = document.querySelectorAll('.doc-row').length;
                document.getElementById('docCount').textContent = count + ' documents in database';
                showToast('Deleted ' + source + ' (' + data.chunks_deleted + ' chunks)', false);
            }} else {{
                showToast('Error: ' + (data.error || 'unknown'), true);
            }}

            pendingDeleteSource = null;
        }}

        async function fileSelected() {{
            const input = document.getElementById('fileInput');
            const label = document.getElementById('selectedFile');
            const btn   = document.getElementById('uploadBtn');
            const titleSection = document.getElementById('titleConfirmSection');
            const titleInput   = document.getElementById('titleInput');

            if (input.files.length > 0) {{
                label.textContent = input.files[0].name;

                // Send file to extract title
                titleSection.style.display = 'block';
                titleInput.value = 'Extracting title...';
                titleInput.disabled = true;
                btn.style.display = 'none';

                const formData = new FormData();
                formData.append('file', input.files[0]);

                try {{
                    const resp = await fetch('/extract-title', {{
                        method: 'POST',
                        body: formData
                    }});
                    const data = await resp.json();
                    if (data.ok) {{
                        titleInput.value = data.guessed_title;
                    }} else {{
                        titleInput.value = input.files[0].name.replace('.pdf', '').replace(/_/g, ' ');
                    }}
                }} catch (e) {{
                    titleInput.value = input.files[0].name.replace('.pdf', '').replace(/_/g, ' ');
                }}

                titleInput.disabled = false;
                btn.style.display = 'inline-block';
            }} else {{
                label.textContent = '';
                btn.style.display = 'none';
                titleSection.style.display = 'none';
            }}
        }}

            async function uploadFile() {{
                const input = document.getElementById('fileInput');
                if (!input.files.length) return;

                const btn  = document.getElementById('uploadBtn');
                const bar  = document.getElementById('progressBar');
                const fill = document.getElementById('progressFill');
                const titleInput = document.getElementById('titleInput');

                btn.disabled = true;
                btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Looking up metadata...';
                bar.style.display = 'block';
                fill.style.width = '30%';

                const formData = new FormData();
                formData.append('file', input.files[0]);
                formData.append('title', titleInput.value);

                try {{
                    fill.style.width = '60%';
                    const resp = await fetch('/upload-doc', {{
                        method: 'POST',
                        body: formData
                    }});

                    fill.style.width = '90%';
                    const data = await resp.json();

                    if (data.ok) {{
                        fill.style.width = '100%';
                        showToast('Added ' + data.filename + ' (' + data.chunks + ' chunks)', false);
                        setTimeout(function() {{ location.reload(); }}, 3000);
                    }} else if (data.needs_metadata) {{
                        // CrossRef didn't find it — show metadata form
                        fill.style.width = '0%';
                        bar.style.display = 'none';
                        btn.disabled = false;
                        btn.innerHTML = '<i class="fa-solid fa-upload"></i> Upload PDF';

                        pendingUploadFile = input.files[0];
                        document.getElementById('metaTitle').value = data.guessed_title || '';
                        document.getElementById('metaAuthors').value = '';
                        document.getElementById('metaJournal').value = '';
                        document.getElementById('metaYear').value = '';
                        document.getElementById('metaDoi').value = '';
                        document.getElementById('metaUrl').value = '';
                        document.getElementById('metadataOverlay').style.display = 'flex';
                    }} else {{
                        showToast('Error: ' + (data.error || 'unknown'), true);
                    }}
                }} catch (e) {{
                    showToast('Upload failed: ' + e.message, true);
                }}

                btn.disabled = false;
                btn.innerHTML = '<i class="fa-solid fa-upload"></i> Upload PDF';
                setTimeout(function() {{
                    bar.style.display = 'none';
                    fill.style.width = '0%';
                }}, 1500);
            }}

            function cancelMetadata() {{
                pendingUploadFile = null;
                document.getElementById('metadataOverlay').style.display = 'none';
            }}

            async function submitWithMetadata() {{
                if (!pendingUploadFile) return;

                document.getElementById('metadataOverlay').style.display = 'none';

                const btn  = document.getElementById('uploadBtn');
                const bar  = document.getElementById('progressBar');
                const fill = document.getElementById('progressFill');

                btn.disabled = true;
                btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Indexing...';
                bar.style.display = 'block';
                fill.style.width = '30%';

                const formData = new FormData();
                formData.append('file', pendingUploadFile);
                formData.append('title', document.getElementById('metaTitle').value);
                formData.append('authors', document.getElementById('metaAuthors').value);
                formData.append('journal', document.getElementById('metaJournal').value);
                formData.append('year', document.getElementById('metaYear').value);
                formData.append('doi', document.getElementById('metaDoi').value);
                formData.append('url', document.getElementById('metaUrl').value);
                formData.append('skip_crossref', 'true');

                try {{
                    fill.style.width = '60%';
                    const resp = await fetch('/upload-doc', {{
                        method: 'POST',
                        body: formData
                    }});

                    fill.style.width = '90%';
                    const data = await resp.json();

                    if (data.ok) {{
                        fill.style.width = '100%';
                        showToast('Added ' + data.filename + ' (' + data.chunks + ' chunks)', false);
                        setTimeout(function() {{ location.reload(); }}, 3000);
                    }} else {{
                        showToast('Error: ' + (data.error || 'unknown'), true);
                    }}
                }} catch (e) {{
                    showToast('Upload failed: ' + e.message, true);
                }}

                pendingUploadFile = null;
                btn.disabled = false;
                btn.innerHTML = '<i class="fa-solid fa-upload"></i> Upload PDF';
                setTimeout(function() {{
                    bar.style.display = 'none';
                    fill.style.width = '0%';
                }}, 1500);
            }}


        const uploadSection = document.getElementById('uploadSection');
        uploadSection.addEventListener('dragover', (e) => {{
            e.preventDefault();
            uploadSection.classList.add('dragover');
        }});
        uploadSection.addEventListener('dragleave', () => {{
            uploadSection.classList.remove('dragover');
        }});
        uploadSection.addEventListener('drop', (e) => {{
            e.preventDefault();
            uploadSection.classList.remove('dragover');
            const input = document.getElementById('fileInput');
            input.files = e.dataTransfer.files;
            fileSelected();
        }});
        function resetAllWeights() {{
            document.getElementById('resetOverlay').style.display = 'flex';
        }}
        function cancelReset() {{
            document.getElementById('resetOverlay').style.display = 'none';
        }}
        async function confirmReset() {{
            document.getElementById('resetOverlay').style.display = 'none';

            const resp = await fetch('/reset-weights', {{
                method: 'POST',
                headers: {{'Content-Type': 'application/json'}}
            }});

            const data = await resp.json();
            if (data.ok) {{
                document.querySelectorAll('.doc-row').forEach(function(row) {{
                    const hidden = row.querySelector('input[type="hidden"]');
                    if (hidden) hidden.value = 50;
                    const valSpan = row.querySelector('[id^="val-"]');
                    if (valSpan) valSpan.textContent = '50/100';
                    const barSpan = row.querySelector('[id^="bar-"]');
                    if (barSpan) barSpan.innerHTML = getBar(50);
                }});
                showToast('All weights reset to 50/100', false);
            }} else {{
                showToast('Error: ' + (data.error || 'unknown'), true);
            }}
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

@admin_app.post("/reset-weights")
async def reset_weights():
    """Reset all document weights to default (50)."""
    try:
        save_weights({})
        print("[RESET] All weights reset to default")
        return {"ok": True}
    except Exception as e:
        print(f"[RESET ERROR] {e}")
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


@admin_app.post("/delete-doc")
async def delete_doc(request: Request):
    """Delete all chunks for a document from Qdrant and remove its weight."""
    try:
        data = await request.json()
        filename = data["filename"]

        # Scroll to find all point IDs matching this source
        point_ids = []
        offset = None
        while True:
            results, offset = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in results:
                source = os.path.basename(point.payload.get("source", ""))
                if source == filename:
                    point_ids.append(point.id)

            if offset is None:
                break

        if not point_ids:
            return JSONResponse(
                status_code=404,
                content={"ok": False, "error": f"No chunks found for {filename}"}
            )

        # Delete all matching points by ID
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=point_ids
        )

        # Remove from weights
        weights = load_weights()
        if filename in weights:
            del weights[filename]
            save_weights(weights)

        print(f"[DELETE] Removed {len(point_ids)} chunks for: {filename}")
        return {"ok": True, "chunks_deleted": len(point_ids)}

    except Exception as e:
        print(f"[DELETE ERROR] {e}")
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})

@admin_app.post("/extract-title")
async def extract_title(file: UploadFile = File(...)):
    """Extract a guessed title from the first page of a PDF."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        os.unlink(tmp_path)

        skip_words = [
            "research papers", "research article", "original article",
            "full paper", "short communication", "letter", "review",
            "open access", "crossmark", "downloaded from", "published by",
            "copyright", "all rights reserved", "doi:", "http",
            "volume", "issue", "pages", "received", "accepted",
            "correspondence", "e-mail", "abstract", "introduction",
            "keywords", "contents lists", "journal of", "acta",
            "edited by", "editor", "university of", "department of",
            "manuscript", "submitted", "revised", "available online",
            "elsevier", "springer", "wiley", "nature", "science",
            "orcid", "author contributions", "funding", "acknowledgment",
            "supplementary", "supporting information", "table of contents",
            "issn", "printed in", "published online", "article in press",
        ]

        guessed_title = file.filename.replace(".pdf", "").replace("_", " ").replace("-", " ")

        if pages and len(pages[0].page_content) > 50:
            first_lines = pages[0].page_content[:500].split("\n")
            for line in first_lines:
                cleaned = line.strip()
                if len(cleaned) < 20:
                    continue
                if any(skip in cleaned.lower() for skip in skip_words):
                    continue
                if "@" in cleaned or "http" in cleaned:
                    continue
                guessed_title = cleaned[:200]
                break

        return {"ok": True, "guessed_title": guessed_title}

    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})

@admin_app.post("/upload-doc")
async def upload_doc(
    file: UploadFile = File(...),
    title: str = Form(""),
    authors: str = Form(""),
    journal: str = Form(""),
    year: str = Form(""),
    doi: str = Form(""),
    url: str = Form(""),
    skip_crossref: str = Form("false"),
):
    """Upload a PDF, split into chunks, embed, and add to Qdrant."""
    try:
        if not file.filename.endswith('.pdf'):
            return JSONResponse(status_code=400, content={"ok": False, "error": "Only PDF files are supported"})


        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(pages)

        # Use provided metadata directly if skip_crossref is true
        if skip_crossref == "true":
            doc_title = title.strip() or file.filename.replace(".pdf", "").replace("_", " ")
            doc_authors = [a.strip() for a in authors.split(",") if a.strip()]
            doc_journal = journal.strip()
            doc_year = year.strip()
            doc_doi = doi.strip()
            doc_url = url.strip()
            print(f"[UPLOAD] Using user-provided metadata for {file.filename}")
        else:
            search_title = title.strip() if title.strip() else file.filename.replace(".pdf", "").replace("_", " ")
            print(f"[CROSSREF] Looking up: {search_title[:100]}...")
            metadata = lookup_crossref_metadata(search_title)

            if metadata.get("title"):
                found_lower = metadata["title"].lower()
                search_lower = search_title.lower()
                search_words = set(search_lower.split())
                found_words = set(found_lower.split())
                overlap = search_words & found_words
                min_overlap = max(3, len(search_words) * 0.3)
                if len(overlap) < min_overlap:
                    print(f"[CROSSREF] Poor match — ignoring")
                    metadata = {}

            if metadata.get("title"):
                doc_title = metadata["title"]
                doc_authors = metadata.get("authors", [])
                doc_journal = metadata.get("journal", "")
                doc_year = metadata.get("year", "")
                doc_doi = metadata.get("doi", "")
                doc_url = metadata.get("url", "")
                print(f"[CROSSREF] Found: {doc_title}")
            else:
                # No match — return to let user fill in metadata
                os.unlink(tmp_path)
                return {
                    "ok": False,
                    "needs_metadata": True,
                    "filename": file.filename,
                    "chunks_count": len(chunks),
                    "guessed_title": search_title,
                }

        points = []
        for i, chunk in enumerate(chunks):
            vector = embeddings.embed_query(chunk.page_content)
            point_id = str(uuid.uuid4())
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "source": file.filename,
                        "title": doc_title,
                        "text": chunk.page_content,
                        "page": chunk.metadata.get("page", 0),
                        "authors": doc_authors,
                        "journal": doc_journal,
                        "year": doc_year,
                        "doi": doc_doi,
                        "url": doc_url,
                    }
                )
            )

        BATCH_SIZE = 50
        for i in range(0, len(points), BATCH_SIZE):
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points[i:i + BATCH_SIZE]
            )

        os.unlink(tmp_path)

        print(f"[UPLOAD] Added {file.filename}: {len(chunks)} chunks")
        return {"ok": True, "filename": file.filename, "chunks": len(chunks)}

    except Exception as e:
        print(f"[UPLOAD ERROR] {e}")
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})



def start_admin_server():
    uvicorn.run(admin_app, host="0.0.0.0", port=8001, log_level="warning")


def launch_admin():
    thread = threading.Thread(target=start_admin_server, daemon=True)
    thread.start()
    print("Admin page running at http://localhost:8001")
