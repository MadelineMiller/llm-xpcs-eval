import threading
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse



admin_app = FastAPI()




@admin_app.get("/", response_class=HTMLResponse)
async def admin_page():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>XPCS Document Manager</title>
        <style>
            body { font-family: Arial, sans-serif; padding: 30px; max-width: 1000px; margin: 0 auto; }
            h1   { color: #333; }
            a.back { color: #666; text-decoration: none; font-size: 14px; }
        </style>
    </head>
    <body>
        <a class="back" href="http://localhost:8000">← Back to Chat</a>
        <h1>📚 XPCS Document Manager</h1>
        <p>Document weight controls will appear here.</p>
    </body>
    </html>
    """
    return HTMLResponse(content=html)




def start_admin_server():
    uvicorn.run(admin_app, host="0.0.0.0", port=8001, log_level="warning")


def launch_admin():
    thread = threading.Thread(target=start_admin_server, daemon=True)
    thread.start()
    print("Admin page running at http://localhost:8001")