import requests
import json
import importlib.util
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env and config the same way agent.py does
load_dotenv(Path(__file__).parent.parent / ".env")

config_path = Path(__file__).parent.parent / "config.py"
spec = importlib.util.spec_from_file_location("project_config", config_path)
project_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(project_config)
LLM_CONFIG = project_config.LLM_CONFIG

ARGO_API_URL = os.getenv("ARGO_API_URL")
ARGO_USER    = os.getenv("ARGO_USER")

# Define a simple test tool
tools = [
    {
        "name": "get_paper_info",
        "description": "Get information about a paper given its title",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "The title of the paper"
                }
            },
            "required": ["title"]
        }
    }
]

messages = [
    {
        "role": "user",
        "content": "Use the get_paper_info tool to look up this paper: 'Coherent X-ray scattering and speckle dynamics in colloidal systems'"
    }
]

payload = {
    "user":        ARGO_USER,
    "model":       LLM_CONFIG["model"],
    "messages":    messages,
    "tools":       tools,
    "temperature": LLM_CONFIG["temperature"],
    "top_p":       LLM_CONFIG["top_p"],
    "max_tokens":  500,
}

print("Sending request to Argo API with tool definitions...")
print(f"Model: {LLM_CONFIG['model']}")
print(f"URL:   {ARGO_API_URL}")
print()

response = requests.post(ARGO_API_URL, json=payload, timeout=60)

print(f"Status code: {response.status_code}")
print()
print("Raw response:")
print(json.dumps(response.json(), indent=2))

