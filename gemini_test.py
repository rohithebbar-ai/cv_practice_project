"""
AGENT_MINIMAL.PY
================
Stripped-down version of agent.py with ONLY the proven-working
mechanism (temp file + files=) and a BASIC FREE-TEXT prompt —
no JSON schema, no YES/NO framing, no strict extraction rules.

Purpose: confirm, with the least possible complexity, whether Gemini
recognizes and describes the actual content of an uploaded PDF page.

Run: python .\agent_minimal.py
Open: http://localhost:8083/
"""

import io
import os
import cv2
import json
import time
import base64
import requests
import logging
import pandas as pd
import numpy as np
import uuid
import datetime
from typing import List, Dict, Any
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pypdfium2 as pdfium
from PIL import Image

from google.oauth2 import service_account
from google.auth.transport.requests import AuthorizedSession
from dotenv import load_dotenv

load_dotenv()

SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE")
url1 = os.getenv("URL1")
url2 = os.getenv("URL2")

# ========================================================
# AUTH SESSION
# ========================================================

credentials = service_account.IDTokenCredentials.from_service_account_file(SERVICE_ACCOUNT_FILE, target_audience=url1)

authed_session = AuthorizedSession(credentials)

# ========================================================
# GEMINI CALL
# ========================================================

def call_gemini_llm(sys_prompt, user_prompt, file_path, max_tokens=12000):

    for attempt in range(3):
        try:
            payload = {
                "deployment_name": "gemini-3.5-flash",
                "temperature": "0.0",
                "adid": os.getenv("P_No"),
                "apikey": os.getenv("API_KEY"),
                "grounding": "0",
                "max_tokens": str(max_tokens),
                "messages": json.dumps([
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ])
            }

            file_extension = os.path.splitext(file_path)[1].lower()

            mime_type = ''
            if file_extension in ['.jpg', '.jpeg']:
                mime_type = 'image/jpeg'
            else:
                raise ValueError("Unsupported file type")

            files = [('file', (os.path.basename(file_path), open(file_path, 'rb'), mime_type))]
            print(files)
            headers = {}

            response = authed_session.post(url2, headers=headers, data=payload, files=files, timeout = 180)
            # response = authed_session.post(url2, data=payload)
            # response = authed_session.post(url2, headers=headers, data=payload, files=files)

            if response.status_code == 200:
                break

            logging.warning(f"Retry {attempt+1} - Status: {response.status_code}")
            time.sleep(2)

        except Exception as e:
            logging.error(f"Retry {attempt+1} failed: {e}")
            time.sleep(2)

    if response.status_code != 200:
        raise Exception(f"API Error: {response.status_code}")

    response_json = response.json()

    try:
        return response_json["candidates"][0]["content"]["parts"][0]["text"]
    except:
        return ""

# ========================================================
# FASTAPI APP / ENDPOINT
# ========================================================

LOCAL_LOG_DIR = "minimal_logs"
os.makedirs(LOCAL_LOG_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Agent Minimal")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class UploadPayload(BaseModel):
    filename: str
    content_b64: str

@app.post("/api/describe")
async def describe(payload: UploadPayload):
    session_id = "MIN-" + str(uuid.uuid4())[:8]

    header_split = payload.content_b64.split(',', 1)
    encoded = header_split[1] if len(header_split) == 2 else header_split[0]
    pdf_bytes = base64.b64decode(encoded)

    pdf = pdfium.PdfDocument(pdf_bytes)
    page = pdf[0]
    bitmap = page.render(scale=1.0)
    pil_img = bitmap.to_pil()
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    buffered = io.BytesIO()
    pil_img.save(buffered, format="JPEG", quality=85)
    image_bytes = buffered.getvalue()

    # Save the exact rendered image for visual inspection
    debug_img_path = os.path.join(LOCAL_LOG_DIR, f"{session_id}_rendered.jpg")
    with open(debug_img_path, "wb") as f:
        f.write(image_bytes)

    # BASIC FREE-TEXT PROMPT — no JSON, no YES/NO framing, just describe
    sys_prompt = "You are a helpful assistant."
    user_prompt = "Describe exactly what you see in this engineering drawing. Include any titles, section labels, dimension numbers, drawing numbers, or table content you can read."

    description = call_gemini_llm(sys_prompt, user_prompt, debug_img_path)

    output = {
        "session_id": session_id,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "filename": payload.filename,
        "image_bytes_sent": len(image_bytes),
        "description": description,
        "rendered_image_saved_at": debug_img_path,
    }

    log_path = os.path.join(LOCAL_LOG_DIR, f"{session_id}.json")
    with open(log_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Saved log: {log_path}")
    return output

HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>Agent Minimal - Basic Image Recognition Test</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 700px; margin: 40px auto; padding: 0 20px; }
        h1 { color: #333; }
        button { padding: 10px 20px; font-size: 16px; cursor: pointer; margin-top: 10px; }
        #result { margin-top: 20px; padding: 15px; background: #f5f5f5; border-radius: 5px; white-space: pre-wrap; font-family: monospace; }
    </style>
</head>
<body>
    <h1>Agent Minimal - Basic Recognition Test</h1>
    <p>Upload a PDF. This sends ONLY a simple "describe what you see" prompt - no JSON schema, no YES/NO question.</p>

    <input type="file" id="fileInput" accept=".pdf" />
    <button onclick="runTest()">Describe Image</button>

    <div id="result"></div>

    <script>
        function runTest() {
            const fileInput = document.getElementById('fileInput');
            const resultDiv = document.getElementById('result');
            if (!fileInput.files.length) { alert('Select a PDF first.'); return; }
            const file = fileInput.files[0];
            resultDiv.textContent = 'Processing... may take up to 30 seconds.';

            const reader = new FileReader();
            reader.onload = async function(e) {
                try {
                    const resp = await fetch('/api/describe', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ filename: file.name, content_b64: e.target.result })
                    });
                    const data = await resp.json();
                    resultDiv.textContent =
                        'Session: ' + data.session_id + '\\n' +
                        'Image bytes sent: ' + data.image_bytes_sent + '\\n' +
                        'Rendered image saved at: ' + data.rendered_image_saved_at + '\\n\\n' +
                        'DESCRIPTION:\\n' + data.description;
                } catch (err) {
                    resultDiv.textContent = 'Error: ' + err;
                }
            };
            reader.readAsDataURL(file);
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(content=HTML_CONTENT)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8083)
