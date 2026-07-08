"""
AGENT_MINIMAL.PY
==================
Stripped-down version of agent.py with ONLY the proven-working
mechanism (temp file + files=) and a BASIC FREE-TEXT prompt —
no JSON schema, no YES/NO framing, no strict extraction rules.

Purpose: confirm, with the least possible complexity, whether GPT-5.2
recognizes and describes the actual content of an uploaded PDF page.

Run: python .\agent_minimal.py
Open: http://localhost:8083/
"""

import io
import base64
import json
import logging
import tempfile
import os
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

# ============================================================
# CONFIG (confirmed working values)
# ============================================================
GENAI_AUTH_URL = "https://genai-api-development-one-it-423929642383.asia-south1.run.app"
LLM_API_URL    = "https://tslgenaiapidev.corp.tatasteel.com/genai"
GENAI_SERVICE_ACCOUNT = "secrets/svc-genai-api-dev-oneit.json"

ADID       = "ayfph2508h"
API_KEY    = "SGB7QI6ZVDLCL6W1"
DEPLOYMENT = "gpt-5.2"

LOCAL_LOG_DIR = "minimal_logs"
os.makedirs(LOCAL_LOG_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Agent Minimal")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ============================================================
# LLM CALL — fresh session EVERY call (not shared globally),
# in case session/token reuse was ever a factor
# ============================================================
def call_llm_with_image(prompt_text: str, image_bytes: bytes) -> Dict[str, Any]:
    creds = service_account.IDTokenCredentials.from_service_account_file(
        GENAI_SERVICE_ACCOUNT, target_audience=GENAI_AUTH_URL
    )
    session = AuthorizedSession(creds)

    messages = [{"role": "user", "content": prompt_text}]
    payload = {
        "deployment_name": DEPLOYMENT,
        "temperature": "0.1",
        "adid": ADID,
        "apikey": API_KEY,
        "messages": json.dumps(messages),
        "max_tokens": "500",
    }

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            files = {"files": ("page.jpg", f, "image/jpeg")}
            logger.info(f"Sending image, size={len(image_bytes)} bytes, prompt={prompt_text!r}")
            response = session.post(
                LLM_API_URL, headers={}, data=payload, files=files, timeout=300
            )
    finally:
        os.remove(tmp_path)

    response.raise_for_status()
    parsed = response.json()
    usage = parsed.get("usage", {})
    content = parsed["choices"][0]["message"].get("content", "")

    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "raw_content": content,
        "full_raw_response": parsed,
    }


# ============================================================
# ENDPOINT
# ============================================================
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
    bitmap = page.render(scale=2.0)
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
    prompt_text = "Describe in detail what you see in this image. Be specific about any text, numbers, labels, or diagrams visible."

    result = call_llm_with_image(prompt_text, image_bytes)

    output = {
        "session_id": session_id,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "filename": payload.filename,
        "image_bytes_sent": len(image_bytes),
        "prompt_tokens": result["prompt_tokens"],
        "completion_tokens": result["completion_tokens"],
        "description": result["raw_content"],
        "rendered_image_saved_at": debug_img_path,
    }

    log_path = os.path.join(LOCAL_LOG_DIR, f"{session_id}.json")
    with open(log_path, "w") as f:
        json.dump({**output, "full_raw_response": result["full_raw_response"]}, f, indent=2)

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
                        'Prompt tokens: ' + data.prompt_tokens + '\\n' +
                        'Completion tokens: ' + data.completion_tokens + '\\n' +
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
