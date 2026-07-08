"""
MINIMAL DIAGNOSTIC APP
========================
Purpose: bare-minimum test to SEE whether the image is really reaching
GPT-5.2, with a simple upload UI, JSON output, and a saved JSON log
you can inspect directly.

Flow:
1. Open browser, upload a PDF
2. Server renders first page to JPEG (pypdfium2)
3. Sends to GPT-5.2 via the PROVEN working mechanism (temp file + files=)
4. Asks the model to describe what it sees AND confirm image was received
5. Returns JSON: {"image_received": true/false, "description": "...", "raw_response": "..."}
6. Saves the full request/response as a JSON log file you can open and inspect

Run: python .\diagnostic_app.py
Then open: http://localhost:8082/
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

LOCAL_LOG_DIR = "diagnostic_logs"
os.makedirs(LOCAL_LOG_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Diagnostic App")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_creds = service_account.IDTokenCredentials.from_service_account_file(
    GENAI_SERVICE_ACCOUNT, target_audience=GENAI_AUTH_URL
)
_authed_session = AuthorizedSession(_creds)


# ============================================================
# LLM CALL (exact proven-working mechanism)
# ============================================================
def call_llm_with_image(prompt_text: str, image_bytes: bytes) -> Dict[str, Any]:
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
            logger.info(f"Sending image to GPT-5.2, size={len(image_bytes)} bytes")
            response = _authed_session.post(
                LLM_API_URL, headers={}, data=payload, files=files, timeout=300
            )
    finally:
        os.remove(tmp_path)

    response.raise_for_status()
    parsed = response.json()
    usage = parsed.get("usage", {})
    content = parsed["choices"][0]["message"].get("content", "")

    return {
        "status_code": response.status_code,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "raw_content": content,
        "full_raw_response": parsed,
    }


# ============================================================
# ENDPOINTS
# ============================================================
class UploadPayload(BaseModel):
    filename: str
    content_b64: str


@app.post("/api/diagnose")
async def diagnose(payload: UploadPayload):
    session_id = "DIAG-" + str(uuid.uuid4())[:8]

    header_split = payload.content_b64.split(',', 1)
    encoded = header_split[1] if len(header_split) == 2 else header_split[0]
    pdf_bytes = base64.b64decode(encoded)

    # Render first page to JPEG
    pdf = pdfium.PdfDocument(pdf_bytes)
    page = pdf[0]
    bitmap = page.render(scale=2.0)
    pil_img = bitmap.to_pil()
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    buffered = io.BytesIO()
    pil_img.save(buffered, format="JPEG", quality=85)
    image_bytes = buffered.getvalue()

    # Ask the model to describe AND explicitly confirm image receipt
    prompt_text = (
        "First, state clearly: did you receive an image attachment with this "
        "message? Answer 'YES' or 'NO'. Then, if YES, describe in 2-3 sentences "
        "what the image shows. Return your answer as plain text, not JSON."
    )

    result = call_llm_with_image(prompt_text, image_bytes)

    image_received = "yes" in result["raw_content"].lower()[:50]

    output = {
        "session_id": session_id,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "filename": payload.filename,
        "image_bytes_sent": len(image_bytes),
        "prompt_tokens": result["prompt_tokens"],
        "completion_tokens": result["completion_tokens"],
        "image_received_by_model": image_received,
        "model_description": result["raw_content"],
    }

    # Save full JSON log (including full raw response) for inspection
    log_path = os.path.join(LOCAL_LOG_DIR, f"{session_id}.json")
    with open(log_path, "w") as f:
        json.dump({
            **output,
            "full_raw_response": result["full_raw_response"],
        }, f, indent=2)

    logger.info(f"Saved diagnostic log: {log_path}")

    return output


HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>Diagnostic App</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 700px; margin: 40px auto; padding: 0 20px; }
        h1 { color: #333; }
        #dropzone { border: 2px dashed #999; padding: 40px; text-align: center; margin: 20px 0; }
        button { padding: 10px 20px; font-size: 16px; cursor: pointer; }
        #result { margin-top: 20px; padding: 15px; background: #f5f5f5; border-radius: 5px; white-space: pre-wrap; font-family: monospace; }
        .yes { color: green; font-weight: bold; }
        .no { color: red; font-weight: bold; }
    </style>
</head>
<body>
    <h1>Diagnostic App - Image Detection Test</h1>
    <p>Upload a PDF. This will render page 1, send it to GPT-5.2, and tell you plainly whether the model received the image.</p>

    <input type="file" id="fileInput" accept=".pdf" />
    <button onclick="runDiagnostic()">Run Diagnostic</button>

    <div id="result"></div>

    <script>
        function runDiagnostic() {
            const fileInput = document.getElementById('fileInput');
            const resultDiv = document.getElementById('result');
            if (!fileInput.files.length) {
                alert('Please select a PDF file first.');
                return;
            }
            const file = fileInput.files[0];
            resultDiv.textContent = 'Processing... this may take up to 30 seconds.';

            const reader = new FileReader();
            reader.onload = async function(e) {
                const base64 = e.target.result;
                try {
                    const resp = await fetch('/api/diagnose', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ filename: file.name, content_b64: base64 })
                    });
                    const data = await resp.json();
                    const receivedClass = data.image_received_by_model ? 'yes' : 'no';
                    const receivedText = data.image_received_by_model ? 'YES - Image was received' : 'NO - Image was NOT received';
                    resultDiv.innerHTML =
                        '<div class="' + receivedClass + '">' + receivedText + '</div><br>' +
                        'Session ID: ' + data.session_id + '\\n' +
                        'Image bytes sent: ' + data.image_bytes_sent + '\\n' +
                        'Prompt tokens used: ' + data.prompt_tokens + '\\n' +
                        'Completion tokens: ' + data.completion_tokens + '\\n\\n' +
                        'Model response:\\n' + data.model_description;
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
    uvicorn.run(app, host="0.0.0.0", port=8082)
