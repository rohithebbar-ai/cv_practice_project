import io
import base64
import json
import re
import logging
import requests
import traceback
import uuid
import datetime
from typing import List, Dict, Any, Optional
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pypdfium2 as pdfium
from PIL import Image
import pandas as pd
import os

from google.oauth2 import service_account
from google.auth.transport.requests import AuthorizedSession

# ============================================================
# 0. GCP CONFIGURATION (disabled for local testing)
# ============================================================

# prod_credentials_path = "svc-openai-prod.json"
# dev_credentials_path = "tsl-generative-ai-430637b950be.json"
# bq_client = bigquery.Client.from_service_account_json(prod_credentials_path)
# storage_client = storage.Client.from_service_account_json(dev_credentials_path)

bq_client = None       # TEMP: disabled for local testing
storage_client = None  # TEMP: disabled for local testing

GCP_PROJECT_ID = "tsl-datalake-prod"
GCS_BUCKET_NAME = "engg_drawing_analysis"
BQ_DATASET = "TSLDIGITALASSISTANT"
BQ_EXTRACTION_TABLE = GCP_PROJECT_ID + "." + BQ_DATASET + ".engg_draw_2d_extract"
BQ_FEEDBACK_TABLE = GCP_PROJECT_ID + "." + BQ_DATASET + ".engg_draw_2d_extract_feedback"

# ============================================================
# LOGGING SETUP
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Alekh2D")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 1. LOCAL STORAGE HELPERS (GCS/BigQuery disabled for testing)
# ============================================================

LOCAL_UPLOADS_DIR = "local_uploads"
os.makedirs(LOCAL_UPLOADS_DIR, exist_ok=True)

def upload_image_to_gcs(base64_str: str, filename: str) -> str:
    """TEMP (local testing): saves image locally instead of GCS."""
    try:
        header_split = base64_str.split(',', 1)
        encoded = header_split[1] if len(header_split) == 2 else header_split[0]
        image_bytes = base64.b64decode(encoded)

        local_path = os.path.join(LOCAL_UPLOADS_DIR, filename)
        with open(local_path, "wb") as f:
            f.write(image_bytes)

        logger.info(f"Saved image locally: {local_path}")
        return f"local://{local_path}"
    except Exception as e:
        logger.error(f"Failed to save image locally: {e}")
        return f"local://error/{filename}"


LOCAL_LOG_FILE = "local_logs.jsonl"

def log_to_bigquery(table_id: str, rows_to_insert: List[Dict[str, Any]]):
    """TEMP (local testing): logs to a local JSONL file instead of BigQuery."""
    try:
        with open(LOCAL_LOG_FILE, "a") as f:
            for row in rows_to_insert:
                row_with_table = {"table_id": table_id, **row}
                f.write(json.dumps(row_with_table) + "\n")
        logger.info(f"Logged {len(rows_to_insert)} row(s) locally (BigQuery disabled).")
    except Exception as e:
        logger.error(f"Local logging failed: {e}")


# ============================================================
# 2. GENAI SETUP & LLM LOGIC (files= based image upload)
# ============================================================

GENAI_AUTH_URL = "https://genai-api-development-one-it-423929642383.asia-south1.run.app"
LLM_API_URL    = "https://tslgenaiapidev.corp.tatasteel.com/genai"
GENAI_SERVICE_ACCOUNT = "secrets/svc-genai-api-dev-oneit.json"

ADID       = "ayfph2508h"          # confirmed working adid
API_KEY    = "SGB7QI6ZVDLCL6W1"    # confirmed working genai api key
DEPLOYMENT = "gpt-5.2"

_creds = service_account.IDTokenCredentials.from_service_account_file(
    GENAI_SERVICE_ACCOUNT,
    target_audience=GENAI_AUTH_URL
)
_authed_session = AuthorizedSession(_creds)


def call_llm(prompt_text: str, image_files: List[Dict[str, Any]] = None) -> str:
    """
    Calls the GENAI gateway.
    - prompt_text: full text instruction (system prompt + filenames etc.)
    - image_files: list of {"filename": str, "bytes": bytes} dicts,
      sent via files= (multipart), NOT embedded in messages content.
    """
    logger.info("Preparing LLM payload (files= based)...")

    messages = [{"role": "user", "content": prompt_text}]
    payload = {
        "deployment_name": DEPLOYMENT,
        "temperature": "0.1",
        "adid": ADID,
        "apikey": API_KEY,
        "messages": json.dumps(messages),
        "max_tokens": "4000",
    }

    files = []
    if image_files:
        for img in image_files:
            files.append(
                ("files", (img["filename"], img["bytes"], "image/jpeg"))
            )

    try:
        response = _authed_session.post(
            LLM_API_URL,
            headers={},
            data=payload,
            files=files if files else None,
            timeout=300
        )
        response.raise_for_status()
        raw_response_text = response.text

        llm_output = raw_response_text
        try:
            parsed_json = json.loads(raw_response_text)
            if isinstance(parsed_json, dict):
                if 'choices' in parsed_json and len(parsed_json['choices']) > 0:
                    msg = parsed_json['choices'][0].get('message', {})
                    if 'content' in msg:
                        llm_output = msg['content']
                elif 'response' in parsed_json:
                    llm_output = parsed_json['response']
                elif 'result' in parsed_json:
                    llm_output = parsed_json['result']
        except json.JSONDecodeError:
            pass

        return str(llm_output)
    except Exception as e:
        logger.error("LLM Call Failed: " + str(e))
        return "[]"


def extract_json_array(text: str) -> List[Dict[str, Any]]:
    clean_text = text.strip()
    if clean_text.startswith("```json"):
        clean_text = clean_text[7:]
    if clean_text.startswith("```"):
        clean_text = clean_text[3:]
    if clean_text.endswith("```"):
        clean_text = clean_text[:-3]
    clean_text = clean_text.strip()

    try:
        parsed = json.loads(clean_text)
        if isinstance(parsed, str):
            parsed = json.loads(parsed)  # handle double-encoded JSON string
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict) and "data" in parsed:
            return parsed["data"]
        if isinstance(parsed, dict) and "content" in parsed:
            inner = parsed["content"]
            if isinstance(inner, str):
                inner = json.loads(inner)
            if isinstance(inner, list):
                return inner
    except Exception:
        pass

    match = re.search(r'\[\s*\{.*?\}\s*\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    return []


# ============================================================
# 3. STATE MANAGEMENT & IMAGE UTILS
# ============================================================

class ImagePayload(BaseModel):
    name: str
    b64: str

class Base64File(BaseModel):
    name: str
    content_b64: str

class UploadJSONRequest(BaseModel):
    files: List[Base64File]

class ExtractRequest(BaseModel):
    session_id: str
    images: List[ImagePayload]

class FeedbackRequest(BaseModel):
    session_id: str
    image_name: str
    feedback_remarks: str
    corrected_data: List[Dict[str, Any]]

def pil_to_b64(pil_img: Image.Image, format="JPEG") -> str:
    buffered = io.BytesIO()
    if pil_img.mode in ("RGBA", "P", "LA") or pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    pil_img.save(buffered, format=format, quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode()


# ============================================================
# 4. FASTAPI ENDPOINTS
# ============================================================

@app.post("/api/upload")
async def upload_files_json(payload: UploadJSONRequest):
    logger.info("JSON Upload Endpoint Hit!")
    images_data = []
    try:
        for file in payload.files:
            header_split = file.content_b64.split(',', 1)
            encoded = header_split[1] if len(header_split) == 2 else header_split[0]
            content = base64.b64decode(encoded)
            filename = file.name

            if filename.lower().endswith(".pdf"):
                pdf = pdfium.PdfDocument(content)
                for i in range(len(pdf)):
                    page = pdf[i]
                    bitmap = page.render(scale=2.0)
                    images_data.append({
                        "name": filename + "_p" + str(i + 1),
                        "b64": pil_to_b64(bitmap.to_pil())
                    })
            else:
                pil_img = Image.open(io.BytesIO(content))
                images_data.append({"name": filename, "b64": pil_to_b64(pil_img)})

        return {"images": images_data}
    except Exception as e:
        logger.error("Upload error: " + str(e))
        return {"error": "SERVER ERROR: " + str(e)}


@app.post("/api/extract")
async def extract_data(payload: ExtractRequest):
    session_id = payload.session_id

    system_instruction = """You are an expert QA Engineering AI for TATA STEEL GROWTH SHOP (TGS). Extract a comprehensive inspection checklist from the provided engineering drawing(s).

CRITICAL TGS-STYLE RULES (M-34597 Template):
1. EXTRACT ALL VISIBLE DIMENSIONS: Extract every OD, Length, Chamfer, Groove, Hole, Tap, PCD, Radius, and GD&T parameter.
2. GROUPING: Synchronize blocks by diameter zone (e.g., OD + its length + its chamfer + its radius must be grouped sequentially). Follow the drawing flow.
3. Dim_Description: MUST strictly be the feature name ONLY. Do not include values here. Use exact vocabulary: OUTER DIA, LENGTH, TOTAL LENGTH, CHAMFER, RELIEF GROOVE, RADIUS, KEY WAY, COUNTER BORE, REAMED HOLE, DRILL SIZE, TAP SIZE, PCD, DIM, CONCENTRICITY. (Do NOT put dimensions in this column!).
4. Specified: Only the dimension/fit/thread (e.g., \u00d8320 r6, 1546, 2x45\u00b0, M16x30).
5. Tolerance: Extract the EXACT numerical limits (e.g., +0.027/+0.059). If the drawing only shows the fit class (e.g., 'h9', 'f7'), output '(num tol to fill)'. If there is no tolerance, output '-'.
6. Dim_Type MUST be one of these exact codes: OD, LD, CH, R, DH, INT, EXT, KS, GD.
7. Dwg_View: Identify Grid references (e.g., A1, B2) if visible on drawing borders.

ALSO extract physical named components (e.g. hopper, pulley, idler, conveyor,
walkway, monorail, sizer, scraper, feeder, magnetic separator) visible on the
drawing, using this additional schema per row:
- "Category": "dimension" or "component"
- "Component_Name": populated only for component rows (e.g. "Truck Dump Hopper")
For dimension rows, Component_Name stays empty. For component rows, dimension-
specific fields (Tolerance, Measuring_Tools, etc.) stay empty or "N/A".

OUTPUT FORMAT:
Return ONLY a valid JSON array of objects. Do not write markdown, do not write explanations.
Each object MUST have the following keys exactly:
"image_name", "Pt_No", "Dwg_View", "Dim_Type", "Dim_Description", "Specified", "Tolerance", "Measuring_Tools", "MC_No", "Insp_Type", "Category", "Component_Name", "bbox"

"bbox" MUST be an array of exactly 4 integers [x, y, width, height] representing the bounding box of the feature on the image. If you cannot confidently identify the bbox, return an empty array [].
"""

    prompt_text = system_instruction + "\n\n=== CURRENT BLUEPRINTS TO ANALYZE ===\n"

    gcs_uris = []
    image_files_for_llm = []

    for img in payload.images:
        prompt_text += f"\nFilename: {img.name}\n"

        header_split = img.b64.split(',', 1)
        encoded = header_split[1] if len(header_split) == 2 else header_split[0]
        image_bytes = base64.b64decode(encoded)

        image_files_for_llm.append({
            "filename": img.name,
            "bytes": image_bytes
        })

        uri = upload_image_to_gcs(img.b64, f"{session_id}_{img.name}")
        gcs_uris.append({"image_name": img.name, "gcs_uri": uri})

    # Call AI — images sent via files=, not embedded in prompt text
    raw_response = call_llm(prompt_text, image_files_for_llm)
    parsed_data = extract_json_array(raw_response)

    if not parsed_data and payload.images:
        parsed_data = [{
            "image_name": payload.images[0].name, "Pt_No": 1, "Dwg_View": "FAIL",
            "Dim_Description": "API PARSE ERROR", "Specified": "Check Terminal Logs",
            "Tolerance": "N/A", "Measuring_Tools": "N/A", "MC_No": "", "Insp_Type": "F",
            "Category": "", "Component_Name": "", "bbox": []
        }]

    for item in parsed_data:
        item["_id"] = str(uuid.uuid4())
        if "bbox" not in item:
            item["bbox"] = []

    # Prepare sanitized log (text prompt only, no raw image bytes)
    log_row = {
        "session_id": session_id,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_used": DEPLOYMENT,
        "prompt_used": prompt_text,
        "images": json.dumps(gcs_uris),
        "raw_llm_response": raw_response,
        "parsed_json": json.dumps(parsed_data)
    }
    log_to_bigquery(BQ_EXTRACTION_TABLE, [log_row])

    return {"data": parsed_data}


@app.post("/api/feedback")
async def receive_feedback(payload: FeedbackRequest):
    logger.info("Feedback Submission Endpoint Hit!")

    log_row = {
        "session_id": payload.session_id,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "image_name": payload.image_name,
        "feedback_remarks": payload.feedback_remarks,
        "corrected_data": json.dumps(payload.corrected_data)
    }
    log_to_bigquery(BQ_FEEDBACK_TABLE, [log_row])

    return {"status": "success"}


@app.post("/api/download_excel")
async def download_excel(payload: List[Dict[str, Any]]):
    formatted_data = []
    for item in payload:
        formatted_data.append({
            "Pt. No.": item.get("Pt_No", ""), "Dwg View": item.get("Dwg_View", ""),
            "Dim Type": item.get("Dim_Type", ""), "Dim Description": item.get("Dim_Description", ""),
            "Specified": item.get("Specified", ""), "Tolerance": item.get("Tolerance", ""),
            "Measuring Tools": item.get("Measuring_Tools", ""), "M/C No.": item.get("MC_No", ""),
            "Insp. Type": item.get("Insp_Type", ""), "Actual(S)": "", "Actual(F)": "",
            "Status": "", "Remarks": ""
        })
    df = pd.DataFrame(formatted_data)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='TGS Checklist')
    output.seek(0)
    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=TGS_Checklist_Corrected.xlsx"}
    )


# ============================================================
# 5. FRONTEND HTML/JS  (unchanged — keep your existing HTML_CONTENT here)
# ============================================================

HTML_CONTENT = """
<!-- KEEP YOUR EXISTING HTML_CONTENT STRING EXACTLY AS-IS.
     Nothing in the frontend needs to change for this fix —
     paste your current HTML_CONTENT block back in here. -->
"""

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    return HTMLResponse(content=HTML_CONTENT, headers={"Cache-Control": "no-cache"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)
