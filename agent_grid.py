import io
import base64
import json
import re
import logging
import uuid
import datetime
from typing import List, Dict, Any
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time
import pypdfium2 as pdfium
from PIL import Image
import pandas as pd

from google.oauth2 import service_account
from google.auth.transport.requests import AuthorizedSession
import os

from dotenv import load_dotenv
load_dotenv()

# ========================================================
# 0. GCP CONFIGURATION (UPDATE THESE WITH YOUR ACTUAL DETAILS)
# ========================================================
# Point to your downloaded service account JSON file
prod_credentials_path = "svc-openai-prod.json"
dev_credentials_path = "svc-genai-api-dev-oneit.json"
bq_client = None
storage_client = None

GCP_PROJECT_ID = "tsl-datalake-prod"
GCS_BUCKET_NAME = "engg_drawing_analysis"
BQ_DATASET = "TSLDIGITALASSISTANT"
BQ_EXTRACTION_TABLE = GCP_PROJECT_ID + "." + BQ_DATASET + ".engg_draw_2d_extract"
BQ_FEEDBACK_TABLE = GCP_PROJECT_ID + "." + BQ_DATASET + ".engg_draw_2d_extract_feedback"

# --- METICULOUS LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Allekh2D 📐🤖")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LOCAL_UPLOADS_DIR = "local_uploads"
os.makedirs(LOCAL_UPLOADS_DIR, exist_ok=True)

def upload_image_to_gcs(base64_str: str, filename: str) -> str:
    """Uploads a base64 image to GCS and returns the GS URI."""
    try:
        header_split = base64_str.split(',', 1)
        encoded = header_split[1] if len(header_split) == 2 else header_split[0]
        image_bytes = base64.b64decode(encoded)
        logger.info(f"Decoded image bytes for {filename}: {len(image_bytes)} bytes")
        local_path = os.path.join(LOCAL_UPLOADS_DIR, filename)
        with open(local_path, "wb") as f:
            f.write(image_bytes)
        logger.info(f"Saved image locally: {local_path}")
        return f"local://{local_path}"
    except Exception as e:
        logger.error(f"failed to save image locally: {e}")
        return f"local://error/{filename}"

LOCAL_LOG_FILE = "local_logs.jsonl"

def log_to_bigquery(table_id:str, rows_to_insert: List[Dict[str, Any]]):
    try:
        with open(LOCAL_LOG_FILE, "a") as f:
            for row in rows_to_insert:
                row_with_table = {"table_id": table_id, **row}
                f.write(json.dumps(row_with_table)+ "\n")
        logger.info(f"Logged {len(rows_to_insert)} row(s) locally")
    except Exception as e:
        logger.error(f"Locall logging failed: {e}")

# ========================================================
# 2. OPENAI SETUP & LLM LOGIC (now routed to Gemini)
# ========================================================

SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE")
url1 = os.getenv("URL1")
url2 = os.getenv("URL2")

DEPLOYMENT = "gemini-3.5-flash"

_creds = service_account.IDTokenCredentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE,
    target_audience=url1
)

_authed_session = AuthorizedSession(_creds)

def call_llm(prompt_text: str, image_files: List[Dict[str, Any]] = None) -> str:
    """
    Calls the GENAI gateway (Gemini).
    - prompt_text: full text instruction (system prompt + filenames etc.)
    - image_files: list of {"filename": str, "bytes": bytes} dicts,
      sent via files= (multipart, directly from memory), NOT embedded in messages content.
    """
    logger.info("Preparing LLM payload (files= based, Gemini)...")

    messages = [{"role": "user", "content": prompt_text}]
    payload = {
        "deployment_name": DEPLOYMENT,
        "temperature": "0.0",
        "adid": os.getenv("P_No"),
        "apikey": os.getenv("API_KEY"),
        "grounding": "0",
        "max_tokens": "12000",
        "messages": json.dumps(messages)
    }

    response = None
    for attempt in range(3):
        try:
            if image_files:
                img = image_files[0]
                safe_filename = img["filename"]
                if not safe_filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    safe_filename = safe_filename + ".jpg"

                files = [('file', (safe_filename, img["bytes"], "image/jpeg"))]

                response = _authed_session.post(
                    url2,
                    headers={},
                    data=payload,
                    files=files,
                    timeout=180
                )
            else:
                response = _authed_session.post(
                    url2,
                    headers={},
                    data=payload,
                    files=[],
                    timeout=180
                )

            if response.status_code == 200:
                break

            logger.warning(f"Retry {attempt+1} - Status: {response.status_code}")
            time.sleep(2)

        except Exception as e:
            logger.error(f"Retry {attempt+1} failed: {e}")
            time.sleep(2)

    if response is None or response.status_code != 200:
        logger.error(f"LLM Call Failed: status {response.status_code if response else 'no response received'}")
        return "[]"

    try:
        parsed_json = response.json()
        llm_output = parsed_json["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        logger.error("LLM Call Failed (parse): " + str(e))
        logger.error(f"RAW RESPONSE WAS: {response.text}")
        return "[]"

    return str(llm_output)

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

# ========================================================
# 3. STATE MANAGEMENT & IMAGE UTILS
# ========================================================

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
    feedback_remarks: str  # <--- ADD THIS LINE
    corrected_data: List[Dict[str, Any]]

def pil_to_b64(pil_img: Image.Image, format="JPEG") -> str:
    buffered = io.BytesIO()
    if pil_img.mode in ("RGBA", "P", "LA") or pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    pil_img.save(buffered, format=format, quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode()

# ========================================================
# 4. FASTAPI ENDPOINTS
# ========================================================

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
                    images_data.append({"name": filename + "_p" + str(i+1), "b64": pil_to_b64(bitmap.to_pil())})
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

    system_instruction = """You are an expert QA Engineering AI for TATA STEEL GROWTH SHOP (TGS). Extract a comprehensive inspection checklist from engineering drawings.

=== 0. GRID REFERENCE SYSTEM (mandatory for every row) ===
Like a road-map, this drawing's border has column numbers (e.g. 1,2,3...12)
printed along the top/bottom edge and row letters (e.g. A,B,C...H) printed
along the left/right edge. For EVERY row you output, regardless of Category,
read the nearest column number and row letter from the border and populate
"Dwg_View" with that grid reference in the format "<ROW LETTER>-<COLUMN NUMBER>"
(e.g. "G-11", "C-4"). If a feature spans more than one cell, give the range
(e.g. "F10-G11"). This is how engineers actually reference locations on
these drawings (e.g. "see detail at G-11"), and it is far more reliable
than estimating pixel coordinates, so it must always be filled in - never
leave Dwg_View blank.

Use Dwg_View together with Component_Name to preserve relationships: the
same named component (e.g. "Idler") may legitimately appear in more than
one view on the same sheet (e.g. once in a cross-section detail, once in a
plan view). Keep these as separate rows with the same Component_Name but
different Dwg_View values, so which physical instance is which is never
ambiguous, and nothing gets merged or lost.

This drawing may contain THREE distinct kinds of content. Extract ALL of
them - do not prioritize dimensions over the others. Each kind of content
produces its own row(s) in the output, distinguished by "Category".

=== 1. DIMENSIONS (Category="dimension") ===
CRITICAL TGS-STYLE RULES (M-34597 Template):
1. EXTRACT ALL VISIBLE DIMENSIONS: Extract every OD, Length, Chamfer, Groove, Hole, Tap, PCD, Radius, and GD&T parameter.
2. GROUPING: Synchronize blocks by diameter zone (e.g., OD + its length + its chamfer + its radius must be grouped sequentially).
3. Dim_Description: MUST strictly be the feature name ONLY. Do not include values here. Use exact vocabulary: OUTER DIA, LENGTH, TOTAL LENGTH, CHAMFER...
4. Specified: Only the dimension/fit/thread (e.g., Ø320 r6, 1546, 2x45°, M16x30).
5. Tolerance: Extract the EXACT numerical limits (e.g., +0.027/+0.059). If the drawing only shows the fit class (e.g., 'h9', 'f7'), output the fit class in Tolerance.
6. Dim_Type MUST be one of these exact codes for dimension rows only: OD, LD, CH, R, DH, INT, EXT, KS, GD.
7. Dwg_View: the grid reference per Section 0 above (e.g. "G-11").
For dimension rows, Category="dimension", Component_Name="", Quantity="".

=== 2. NAMED COMPONENTS (Category="component") ===
Extract physical named components (e.g. hopper, pulley, idler, conveyor,
walkway, monorail, sizer, scraper, feeder, magnetic separator) visible on the
drawing:
- Category="component"
- Component_Name=the component's name (e.g. "Truck Dump Hopper")
- Dwg_View=the grid reference per Section 0 above
- Dim_Type="NA"
- Quantity=populate only if the drawing itself explicitly shows a count next to this component (e.g. a legend symbol marked "28 NOS."); otherwise leave "".
- Dimension-specific fields (Tolerance, Measuring_Tools, etc.) stay "" or "N/A".

=== 3. TITLE BLOCK (Category="title_block") ===
Locate the title block, typically bottom-right, containing "TATA STEEL LIMITED".
Its exact position and size vary by sheet, so search for it rather than assuming
a fixed location. Extract every labeled field inside it as its own row:
- Category="title_block"
- Component_Name=the plant/location line (e.g. "RM Location - West Bokaro Coal Mine")
- Dwg_View=the grid reference per Section 0 above (title blocks are typically bottom-right, e.g. "A-1")
- Dim_Description=the field label exactly as printed (e.g. "REV", "DRG NO", "MATERIAL", "SCALE", "SHEET NO", "SHEET SIZE", "DATE", "DEPARTMENT", "EQUIP/AREA", "DETAIL", "DRN", "CHD", "APPD", "WEIGHT IN KG")
- Specified=that field's value (e.g. "4", "GAD-38-01-02-03-307-009", "COAL", "1:100", "2 OF 2", "A1")
- Dim_Type="NA", Quantity=""
Extract every field visible in the block, not only REV and DRG NO.

=== DO NOT EXTRACT: SCHEDULE/SPECIFICATION TABLES ===
Do NOT extract rows from typed or manually-populated schedule tables such as
"CONVEYOR CHARACTERISTICS", "LOAD DATA", "MATERIAL" summary tables, or any
similar administrative table. These are filled in separately from the
drawing itself and are not reliable source data for this extraction. Only
extract what is actually drawn: dimensions, named components, and the
title block.

OUTPUT FORMAT:
Return ONLY a valid JSON array of objects. Do not write markdown, do not write explanations.
Each object MUST have the following keys exactly:
"image_name", "Pt_No", "Dwg_View", "Dim_Type", "Dim_Description", "Specified", "Tolerance", "Measuring_Tools", "MC_No", "Insp_Type", "Category", "Component_Name", "Quantity", "bbox"
"Dwg_View" is mandatory on every row per Section 0 above.
"bbox" MUST be an array of exactly 4 integers [x, y, width, height] in
pixel coordinates, where x+width <= image width and y+height <= image height
as stated next to that image's filename below. Only provide a bbox for
Category="dimension" rows, drawn tightly around that specific dimension's
callout/number. For Category="component" and "title_block" rows, always
return bbox=[] - never box an entire region or table. Dwg_View, not bbox,
is the primary locator for non-dimension rows.
"""

    prompt_text = system_instruction + "\n\n=== CURRENT BLUEPRINTS TO ANALYZE ===\n"
    #prompt_text = "Can you see the uploaded image? Answer only YES or NO. If yes, describe it in one sentence."
    gcs_uris = []
    image_files_for_llm = []

    for img in payload.images:
        header_split = img.b64.split(',', 1)
        encoded = header_split[1] if len(header_split) == 2 else header_split[0]
        image_bytes = base64.b64decode(encoded)

        try:
            with Image.open(io.BytesIO(image_bytes)) as pil_probe:
                img_w, img_h = pil_probe.size
        except Exception:
            img_w, img_h = None, None

        if img_w and img_h:
            prompt_text += f"\nFilename: {img.name} | Image size: {img_w}px width x {img_h}px height\n"
        else:
            prompt_text += f"\nFilename: {img.name}\n"

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
            "image_name": payload.images[0].name,
            "Pt_No": 1,
            "Dwg_View": "FAIL",
            "Dim_Type": "ERR",
            "Dim_Description": "API PARSE ERROR",
            "Specified": "Check Terminal Logs",
            "Tolerance": "N/A",
            "Measuring_Tools": "N/A",
            "MC_No": "",
            "Insp_Type": "F",
            "Category": "",
            "Component_Name": "",
            "Quantity": "",
            "bbox": []
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

    # LOG FEEDBACK TO BIGQUERY
    log_row = {
        "session_id": payload.session_id,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "image_name": payload.image_name,
        "feedback_remarks": payload.feedback_remarks, # <--- ADD THIS LINE
        "corrected_data": json.dumps(payload.corrected_data)
    }
    log_to_bigquery(BQ_FEEDBACK_TABLE, [log_row])

    return {"status": "success"}

@app.post("/api/download_excel")
async def download_excel(payload: List[Dict[str, Any]]):
    formatted_data = []
    for item in payload:
        formatted_data.append({
            "Category": item.get("Category", ""), "Component": item.get("Component_Name", ""),
            "View": item.get("Dwg_View", ""),
            "Pt. No.": item.get("Pt_No", ""),
            "Dim Type": item.get("Dim_Type", ""), "Dim Description": item.get("Dim_Description", ""),
            "Specified": item.get("Specified", ""), "Qty": item.get("Quantity", ""), "Tolerance": item.get("Tolerance", ""),
            "Measuring Tools": item.get("Measuring_Tools", ""), "M/C No.": item.get("MC_No", ""),
            "Insp. Type": item.get("Insp_Type", ""), "Actual(S)": "", "Actual(F)": "", "Status": "", "Remarks": ""
        })
    df = pd.DataFrame(formatted_data)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='TGS Checklist')
    output.seek(0)
    return StreamingResponse(output, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers={"Content-Disposition": "attachment; filename=TGS_Checklist_Corrected.xlsx"})

# ========================================================
# 5. FRONTEND HTML/JS
# ========================================================

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Allekh2D 📐🤖</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/fabric.js/5.3.1/fabric.min.js"></script>
    <style>
        #statusLog { font-family: monospace; font-size: 12px; }
        .brand-title {
            font-family: Arial, Helvetica, sans-serif;
            letter-spacing: -0.5px;
        }

        /* Metallic 3D Steel Beam Effect for the 'l' */
        .steel-beam {
            font-family: "Courier New", Courier, monospace;
            color: #64748b; /* Slate/Steel Gray */
            text-shadow: 1px 1px 0px #cbd5e1, -1px -1px 0px #334155; /* 3D metallic bevel */
            display: inline-block;
            transform: scaleY(1.1); /* Makes the beam slightly taller and heavier */
            margin: 0 1px;
        }
        .resizable-container {
            display: flex;
            height: 80vh;
        }
        .left-panel {
            min-width: 300px;
            max-width: 60%;
            width: 45%;
            resize: horizontal;
            overflow: auto;
            border-right: 4px solid #cbd5e1;
            padding-right: 15px;
            display: flex;
            flex-direction: column;
        }
        .right-panel {
            flex-grow: 1;
            padding-left: 15px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        table input, table select { width: 100%; border: 1px solid transparent; padding: 2px; border-radius: 3px; }
        table input:focus, table select:focus { border: 1px solid #3b82f6; outline: none; background: #eff6ff; }
        .row-correct { background-color: #dcfce7 !important; }
        .drawing-active { border: 2px solid red !important; box-shadow: 0 0 10px red; }
    </style>
</head>
<body class="bg-gray-100 min-h-screen text-gray-800">

<div id="errorBanner" style="display:none; background:red; color:white; padding:10px; font-weight:bold; text-align:center; position:fixed; top:0; width:100%; z-index:9999;"></div>

<div id="labelModal" class="hidden fixed inset-0 bg-gray-900 bg-opacity-70 flex justify-center items-center z-50">
    <div class="bg-white p-6 rounded-lg shadow-xl w-96">
        <h3 class="text-lg font-bold mb-4">Tag Annotated Area</h3>
        <label class="block text-sm font-semibold mb-1">Category:</label>
        <select id="boxCategory" class="w-full border rounded p-2 mb-4 bg-gray-50">
            <option value="Material">Material</option><option value="GD&T">GD&T</option>
            <option value="General Tolerance">General Tolerance</option><option value="Surface Roughness">Surface Roughness</option>
        </select>
        <label class="block text-sm font-semibold mb-1">Unique Label:</label>
        <input type="text" id="boxLabel" class="w-full border rounded p-2 mb-6 bg-gray-50" placeholder="e.g., OD 120p6" />
        <div class="flex justify-end gap-2">
            <button id="cancelBoxBtn" class="px-4 py-2 bg-gray-200 rounded font-semibold">Discard</button>
            <button id="saveBoxBtn" class="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 font-semibold">Save Tag</button>
        </div>
    </div>
</div>

<div class="max-w-[98%] mx-auto p-4 pt-4">
    <div class="flex justify-between items-end border-b pb-2 mb-4">
        <h1 class="text-4xl font-bold text-blue-700 brand-title">
            Allekh2D 📐🤖
        </h1>
        <span class="text-gray-500 font-mono text-sm bg-gray-200 px-2 py-1 rounded" id="sessionIdLabel">Session Loading...</span>
    </div>

    <!-- ON SCREEN CONSOLE LOGS -->
    <div class="bg-black text-green-400 p-2 rounded mb-4 overflow-y-auto h-20 shadow-inner" id="statusLog">
        <div>> System initialized. Connecting to BigQuery and GCS endpoints... Ready.</div>
    </div>

    <div class="resizable-container">

        <!-- LEFT PANEL: UPLOAD & CHECKLIST (RESIZABLE) -->
        <div class="left-panel">
            <div class="bg-white p-4 rounded shadow mb-4 border-t-4 border-blue-500 flex-shrink-0">
                <input type="file" id="fileInput" multiple accept=".pdf,.png,.jpg,.jpeg" class="mb-2 block w-full text-sm" />
                <div class="flex gap-2">
                    <button id="uploadBtn" class="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded flex-1">Upload</button>
                    <button id="extractBtn" class="bg-green-600 hover:bg-green-700 text-white font-bold py-2 px-4 rounded flex-1 hidden">Run AI Extraction</button>
                </div>
            </div>

            <div id="dataPanel" class="bg-white p-4 rounded shadow border-t-4 border-green-500 flex-grow flex flex-col hidden overflow-hidden">
                <h2 class="text-lg font-bold text-green-700 mb-2">Interactive Feedback Checklist</h2>
                <div id="categoryFilterBar" class="flex flex-wrap gap-1 mb-2 flex-shrink-0"></div>
                <div class="overflow-auto border rounded flex-grow mb-2 bg-gray-50 relative">
                    <table class="min-w-full text-xs text-left whitespace-nowrap">
                        <thead class="bg-gray-200 sticky top-0 shadow-sm z-10">
                            <tr>
                                <th class="py-2 px-1 border-r text-center">✓</th>
                                <th class="py-2 px-2 border-r">Category</th>
                                <th class="py-2 px-2 border-r">Component</th>
                                <th class="py-2 px-2 border-r">View</th>
                                <th class="py-2 px-2 border-r">Pt</th>
                                <th class="py-2 px-2 border-r">Type</th>
                                <th class="py-2 px-2 border-r">Description</th>
                                <th class="py-2 px-2 border-r">Specified</th>
                                <th class="py-2 px-2 border-r">Qty</th>
                                <th class="py-2 px-2 border-r">Tolerance</th>
                                <th class="py-2 px-2 border-r">Tool</th>
                                <th class="py-2 px-1 border-r text-center">Box</th>
                                <th class="py-2 px-1 text-center">Del</th>
                            </tr>
                        </thead>
                        <tbody id="dataBody" class="bg-white divide-y"></tbody>
                    </table>
                </div>
                <div class="flex flex-wrap gap-2 flex-shrink-0">
                    <button id="addRowBtn" class="bg-gray-200 hover:bg-gray-300 text-gray-800 font-bold py-1 px-3 rounded text-sm">+ Add Row</button>
                    <button id="downloadBtn" class="bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-1 px-3 rounded text-sm">Download Excel</button>
                    <button id="submitFeedbackBtn" class="bg-purple-600 hover:bg-purple-700 text-white font-bold py-1 px-3 rounded text-sm ml-auto">Submit to BQ</button>
                </div>
            </div>
        </div>

        <!-- RIGHT PANEL: INTERACTIVE CANVAS -->
        <div class="right-panel">
            <div class="bg-white p-4 rounded shadow border-t-4 border-purple-500 flex flex-col h-full">
                <div class="flex justify-between items-center mb-2 flex-shrink-0">
                    <h2 class="text-lg font-bold">Blueprint View</h2>
                    <div class="flex gap-2">
                        <button id="resetZoomBtn" class="px-4 py-1 rounded text-blue-600 font-bold hover:bg-blue-100 bg-gray-100">Reset View</button>
                        <button id="prevBtn" class="px-3 py-1 bg-gray-200 rounded hover:bg-gray-300">&larr;</button>
                        <span id="imgCounter" class="text-gray-700 mt-1 font-bold text-sm">0 of 0</span>
                        <button id="nextBtn" class="px-3 py-1 bg-gray-200 rounded hover:bg-gray-300">&rarr;</button>
                    </div>
                </div>

                <div id="canvasWrapper" class="relative bg-gray-300 border border-gray-400 rounded shadow-inner flex-grow overflow-hidden">
                    <div id="emptyCanvasText" class="absolute inset-0 flex justify-center items-center text-gray-500 font-bold text-xl pointer-events-none">Upload drawings to start</div>
                    <canvas id="fabricCanvas"></canvas>
                </div>
                <!-- NEW FEEDBACK/REMARKS SECTION -->
                <div class="flex-shrink-0 bg-gray-50 p-3 border rounded shadow-sm mb-2">
                    <label class="block text-sm font-bold text-gray-700 mb-1">General Remarks & Feedback:</label>
                    <textarea id="feedbackText" rows="2" class="w-full border border-gray-300 rounded p-2 text-sm focus:ring focus:ring-purple-200 focus:outline-none" placeholder="Enter any general feedback, missing items, or comments about the AI's performance here..."></textarea>
                </div>
            </div>
        </div>
    </div>
</div>

<script>
// Safely prints log outputs dynamically to the screen
function logToScreen(msg) {
    console.log(msg);
    var logDiv = document.getElementById("statusLog");
    if(logDiv) {
        logDiv.innerHTML += "<div>> " + msg + "</div>";
        logDiv.scrollTop = logDiv.scrollHeight;
    }
}

window.onerror = function(message, source, lineno, colno, error) {
    var errBox = document.getElementById("errorBanner");
    errBox.style.display = "block";
    errBox.innerText = "CRITICAL JS ERROR: " + message + " (Line " + lineno + ")";
    logToScreen("ERROR: " + message);
};

window.onload = function() {
    logToScreen("DOM Loaded. Initializing Engine...");

    window.APP_SESSION = "SESS-" + Math.random().toString(36).substr(2, 9);
    document.getElementById('sessionIdLabel').innerText = "Session: " + window.APP_SESSION;

    window.APP_STATE = { images: [], currentIndex: 0, extractedData: [], categoryFilter: 'all' };
    window.CATEGORY_COLORS = {
        dimension: 'bg-indigo-100 text-indigo-800 border-indigo-300',
        component: 'bg-amber-100 text-amber-800 border-amber-300',
        title_block: 'bg-rose-100 text-rose-800 border-rose-300'
    };
    window.fabricScaleRatio = 1;
    window.imgOffsetX = 0;
    window.imgOffsetY = 0;

    // Canvas mapping to link Fabric rects to Row IDs
    window.canvasRects = {};
    window.activeDrawingRowId = null;

    var canvas = new fabric.Canvas('fabricCanvas', { selection: false });

    // View controls (FIXED: Identity matrix added)
    document.getElementById('resetZoomBtn').onclick = function() { canvas.setViewportTransform([1,0,0,1,0,0]); };

    canvas.on('mouse:wheel', function(opt) {
        var zoom = canvas.getZoom() * Math.pow(0.999, opt.e.deltaY);
        if (zoom > 20) zoom = 20; if (zoom < 0.5) zoom = 0.5;
        canvas.zoomToPoint({ x: opt.e.offsetX, y: opt.e.offsetY }, zoom);
        opt.e.preventDefault(); opt.e.stopPropagation();
    });

    // Drawing Logic for Row Bounding Boxes
    var isDrawingRect = false;
    var startX, startY, tempRect;

    canvas.on('mouse:down', function(opt) {
        if (opt.e.altKey) { window.isDraggingPan = true; window.lastPosX = opt.e.clientX; window.lastPosY = opt.e.clientY; return; }
        if (!window.activeDrawingRowId) return;

        isDrawingRect = true;
        var pointer = canvas.getPointer(opt.e); startX = pointer.x; startY = pointer.y;
        tempRect = new fabric.Rect({ left: startX, top: startY, width: 0, height: 0, fill: 'rgba(255,0,0,0.2)', stroke: 'red', strokeWidth: 2/canvas.getZoom(), selectable: true });
        canvas.add(tempRect);
    });

    canvas.on('mouse:move', function(opt) {
        if (window.isDraggingPan) {
            var vpt = canvas.viewportTransform;
            vpt[4] += opt.e.clientX - window.lastPosX;
            vpt[5] += opt.e.clientY - window.lastPosY;
            canvas.requestRenderAll(); window.lastPosX = opt.e.clientX; window.lastPosY = opt.e.clientY; return;
        }
        if (!isDrawingRect) return;
        var pointer = canvas.getPointer(opt.e);
        if (pointer.x < startX) tempRect.set({ left: pointer.x });
        if (pointer.y < startY) tempRect.set({ top: pointer.y });
        tempRect.set({ width: Math.abs(pointer.x - startX), height: Math.abs(pointer.y - startY) });
        canvas.renderAll();
    });

    canvas.on('mouse:up', function() {
        if (window.isDraggingPan) { canvas.setViewportTransform(canvas.viewportTransform); window.isDraggingPan = false; return; }
        if (!isDrawingRect) return;
        isDrawingRect = false;

        if (tempRect.width < 5 || tempRect.height < 5) { canvas.remove(tempRect); return; }

        var rowId = window.activeDrawingRowId;
        var rowData = window.APP_STATE.extractedData.find(function(r){ return r._id === rowId; });
        if(rowData) {
            rowData.bbox = [
                (tempRect.left - window.imgOffsetX) / window.fabricScaleRatio,
                (tempRect.top - window.imgOffsetY) / window.fabricScaleRatio,
                (tempRect.width * tempRect.scaleX) / window.fabricScaleRatio,
                (tempRect.height * tempRect.scaleY) / window.fabricScaleRatio
            ];
            logToScreen("Box drawn and linked to row: " + rowData.Dim_Type);
        }

        // --- NEW LINE ADDED HERE ---
        canvas.remove(tempRect); // Destroy the temporary red drawing box
        // ----------------------------

        window.activeDrawingRowId = null;
        document.getElementById('canvasWrapper').classList.remove('drawing-active');
        canvas.defaultCursor = 'default';
        window.renderCanvasBoxes();
        window.renderChecklistTable();
    });

    canvas.on('object:modified', function(e) {
        var obj = e.target;
        if(obj.rowId) {
            var rowData = window.APP_STATE.extractedData.find(function(r){ return r._id === obj.rowId; });
            if(rowData) {
                rowData.bbox = [
                    (obj.left - window.imgOffsetX) / window.fabricScaleRatio,
                    (obj.top - window.imgOffsetY) / window.fabricScaleRatio,
                    (obj.width * obj.scaleX) / window.fabricScaleRatio,
                    (obj.height * obj.scaleY) / window.fabricScaleRatio
                ];
            }
        }
    });

    window.updateRowData = function(id, field, value) {
        var row = window.APP_STATE.extractedData.find(function(r) { return r._id === id; });
        if(row) {
            if(field === 'is_correct') {
                row.is_correct = value;
                var tr = document.getElementById("tr_" + id);
                if(value) tr.classList.add('row-correct'); else tr.classList.remove('row-correct');
                window.renderCanvasBoxes();
            } else {
                row[field] = value;
            }
        }
    };

    window.activateDrawing = function(id) {
        window.activeDrawingRowId = id;
        document.getElementById('canvasWrapper').classList.add('drawing-active');
        canvas.defaultCursor = 'crosshair';
        logToScreen("Click and drag on the image to draw a bounding box for this item.");
    };

    window.deleteRow = function(id) {
        window.APP_STATE.extractedData = window.APP_STATE.extractedData.filter(function(r){ return r._id !== id; });
        window.renderChecklistTable();
        window.renderCanvasBoxes();
    };

    document.getElementById('addRowBtn').onclick = function() {
        if(window.APP_STATE.images.length === 0) return alert("Upload images first.");
        var currentImg = window.APP_STATE.images[window.APP_STATE.currentIndex].name;
        var newRow = {
            _id: "MANUAL-" + Math.random().toString(36).substr(2, 9),
            image_name: currentImg,
            Category: "dimension", Component_Name: "", Dwg_View: "",
            Pt_No: "", Dim_Type: "NEW", Dim_Description: "NEW DIM", Specified: "", Quantity: "", Tolerance: "", Measuring_Tools: "", bbox: [], is_correct: false
        };
        window.APP_STATE.extractedData.push(newRow);
        window.renderChecklistTable();
    };

    window.renderCategoryFilterBar = function(pageData) {
        var bar = document.getElementById('categoryFilterBar');
        if (!bar) return;
        var cats = ['all', 'dimension', 'component', 'title_block'];
        var counts = { all: pageData.length };
        cats.slice(1).forEach(function(c) { counts[c] = pageData.filter(function(r){ return r.Category === c; }).length; });

        bar.innerHTML = cats.map(function(c) {
            var active = window.APP_STATE.categoryFilter === c;
            var label = c === 'all' ? 'All' : c.replace('_', ' ');
            var activeClasses = active ? 'bg-blue-600 text-white border-blue-600' : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-100';
            return '<button class="px-2 py-1 rounded-full text-xs font-semibold border ' + activeClasses + '" onclick="window.setCategoryFilter(\\'' + c + '\\')">' + label + ' (' + counts[c] + ')</button>';
        }).join('');
    };

    window.setCategoryFilter = function(cat) {
        window.APP_STATE.categoryFilter = cat;
        window.renderChecklistTable();
    };

    window.renderChecklistTable = function() {
        var tbody = document.getElementById('dataBody'); tbody.innerHTML = '';
        if (window.APP_STATE.extractedData.length === 0) return;
        var currentImg = window.APP_STATE.images[window.APP_STATE.currentIndex].name;
        var pageData = window.APP_STATE.extractedData.filter(function(r) { return r.image_name === currentImg; });

        window.renderCategoryFilterBar(pageData);

        if (window.APP_STATE.categoryFilter !== 'all') {
            pageData = pageData.filter(function(r) { return r.Category === window.APP_STATE.categoryFilter; });
        }

        pageData.forEach(function(r) {
            var tr = document.createElement('tr');
            tr.id = "tr_" + r._id;
            tr.className = "hover:bg-gray-100 border-b " + (r.is_correct ? "row-correct" : "");

            var categoryOptions = ['dimension','component','title_block'];
            var badgeClass = window.CATEGORY_COLORS[r.Category] || 'bg-gray-100 text-gray-800 border-gray-300';
            var categorySelectHtml = '<select class="rounded-full border px-2 py-0.5 text-xs font-semibold ' + badgeClass + '" onchange="window.updateRowData(\\'' + r._id + '\\', \\'Category\\', this.value)">' +
                categoryOptions.map(function(opt) {
                    return '<option value="' + opt + '"' + (r.Category === opt ? ' selected' : '') + '>' + opt + '</option>';
                }).join('') +
                '</select>';

            var html = '';
            html += '<td class="py-1 px-1 border-r text-center"><input type="checkbox" style="width:16px; height:16px; cursor:pointer;" onchange="window.updateRowData(\\'' + r._id + '\\', \\'is_correct\\', this.checked)" ' + (r.is_correct ? 'checked' : '') + '></td>';
            html += '<td class="py-1 px-1 border-r">' + categorySelectHtml + '</td>';
            html += '<td class="py-1 px-1 border-r"><input type="text" value="' + (r.Component_Name || '') + '" onchange="window.updateRowData(\\'' + r._id + '\\', \\'Component_Name\\', this.value)"></td>';
            html += '<td class="py-1 px-1 border-r"><input type="text" class="font-mono text-xs" value="' + (r.Dwg_View || '') + '" onchange="window.updateRowData(\\'' + r._id + '\\', \\'Dwg_View\\', this.value)"></td>';
            html += '<td class="py-1 px-1 border-r"><input type="text" value="' + (r.Pt_No || '') + '" onchange="window.updateRowData(\\'' + r._id + '\\', \\'Pt_No\\', this.value)"></td>';
            html += '<td class="py-1 px-1 border-r"><input type="text" class="font-bold text-indigo-700" value="' + (r.Dim_Type || '') + '" onchange="window.updateRowData(\\'' + r._id + '\\', \\'Dim_Type\\', this.value)"></td>';
            html += '<td class="py-1 px-1 border-r"><input type="text" value="' + (r.Dim_Description || '') + '" onchange="window.updateRowData(\\'' + r._id + '\\', \\'Dim_Description\\', this.value)"></td>';
            html += '<td class="py-1 px-1 border-r"><input type="text" value="' + (r.Specified || '') + '" onchange="window.updateRowData(\\'' + r._id + '\\', \\'Specified\\', this.value)"></td>';
            html += '<td class="py-1 px-1 border-r"><input type="text" value="' + (r.Quantity || '') + '" onchange="window.updateRowData(\\'' + r._id + '\\', \\'Quantity\\', this.value)"></td>';
            html += '<td class="py-1 px-1 border-r"><input type="text" class="text-red-600 font-mono" value="' + (r.Tolerance || '') + '" onchange="window.updateRowData(\\'' + r._id + '\\', \\'Tolerance\\', this.value)"></td>';
            // <--- ADD THIS LINE FOR MEASURING TOOLS:
            html += '<td class="py-1 px-1 border-r"><input type="text" value="' + (r.Measuring_Tools || '') + '" onchange="window.updateRowData(\\'' + r._id + '\\', \\'Measuring_Tools\\', this.value)"></td>';
            var boxColor = (r.bbox && r.bbox.length === 4) ? "bg-green-100 text-green-800" : "bg-gray-200 text-gray-700";
            html += '<td class="py-1 px-1 border-r text-center"><button class="' + boxColor + ' px-2 py-1 rounded text-xs font-bold hover:bg-blue-200" onclick="window.activateDrawing(\\'' + r._id + '\\')">Box</button></td>';
            html += '<td class="py-1 px-1 text-center"><button class="text-red-500 hover:text-white hover:bg-red-500 px-2 py-1 rounded text-xs font-bold" onclick="window.deleteRow(\\'' + r._id + '\\')">X</button></td>';

            tr.innerHTML = html;
            tbody.appendChild(tr);
        });
    };

    window.renderCanvasBoxes = function() {
        Object.keys(window.canvasRects).forEach(function(key) {
            canvas.remove(window.canvasRects[key]);
            if(window.canvasRects[key].textObj) canvas.remove(window.canvasRects[key].textObj);
        });
        window.canvasRects = {};

        var currentImg = window.APP_STATE.images[window.APP_STATE.currentIndex].name;
        var pageData = window.APP_STATE.extractedData.filter(function(r) { return r.image_name === currentImg; });

        pageData.forEach(function(r) {
            if(r.bbox && r.bbox.length === 4) {
                var scaledX = (r.bbox[0] * window.fabricScaleRatio) + window.imgOffsetX;
                var scaledY = (r.bbox[1] * window.fabricScaleRatio) + window.imgOffsetY;
                var scaledW = r.bbox[2] * window.fabricScaleRatio;
                var scaledH = r.bbox[3] * window.fabricScaleRatio;

                var color = r.is_correct ? "green" : "blue";

                var rect = new fabric.Rect({
                    left: scaledX, top: scaledY, width: scaledW, height: scaledH,
                    fill: 'rgba(0,0,0,0)', stroke: color, strokeWidth: 2 / canvas.getZoom(),
                    borderColor: color, cornerColor: color, transparentCorners: false,
                    selectable: true, hasRotatingPoint: false
                });
                rect.rowId = r._id;

                var text = new fabric.Text(r.Dim_Type + (r.Pt_No ? " ("+r.Pt_No+")" : ""), {
                    left: scaledX, top: scaledY - (15/canvas.getZoom()), fontSize: 14/canvas.getZoom(), fill: 'white', backgroundColor: color, selectable: false
                });
                rect.textObj = text;

                rect.on('moving', function() { text.set({left: rect.left, top: rect.top - (15/canvas.getZoom())}); });
                rect.on('scaling', function() { text.set({left: rect.left, top: rect.top - (15/canvas.getZoom())}); });

                canvas.add(rect, text);
                window.canvasRects[r._id] = rect;
            }
        });
        // <--- ADD THIS LINE TO INSTANTLY CLEAR DELETED BOXES
        canvas.requestRenderAll();
    };

    window.renderImage = function() {
        if(window.APP_STATE.images.length === 0) return;
        var imgObj = window.APP_STATE.images[window.APP_STATE.currentIndex];
        document.getElementById('imgCounter').innerText = "Page " + (window.APP_STATE.currentIndex + 1) + " of " + window.APP_STATE.images.length;

        // (FIXED: Identity matrix added here as well)
        canvas.clear(); canvas.setViewportTransform([1,0,0,1,0,0]);

        fabric.Image.fromURL(imgObj.b64, function(img) {
            var w = document.getElementById('canvasWrapper'); canvas.setWidth(w.clientWidth); canvas.setHeight(w.clientHeight);
            var ratio = Math.min((w.clientWidth - 40)/img.width, (w.clientHeight - 40)/img.height, 1);
            window.fabricScaleRatio = ratio;
            img.set({ originX: 'center', originY: 'center', left: w.clientWidth/2, top: w.clientHeight/2, scaleX: ratio, scaleY: ratio });
            window.imgOffsetX = (w.clientWidth - (img.width * ratio)) / 2;
            window.imgOffsetY = (w.clientHeight - (img.height * ratio)) / 2;
            canvas.setBackgroundImage(img, canvas.renderAll.bind(canvas));

            window.renderChecklistTable();
            window.renderCanvasBoxes();
        });
    };

    document.getElementById('prevBtn').onclick = function() { if(window.APP_STATE.currentIndex > 0) { window.APP_STATE.currentIndex--; window.renderImage(); } };
    document.getElementById('nextBtn').onclick = function() { if(window.APP_STATE.currentIndex < window.APP_STATE.images.length - 1) { window.APP_STATE.currentIndex++; window.renderImage(); } };

    document.getElementById('uploadBtn').onclick = function() {
        var input = document.getElementById('fileInput');
        if(input.files.length === 0) return alert("Please select a file first.");

        document.getElementById('uploadBtn').innerText = "Processing..."; document.getElementById('emptyCanvasText').classList.add('hidden'); document.getElementById('uploadBtn').disabled = true;

        var promises = [];
        for(var i = 0; i < input.files.length; i++) {
            (function(file) {
                promises.push(new Promise(function(resolve, reject) {
                    var reader = new FileReader();
                    reader.onload = function(e) { resolve({ name: file.name, content_b64: e.target.result }); };
                    reader.onerror = function() { reject(new Error("Failed to read " + file.name)); };
                    reader.readAsDataURL(file);
                }));
            })(input.files[i]);
        }

        Promise.all(promises).then(function(results) {
            logToScreen("Sending file payload to backend...");
            return fetch('/api/upload', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ files: results }) });
        }).then(function(res) {
            if(!res.ok) throw new Error("HTTP Status " + res.status);
            return res.json();
        }).then(function(data) {
            if(data.error) throw new Error(data.error);
            window.APP_STATE.images = data.images;
            window.APP_STATE.currentIndex = 0;
            document.getElementById('extractBtn').classList.remove('hidden');
            logToScreen("Upload Success! System ready for extraction.");
            window.renderImage();
        }).catch(function(err) {
            logToScreen("UPLOAD FAILED: " + err.message); alert("Upload Failed: " + err.message);
        }).finally(function() {
            document.getElementById('uploadBtn').innerText = "Upload"; document.getElementById('uploadBtn').disabled = false;
        });
    };

    document.getElementById('extractBtn').onclick = function() {
        logToScreen("Initiating AI Extraction to BigQuery & LLM...");
        document.getElementById('extractBtn').innerText = "AI is thinking..."; document.getElementById('extractBtn').disabled = true;

        var payload = { session_id: window.APP_SESSION, images: window.APP_STATE.images.map(function(img) { return {name: img.name, b64: img.b64}; }) };

        fetch('/api/extract', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) })
        .then(function(res) { if(!res.ok) throw new Error("HTTP " + res.status); return res.json(); })
        .then(function(resp) {
            window.APP_STATE.extractedData = resp.data;
            document.getElementById('dataPanel').classList.remove('hidden');
            logToScreen("Extraction complete and logged to BigQuery.");
            window.renderChecklistTable();
            window.renderCanvasBoxes();
        }).catch(function(err) {
            logToScreen("EXTRACTION FAILED: " + err.message); alert("Extraction Failed: " + err.message);
        }).finally(function() {
            document.getElementById('extractBtn').innerText = "Run AI Extraction"; document.getElementById('extractBtn').disabled = false;
        });
    };

    document.getElementById('submitFeedbackBtn').onclick = function() {
        if(window.APP_STATE.extractedData.length === 0) return alert("Nothing to submit.");

        document.getElementById('submitFeedbackBtn').innerText = "Submitting...";

        // <--- ADD THIS TO GRAB THE TEXT
        var remarks = "";
        var txtElem = document.getElementById('feedbackText');
        if(txtElem) remarks = txtElem.value;

        var payload = {
            session_id: window.APP_SESSION,
            image_name: window.APP_STATE.images[window.APP_STATE.currentIndex].name,
            feedback_remarks: remarks, // <--- ADD THIS TO PAYLOAD
            corrected_data: window.APP_STATE.extractedData
        };

        fetch('/api/feedback', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) })
        .then(function(res) { if(!res.ok) throw new Error("HTTP " + res.status); return res.json(); })
        .then(function() {
            alert("Feedback saved directly to BigQuery!");
            logToScreen("Human Corrected Checklist logged to BQ successfully.");
            // <--- ADD THIS TO CLEAR TEXT BOX ON SUCCESS
            if(document.getElementById('feedbackText')) document.getElementById('feedbackText').value = "";
        }).catch(function(err) {
            alert("Feedback failed: " + err.message);
        }).finally(function() {
            document.getElementById('submitFeedbackBtn').innerText = "Submit to BQ";
        });
    };

    document.getElementById('downloadBtn').onclick = function() {
        if(window.APP_STATE.extractedData.length === 0) return alert("Nothing to download.");
        logToScreen("Generating clean Excel file...");
        fetch('/api/download_excel', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(window.APP_STATE.extractedData) })
        .then(function(res) { if(!res.ok) throw new Error("Failed"); return res.blob(); })
        .then(function(blob) {
            var a = document.createElement('a'); a.href = window.URL.createObjectURL(blob); a.download = 'TGS_Checklist_Corrected.xlsx'; a.click();
        }).catch(function(err) { alert("Download error: " + err.message); });
    };

    logToScreen("Application successfully loaded.");
};
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    return HTMLResponse(content=HTML_CONTENT, headers={"Cache-Control": "no-cache"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)
