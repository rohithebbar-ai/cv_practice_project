import io
import base64
import json
import re
import logging
import uuid
import datetime
import os
from typing import List, Dict, Any
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time
import pypdfium2 as pdfium
from PIL import Image
import pandas as pd
try:
    import pytesseract
    OCR_AVAILABLE = True
    # pip install only gets the Python wrapper - it still needs to find the
    # actual tesseract.exe binary. Auto-point at the default Windows install
    # location if present and not already on PATH.
    _default_win_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.name == "nt" and os.path.exists(_default_win_tesseract):
        pytesseract.pytesseract.tesseract_cmd = _default_win_tesseract
except ImportError:
    OCR_AVAILABLE = False
from difflib import SequenceMatcher

from google.oauth2 import service_account
from google.auth.transport.requests import AuthorizedSession

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

DEPLOYMENT = "gpt-5.2"

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
        llm_output = parsed_json["choices"][0]["message"]["content"]
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
# 3.5 OCR-BASED PRECISE MARKER MATCHING
# ========================================================
# The LLM's Dwg_View grid reference (e.g. "G-11") is reliable for WHICH cell
# an item belongs to, but placing multiple items inside that cell still needs
# a packed layout, which is not the item's real position. This section OCRs
# the page once, then fuzzy-matches each row's own text against OCR'd words
# constrained to that row's grid cell, to recover a real pixel bbox where
# possible. No LLM coordinate guessing is involved anywhere in this section.

GRID_ROW_ORDER = ['H', 'G', 'F', 'E', 'D', 'C', 'B', 'A']  # top-to-bottom, matches frontend
GRID_COLS = 12  # columns numbered 12..1 left-to-right, matches frontend

def parse_grid_ref(dwg_view: str):
    if not dwg_view:
        return None
    m = re.search(r'([A-Ha-h])\s*-?\s*(\d{1,2})', dwg_view)
    if not m:
        return None
    row_letter = m.group(1).upper()
    if row_letter not in GRID_ROW_ORDER:
        return None
    try:
        col_num = int(m.group(2))
    except ValueError:
        return None
    row_idx = GRID_ROW_ORDER.index(row_letter)
    col_idx = GRID_COLS - col_num
    if col_idx < 0 or col_idx >= GRID_COLS:
        return None
    return row_idx, col_idx

def cell_pixel_rect(dwg_view: str, img_w: int, img_h: int, pad_ratio: float = 0.15):
    """Returns (x0, y0, x1, y1) in image pixel space for the cell, padded slightly
    since the grid itself is an even-spacing approximation, not a measured one."""
    ref = parse_grid_ref(dwg_view)
    if not ref or not img_w or not img_h:
        return None
    row_idx, col_idx = ref
    col_step = img_w / GRID_COLS
    row_step = img_h / len(GRID_ROW_ORDER)
    left = col_idx * col_step
    top = row_idx * row_step
    pad_x = col_step * pad_ratio
    pad_y = row_step * pad_ratio
    return (
        max(0, left - pad_x), max(0, top - pad_y),
        min(img_w, left + col_step + pad_x), min(img_h, top + row_step + pad_y)
    )

def _ocr_single_orientation(image_bytes: bytes, angle: int):
    """angle: 0, 90 (CCW), or 270 (CW). Returns word dicts with bbox already
    transformed back into ORIGINAL image pixel coordinates. The inverse
    rotation formulas below were verified empirically against PIL's actual
    transpose() output before use, not just derived on paper."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        orig_w, orig_h = img.size
        if angle == 90:
            rot = img.transpose(Image.Transpose.ROTATE_90)
        elif angle == 270:
            rot = img.transpose(Image.Transpose.ROTATE_270)
        else:
            rot = img
        data = pytesseract.image_to_data(rot, output_type=pytesseract.Output.DICT)
    except Exception as e:
        logger.warning(f"OCR pass (angle={angle}) skipped: {e}")
        return []

    words = []
    n = len(data.get("text", []))
    for i in range(n):
        text = (data["text"][i] or "").strip()
        try:
            conf = float(data["conf"][i])
        except (ValueError, TypeError):
            conf = -1
        if not text or conf < 30:
            continue
        left, top, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]

        if angle == 90:
            x0 = orig_w - 1 - (top + h)
            y0 = left
            box_w, box_h = h, w
        elif angle == 270:
            x0 = top
            y0 = orig_h - 1 - (left + w)
            box_w, box_h = h, w
        else:
            x0, y0, box_w, box_h = left, top, w, h

        words.append({
            "text": text, "left": x0, "top": y0, "width": box_w, "height": box_h,
            "line_key": (angle, data["block_num"][i], data["par_num"][i], data["line_num"][i]),
            "word_num": data["word_num"][i],
        })
    return words

def ocr_tokens(image_bytes: bytes):
    """Runs Tesseract at 0/90/270 degrees to also catch vertically-printed
    dimension text (very common on GA drawings for height/vertical callouts -
    a single horizontal-only OCR pass silently misses all of these), then
    merges results (already in original-image coordinates), plus adjacent-
    word-pair tokens per orientation to catch two-word values like 'ISMC 100'."""
    if not OCR_AVAILABLE:
        return []

    all_words = []
    for angle in (0, 90, 270):
        all_words.extend(_ocr_single_orientation(image_bytes, angle))

    tokens = [{"text": w["text"], "bbox": (w["left"], w["top"], w["width"], w["height"])} for w in all_words]

    words_sorted = sorted(all_words, key=lambda w: (w["line_key"], w["word_num"]))
    for i in range(len(words_sorted) - 1):
        a, b = words_sorted[i], words_sorted[i + 1]
        if a["line_key"] == b["line_key"] and (b["word_num"] - a["word_num"]) == 1:
            left = min(a["left"], b["left"])
            top = min(a["top"], b["top"])
            right = max(a["left"] + a["width"], b["left"] + b["width"])
            bottom = max(a["top"] + a["height"], b["top"] + b["height"])
            tokens.append({"text": a["text"] + " " + b["text"], "bbox": (left, top, right - left, bottom - top)})

    return tokens

def find_best_match(candidate_text: str, tokens, cell_rect, min_ratio: float = 0.6):
    if not candidate_text or not tokens:
        return None
    norm_candidate = re.sub(r'\s+', ' ', candidate_text.strip().upper())
    if not norm_candidate:
        return None

    # Short numeric values (e.g. "150") are unforgiving to fuzzy-match ratios -
    # a single misread digit costs proportionally more than in a longer string.
    is_short_numeric = bool(re.fullmatch(r'\d{1,4}', norm_candidate))
    effective_min_ratio = 0.5 if is_short_numeric else min_ratio

    best_ratio, best_bbox = 0.0, None
    for tok in tokens:
        tx, ty, tw, th = tok["bbox"]
        if cell_rect:
            cx, cy = tx + tw / 2, ty + th / 2
            x0, y0, x1, y1 = cell_rect
            if not (x0 <= cx <= x1 and y0 <= cy <= y1):
                continue
        norm_tok = re.sub(r'\s+', ' ', tok["text"].strip().upper())
        ratio = SequenceMatcher(None, norm_candidate, norm_tok).ratio()
        if ratio > best_ratio:
            best_ratio, best_bbox = ratio, tok["bbox"]

    if best_ratio >= effective_min_ratio:
        return [int(best_bbox[0]), int(best_bbox[1]), int(best_bbox[2]), int(best_bbox[3])]
    return None

# ========================================================
# 3.6 HYBRID SECOND PASS: TARGETED LLM GROUNDING FOR OCR'S LEFTOVERS
# ========================================================
# OCR is fast and free, but struggles on small, dense, cluttered dimension
# text. Rather than re-running a vision call on every extracted item (slow,
# expensive), this only fires for items OCR left unmatched, grouped by grid
# cell so it's one call per affected cell, not one call per leftover item.
# The crop is upscaled so small text becomes large and legible - the same
# reasoning as tiled OCR, but using the LLM's vision instead of Tesseract for
# exactly the cases Tesseract already failed on.
#
# IDENTITY / RELATIONSHIP SAFETY: every row's Category, Component_Name,
# View_Label, and Dwg_View come ONLY from Stage 1 and are never touched here.
# This stage only ever answers "where", never "what". Every item is tracked
# by its own stable UUID (item["_id"], assigned once when Stage 1's JSON is
# first parsed) all the way through - including across recursive splits - so
# a returned position can only ever be written back onto the exact row it
# came from. The model itself never sees or needs to preserve that UUID; it
# only echoes back small local integers, which call_llm_for_grounding
# translates back to the real UUID in Python before anything is merged.
#
# POSITIONING METHOD: rather than asking the model to estimate continuous
# pixel coordinates (a much harder ask it tends to decline or guess badly
# on), each crop is treated as its own small grid, and the model is asked to
# point to a (row, col) cell within that grid - the same "point to grid,
# don't estimate pixels" principle used at Stage 1, just applied at a finer
# resolution inside a zoomed-in crop. This is a simpler, more reliably
# answerable question, so more items end up with a usable position instead
# of being silently skipped.

GROUNDING_CROP_SCALE = 3
GROUNDING_UPSCALED_MAX_DIM = 1600
GROUNDING_MAX_ITEMS_PER_CALL = 12       # trigger for splitting, not a hard drop-the-rest cap
GROUNDING_MAX_RECURSION_DEPTH = 3       # a region needing to split more than this just stays approximate
GROUNDING_REGION_OVERLAP = 0.08         # slight overlap between split halves so boundary items aren't missed by both
GROUNDING_SAFETY_MAX_CALLS = 300        # runaway-loop guard only - should not bind in normal use, unlike a real business cap
GROUNDING_SUBGRID_SIZE = 4              # each crop is divided into this many rows/cols for the model to point into

def crop_cell_for_grounding(image_bytes: bytes, cell_rect):
    """Crops the given (x0,y0,x1,y1) region out of the original image and
    upscales it. Returns (crop_jpeg_bytes, (offset_x, offset_y), scale_factor).
    The offset/scale are kept for reference, but the sub-grid positioning
    below computes directly from cell_rect instead of needing them."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x0, y0, x1, y1 = cell_rect
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(img.width, int(x1)), min(img.height, int(y1))
    if x1 <= x0 or y1 <= y0:
        return None, None, None
    crop = img.crop((x0, y0, x1, y1))

    target_w = max(min(int(crop.width * GROUNDING_CROP_SCALE), GROUNDING_UPSCALED_MAX_DIM), 1)
    scale = target_w / crop.width
    target_h = max(int(crop.height * scale), 1)
    upscaled = crop.resize((target_w, target_h), Image.LANCZOS)

    buf = io.BytesIO()
    upscaled.save(buf, format="JPEG", quality=90)
    return buf.getvalue(), (x0, y0), scale

def split_rect_longer_axis(rect, overlap: float = GROUNDING_REGION_OVERLAP):
    """Splits a region in half along whichever axis is longer, with a small
    overlap so an item sitting right on the split line isn't missed by both halves."""
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    if w >= h:
        mid = x0 + w / 2
        pad = w * overlap
        return (x0, y0, min(x1, mid + pad), y1), (max(x0, mid - pad), y0, x1, y1)
    else:
        mid = y0 + h / 2
        pad = h * overlap
        return (x0, y0, x1, min(y1, mid + pad)), (x0, max(y0, mid - pad), x1, y1)

def build_grounding_prompt(local_items):
    """local_items: list of {"local_id": int, "text": str} - the model only
    ever sees small clean integers here, never the real UUID, since asking a
    model to transcribe a long UUID verbatim invites transcription errors."""
    lines = [
        "You are given a small cropped region of a larger engineering drawing.",
        f"This crop is divided into a {GROUNDING_SUBGRID_SIZE}x{GROUNDING_SUBGRID_SIZE} grid: rows 1 to {GROUNDING_SUBGRID_SIZE} top to bottom, columns 1 to {GROUNDING_SUBGRID_SIZE} left to right.",
        "Below is a numbered list of values/labels known to be visible somewhere in this crop.",
        "For each one, point to which grid cell (row, column) of THIS CROP it falls in.",
        "Do NOT output pixel coordinates, x, y, width, height, or any kind of",
        "bounding box in any form. The ONLY location information allowed is the",
        "row number and column number of the grid cell - two small integers,",
        "nothing else. If you find yourself estimating a pixel position, stop",
        "and instead just pick the single grid cell that position falls within.",
        "",
        "Items:"
    ]
    for c in local_items:
        lines.append(f'{c["local_id"]}. "{c["text"]}"')
    lines.append("")
    lines.append("OUTPUT FORMAT:")
    lines.append("Return ONLY a valid JSON array, no markdown, no explanation.")
    lines.append(f'Each object: {{"id": <the number above>, "row": <1 to {GROUNDING_SUBGRID_SIZE}>, "col": <1 to {GROUNDING_SUBGRID_SIZE}>}}')
    lines.append("If you cannot confidently locate an item in this crop, omit it entirely from the array - do not guess.")
    return "\n".join(lines)

def call_llm_for_grounding(crop_bytes: bytes, items):
    """items: list of {"id": <stable UUID string>, "text": str}.
    Returns list of {"id": <the SAME UUID string>, "row": int, "col": int} -
    the UUID round-trips through a local integer translation internally so
    the model never has to handle it directly."""
    if not items or not crop_bytes:
        return []
    local_items = [{"local_id": i, "text": it["text"]} for i, it in enumerate(items)]
    prompt_text = build_grounding_prompt(local_items)
    raw = call_llm(prompt_text, [{"filename": "crop.jpg", "bytes": crop_bytes}])
    parsed = extract_json_array(raw)
    if not isinstance(parsed, list):
        return []

    results = []
    for res in parsed:
        try:
            local_id = int(res.get("id"))
            row = int(res.get("row"))
            col = int(res.get("col"))
        except (TypeError, ValueError, AttributeError):
            continue
        if 0 <= local_id < len(items) and 1 <= row <= GROUNDING_SUBGRID_SIZE and 1 <= col <= GROUNDING_SUBGRID_SIZE:
            results.append({"id": items[local_id]["id"], "row": row, "col": col})
    return results

def ground_items_recursive(image_bytes: bytes, items, rect, depth: int, call_state: dict):
    """items: list of {"id": <stable UUID string>, "text": str}.
    Returns dict {uuid: {"rect": [x0,y0,x1,y1], "row": r, "col": c, "grid_size": N}}
    - a plain grid pointer, not a bbox. No pixel coordinate is ever fabricated
    here; "rect" is just the region that was asked about (needed so the
    frontend can compute an on-screen position from the pointer at render
    time), and row/col are exactly what the model answered. If a region has
    too many items for one call, splits the CROP (not the item list) and
    recurses - each half is asked about the full remaining item list, since
    we don't know in advance which half an item is in."""
    if not items:
        return {}
    if call_state["calls"] >= GROUNDING_SAFETY_MAX_CALLS:
        if not call_state.get("warned"):
            logger.warning(f"Grounding safety cap hit ({GROUNDING_SAFETY_MAX_CALLS} calls this request) - remaining items stay approximate.")
            call_state["warned"] = True
        return {}

    if len(items) > GROUNDING_MAX_ITEMS_PER_CALL and depth < GROUNDING_MAX_RECURSION_DEPTH:
        rect_a, rect_b = split_rect_longer_axis(rect)
        found = ground_items_recursive(image_bytes, items, rect_a, depth + 1, call_state)
        remaining = [it for it in items if it["id"] not in found]
        found.update(ground_items_recursive(image_bytes, remaining, rect_b, depth + 1, call_state))
        return found

    crop_bytes, offset, scale = crop_cell_for_grounding(image_bytes, rect)
    if not crop_bytes:
        return {}

    call_state["calls"] += 1
    try:
        raw_results = call_llm_for_grounding(crop_bytes, items)
    except Exception as e:
        logger.warning(f"Grounding call failed at depth {depth}: {e}")
        return {}

    x0, y0, x1, y1 = rect
    found = {}
    valid_ids = {it["id"] for it in items}
    for res in raw_results:
        rid, row, col = res.get("id"), res.get("row"), res.get("col")
        if rid not in valid_ids or row is None or col is None:
            continue
        found[rid] = {"rect": [x0, y0, x1, y1], "row": row, "col": col, "grid_size": GROUNDING_SUBGRID_SIZE}
    return found

# ========================================================
# 3.7 LEGEND-FIRST EXTRACTION (relationship consistency)
# ========================================================
# Many engineering drawings include a legend/symbol table mapping a symbol or
# abbreviation to its meaning. If the main extraction pass has to interpret
# each occurrence of that symbol independently across many views, it can
# silently name the same physical thing slightly differently in different
# places. Extracting the legend first, once, and feeding that fixed mapping
# back into the main extraction prompt as known context removes that drift -
# every occurrence gets resolved against the same answer key instead of being
# re-guessed. This is deliberately generic (no fixed legend title or symbol
# set assumed), since not every drawing has a legend, and the ones that do
# use varying titles and conventions depending on discipline.

def build_legend_extraction_prompt() -> str:
    return """You are looking at one page of an engineering drawing. Many engineering
drawings include a legend, symbol table, or key somewhere on the sheet that
maps a symbol, abbreviation, or shorthand code to its full meaning (for
example, a table titled "LEGEND", "LEGEND FOR CONVEYORS", "SYMBOLS",
"ABBREVIATIONS", "KEY", or similar - the exact title and format varies by
drawing and discipline, so look for the pattern, not a specific title).

If such a legend/symbol table exists anywhere on this sheet, extract every
row from it. If no such table exists on this sheet, return an empty array -
do not invent one.

OUTPUT FORMAT:
Return ONLY a valid JSON array, no markdown, no explanation.
Each object: {"symbol": "<the symbol/abbreviation/shorthand as printed>", "meaning": "<its full meaning as printed>"}
"""

def extract_legend(image_bytes: bytes, filename: str):
    try:
        prompt_text = build_legend_extraction_prompt() + f"\nFilename: {filename}\n"
        raw = call_llm(prompt_text, [{"filename": filename, "bytes": image_bytes}])
        parsed = extract_json_array(raw)
    except Exception as e:
        logger.warning(f"Legend extraction skipped: {e}")
        return []
    if not isinstance(parsed, list):
        return []
    legend = []
    for row in parsed:
        if not isinstance(row, dict):
            continue
        sym = str(row.get("symbol", "")).strip()
        meaning = str(row.get("meaning", "")).strip()
        if sym and meaning:
            legend.append({"symbol": sym, "meaning": meaning})
    return legend

def build_legend_context_block(legend_entries) -> str:
    if not legend_entries:
        return ""
    lines = [
        "=== KNOWN SYMBOL/ABBREVIATION LEGEND FOR THIS SHEET ===",
        "The following symbol-to-meaning mappings were already identified on",
        "this sheet. Use these exact meanings consistently whenever you",
        "encounter the corresponding symbol elsewhere in the drawing, instead",
        "of re-interpreting or renaming them differently in different views:",
        ""
    ]
    for entry in legend_entries:
        lines.append(f'{entry["symbol"]} = {entry["meaning"]}')
    lines.append("")
    return "\n".join(lines)

# ========================================================
# 3.8 CHAIN-OF-THOUGHT SURVEY (completeness, before extraction)
# ========================================================
# Asking the model to first narrate what it sees, region by region, before
# committing to structured output measurably reduces silently-skipped
# content on dense pages - it cannot skip a region it was required to
# describe first. This is deliberately generic (no drawing-type-specific
# vocabulary), so it applies equally to any engineering drawing, not just
# this conveyor/steel drawing set.

BASE_SYSTEM_INSTRUCTION = """You are an expert QA Engineering AI for TATA STEEL GROWTH SHOP (TGS). Extract a comprehensive inspection checklist from engineering drawings.

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

=== 0.5 VIEW / DETAIL LABELING (mandatory for every row) ===
This sheet is made up of several separately-titled views and details (e.g.
"TYP. GALLERY CROSS SECTION", "DETAIL # D3", "VIEW - AA", "VIEW 1-1"), each
printed with its own underlined title on the drawing. For EVERY row, read
which titled view/detail it visually belongs to and populate "View_Label"
with that title exactly as printed. This groups rows by the drawing's own
named sub-drawings, not just by raw grid coordinate.

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
7. Dwg_View: the grid reference per Section 0 above (e.g. "G-11"). View_Label: the titled view/detail per Section 0.5 above.
8. Measuring_Tools MUST be chosen from this fixed list only, based on Dim_Type and the dimension's magnitude - use the exact same wording every time for the same condition, never invent alternate phrasing (e.g. never write "Measuring Tape" as a synonym for "Steel Tape" - they must not be mixed):
   - "Steel Tape" for Dim_Type=LD when Specified >= 500 (large lengths/heights/widths)
   - "Vernier Caliper" for Dim_Type=LD when Specified < 500, and for Dim_Type=CH (chamfers) and Dim_Type=DH (hole diameters)
   - "Radius Gauge" for Dim_Type=R
   - "Vernier Caliper" for Dim_Type=KS (keyways/slots)
   - "Feeler Gauge" for Dim_Type=GD (GD&T/general tolerance checks)
   - "Vernier Caliper" for Dim_Type=INT or EXT when Specified < 500, otherwise "Steel Tape"
For dimension rows, Category="dimension", Component_Name="", Quantity="".

=== 2. COMPONENTS, MEMBER CALLOUTS & REFERENCE LABELS (Category="component") ===
This category covers every labeled thing on the drawing that is not a
dimension value and not a title block field. It has three sub-types, all
using Category="component":
a) Named physical equipment/components (e.g. hopper, pulley, idler, conveyor,
   walkway, monorail, sizer, scraper, feeder, magnetic separator, cable tray,
   safety guard, tramp iron chute).
b) Structural member/section callouts - any label specifying a steel section
   attached to a drawn member (e.g. "ISA 65x65x6", "ISA 50x50x6", "ISMC 100",
   "ISMB 250"). Extract every one visible, not just the first.
c) Centerline and reference labels (e.g. "OF GALLERY" / "CL OF GALLERY",
   "OF CONVEYOR", "IDLER FXG CRS", "SHORT POST CRS", "T.O.S.", "F.G.L.").
Do not skip (b) and (c) just because they look like annotations rather than
named equipment - if it is printed text labeling a feature on the drawing
and it is not a dimension number and not in the title block, it belongs here.
For every row in this category:
- Category="component"
- Component_Name=the label exactly as printed (e.g. "ISA 65x65x6", "IDLER FXG CRS", "Truck Dump Hopper")
- Dwg_View=the grid reference per Section 0 above
- View_Label=the titled view/detail per Section 0.5 above
- Dim_Type="NA"
- Quantity=populate only if the drawing itself explicitly shows a count next to this item (e.g. a legend symbol marked "28 NOS."); otherwise leave "".
- Dimension-specific fields (Tolerance, Measuring_Tools, etc.) stay "" or "N/A".

=== 3. TITLE BLOCK (Category="title_block") ===
Locate the title block, typically bottom-right, containing "TATA STEEL LIMITED".
Its exact position and size vary by sheet, so search for it rather than assuming
a fixed location. Extract every labeled field inside it as its own row:
- Category="title_block"
- Component_Name=the plant/location line (e.g. "RM Location - West Bokaro Coal Mine")
- Dwg_View=the grid reference per Section 0 above - read the actual row letter and column number printed nearest the title block on this sheet; do not assume which letter/number that will be, since it varies by sheet layout
- View_Label="Title Block"
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
After your <<<SURVEY>>>...<<<END_SURVEY>>> block, output the marker <<<RESULT>>>
followed by ONLY a valid JSON array of objects and nothing else. Do not write
markdown, do not write explanations after the JSON.
Each object MUST have the following keys exactly:
"image_name", "Pt_No", "Dwg_View", "View_Label", "Dim_Type", "Dim_Description", "Specified", "Tolerance", "Measuring_Tools", "MC_No", "Insp_Type", "Category", "Component_Name", "Quantity"
"Dwg_View" and "View_Label" are mandatory on every row per Sections 0 and 0.5 above - they are the only locators needed, do not attempt to estimate pixel coordinates.
"""

SURVEY_INSTRUCTIONS = """=== STEP 1: SURVEY THE DRAWING BEFORE EXTRACTING (do this first, in your own words) ===
Before producing any structured output, write a brief plain-language survey
of the ENTIRE sheet, region by region. Go through every titled view or detail
on the sheet one at a time (in any order) and describe, in a sentence or two
each, what that region contains - e.g. what dimensions, components, labeled
parts, tables, or symbol legends are visible in it. Do this for every region
on the sheet, including small or easy-to-miss ones (corner details, small
tables, symbol keys, revision blocks, notes). Do not skip a region just
because it looks minor - actually look at each one, do not skip ahead.
Wrap this survey between the markers <<<SURVEY>>> and <<<END_SURVEY>>>.

After the survey, extract the structured data as instructed below, covering
everything you just surveyed. Output the final JSON array after the marker
<<<RESULT>>> and put nothing else after that marker except the JSON array
itself.

"""

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

    # Each page gets its own fully independent semantic extraction (legend
    # pass, survey, main extraction call) because the gateway only reliably
    # handles one image per call - looping here, rather than sending only
    # the first page, is what makes multi-page PDFs actually work.
    gcs_uris = []
    image_files_for_llm = []
    parsed_data = []
    raw_responses_by_page = {}

    for img in payload.images:
        header_split = img.b64.split(',', 1)
        encoded = header_split[1] if len(header_split) == 2 else header_split[0]
        image_bytes = base64.b64decode(encoded)

        image_files_for_llm.append({"filename": img.name, "bytes": image_bytes})
        uri = upload_image_to_gcs(img.b64, f"{session_id}_{img.name}")
        gcs_uris.append({"image_name": img.name, "gcs_uri": uri})

        # Legend-first pass, scoped to THIS page only - a multi-page PDF's
        # pages can each have their own legend, or none at all.
        try:
            legend_entries = extract_legend(image_bytes, img.name)
        except Exception as e:
            logger.warning(f"Legend pre-pass skipped for {img.name}: {e}")
            legend_entries = []

        system_instruction = SURVEY_INSTRUCTIONS + build_legend_context_block(legend_entries) + BASE_SYSTEM_INSTRUCTION
        prompt_text = system_instruction + "\n\n=== CURRENT BLUEPRINT TO ANALYZE ===\n" + f"\nFilename: {img.name}\n"

        raw_response = call_llm(prompt_text, [{"filename": img.name, "bytes": image_bytes}])
        raw_responses_by_page[img.name] = raw_response

        raw_response_for_parsing = raw_response.split("<<<RESULT>>>", 1)[1] if "<<<RESULT>>>" in raw_response else raw_response
        page_parsed = extract_json_array(raw_response_for_parsing)

        if not page_parsed:
            page_parsed = [{
                "image_name": img.name,
                "Pt_No": 1,
                "Dwg_View": "FAIL",
                "View_Label": "",
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
                "matched_bbox": None,
                "grid_pointer": None
            }]

        # Authoritatively set image_name ourselves - we know exactly which
        # page this call was scoped to, so there's no need to trust (or risk
        # a mistake from) the model's own self-reported filename.
        for item in page_parsed:
            item["image_name"] = img.name

        parsed_data.extend(page_parsed)

    for item in parsed_data:
        item["_id"] = str(uuid.uuid4())

    # OCR-based precise marker positions (best-effort; frontend falls back to
    # its packed grid-cell layout when matched_bbox is null)
    image_bytes_by_name = {f["filename"]: f["bytes"] for f in image_files_for_llm}
    ocr_cache = {}
    ocr_matched_count = 0
    ocr_tokens_found_any = False

    for item in parsed_data:
        img_name = item.get("image_name")
        img_bytes = image_bytes_by_name.get(img_name)
        item["matched_bbox"] = None
        item["grid_pointer"] = None
        if not img_bytes:
            continue

        if img_name not in ocr_cache:
            try:
                with Image.open(io.BytesIO(img_bytes)) as probe:
                    iw, ih = probe.size
            except Exception:
                iw, ih = None, None
            tokens = ocr_tokens(img_bytes)
            if tokens:
                ocr_tokens_found_any = True
            ocr_cache[img_name] = {"tokens": tokens, "w": iw, "h": ih}

        cache = ocr_cache[img_name]
        if not cache["w"] or not cache["h"]:
            continue

        candidate = item.get("Component_Name") or item.get("Specified") or item.get("Dim_Description") or ""
        rect = cell_pixel_rect(item.get("Dwg_View", ""), cache["w"], cache["h"])
        item["matched_bbox"] = find_best_match(candidate, cache["tokens"], rect)
        if item["matched_bbox"]:
            ocr_matched_count += 1

    if OCR_AVAILABLE and not ocr_tokens_found_any:
        logger.warning("OCR produced zero tokens across all images - Tesseract binary is likely not installed/reachable, even though pytesseract imported successfully.")

    # Hybrid second pass: targeted LLM grounding for exactly what OCR missed.
    # One call per affected grid cell (more if that cell needs to split), not
    # a fixed number of calls per image - density decides the call count.
    llm_grounding_matched_count = 0
    grounding_call_state = {"calls": 0}

    for img_name, cache in ocr_cache.items():
        img_bytes = image_bytes_by_name.get(img_name)
        if not img_bytes or not cache["w"] or not cache["h"]:
            continue

        leftover = [it for it in parsed_data if it.get("image_name") == img_name and not it.get("matched_bbox")]
        if not leftover:
            continue

        cell_groups = {}
        for it in leftover:
            ref = parse_grid_ref(it.get("Dwg_View", ""))
            if not ref:
                continue
            cell_groups.setdefault(ref, []).append(it)

        for ref, items in cell_groups.items():
            items_by_id = {}
            candidate_items = []
            for it in items:
                text = it.get("Component_Name") or it.get("Specified") or it.get("Dim_Description") or ""
                if text:
                    candidate_items.append({"id": it["_id"], "text": text})
                    items_by_id[it["_id"]] = it
            if not candidate_items:
                continue

            row_idx, col_idx = ref
            col_step = cache["w"] / GRID_COLS
            row_step = cache["h"] / len(GRID_ROW_ORDER)
            left = col_idx * col_step
            top = row_idx * row_step
            pad_x, pad_y = col_step * 0.25, row_step * 0.25
            rect = (
                max(0, left - pad_x), max(0, top - pad_y),
                min(cache["w"], left + col_step + pad_x), min(cache["h"], top + row_step + pad_y)
            )

            found = ground_items_recursive(img_bytes, candidate_items, rect, 0, grounding_call_state)
            for gid, pointer in found.items():
                target = items_by_id.get(gid)
                if not target:
                    continue
                target["grid_pointer"] = pointer
                target["match_source"] = "llm_grid_pointer"
                llm_grounding_matched_count += 1

    llm_grounding_calls_made = grounding_call_state["calls"]

    for item in parsed_data:
        if item.get("matched_bbox") and "match_source" not in item:
            item["match_source"] = "ocr"

    # Prepare sanitized log (text prompt only, no raw image bytes)
    log_row = {
        "session_id": session_id,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_used": DEPLOYMENT,
        "pages_processed": json.dumps(list(raw_responses_by_page.keys())),
        "images": json.dumps(gcs_uris),
        "raw_llm_response": json.dumps(raw_responses_by_page),
        "parsed_json": json.dumps(parsed_data)
    }
    log_to_bigquery(BQ_EXTRACTION_TABLE, [log_row])

    return {
        "data": parsed_data,
        "ocr_status": {
            "library_available": OCR_AVAILABLE,
            "tokens_found": ocr_tokens_found_any,
            "ocr_matched_count": ocr_matched_count,
            "llm_grounding_matched_count": llm_grounding_matched_count,
            "llm_grounding_calls": llm_grounding_calls_made,
            "total_count": len(parsed_data)
        }
    }

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
            "View": item.get("Dwg_View", ""), "View Label": item.get("View_Label", ""),
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
        .marker-flash { background-color: #fef08a !important; transition: background-color 0.3s ease; }
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
                        <button id="zoomOutBtn" class="px-3 py-1 rounded text-gray-700 font-bold hover:bg-gray-200 bg-gray-100">-</button>
                        <button id="zoomInBtn" class="px-3 py-1 rounded text-gray-700 font-bold hover:bg-gray-200 bg-gray-100">+</button>
                        <span class="inline-grid grid-cols-3 gap-0.5" style="width:76px;">
                            <span></span>
                            <button class="px-1 py-0.5 rounded text-gray-700 font-bold hover:bg-gray-200 bg-gray-100 text-xs" onclick="window.panCanvas(0,80)">&uarr;</button>
                            <span></span>
                            <button class="px-1 py-0.5 rounded text-gray-700 font-bold hover:bg-gray-200 bg-gray-100 text-xs" onclick="window.panCanvas(80,0)">&larr;</button>
                            <span></span>
                            <button class="px-1 py-0.5 rounded text-gray-700 font-bold hover:bg-gray-200 bg-gray-100 text-xs" onclick="window.panCanvas(-80,0)">&rarr;</button>
                            <span></span>
                            <button class="px-1 py-0.5 rounded text-gray-700 font-bold hover:bg-gray-200 bg-gray-100 text-xs" onclick="window.panCanvas(0,-80)">&darr;</button>
                            <span></span>
                        </span>
                        <button id="gridToggleBtn" class="px-4 py-1 rounded text-red-600 font-bold hover:bg-red-100 bg-gray-100" onclick="window.toggleGridOverlay()">Grid</button>
                        <button id="prevBtn" class="px-3 py-1 bg-gray-200 rounded hover:bg-gray-300">&larr;</button>
                        <span id="imgCounter" class="text-gray-700 mt-1 font-bold text-sm">0 of 0</span>
                        <button id="nextBtn" class="px-3 py-1 bg-gray-200 rounded hover:bg-gray-300">&rarr;</button>
                    </div>
                </div>
                <div class="flex gap-3 mb-2 flex-shrink-0 text-xs text-gray-600">
                    <span><span class="inline-block w-2.5 h-2.5 mr-1 border-2" style="border-color:#4f46e5"></span>Dimension</span>
                    <span><span class="inline-block w-2.5 h-2.5 mr-1 border-2" style="border-color:#d97706"></span>Component</span>
                    <span><span class="inline-block w-2.5 h-2.5 mr-1 border-2" style="border-color:#e11d48"></span>Title block</span>
                    <span><span class="inline-block w-2.5 h-2.5 mr-1 border-2" style="border-color:#16a34a"></span>Marked correct</span>
                    <span class="text-gray-400">Solid = matched position (OCR or AI), dashed = approximate. Click to jump to its row</span>
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

    // Tracks the grid-overlay fabric objects so they can be cleared/redrawn
    window.canvasRects = {};

    var canvas = new fabric.Canvas('fabricCanvas', { selection: false });

    // View controls (FIXED: Identity matrix added)
    document.getElementById('resetZoomBtn').onclick = function() { canvas.setViewportTransform([1,0,0,1,0,0]); };
    document.getElementById('zoomInBtn').onclick = function() {
        var zoom = Math.min(canvas.getZoom() * 1.3, 20);
        canvas.zoomToPoint({ x: canvas.getWidth()/2, y: canvas.getHeight()/2 }, zoom);
    };
    document.getElementById('zoomOutBtn').onclick = function() {
        var zoom = Math.max(canvas.getZoom() / 1.3, 0.5);
        canvas.zoomToPoint({ x: canvas.getWidth()/2, y: canvas.getHeight()/2 }, zoom);
    };

    canvas.on('mouse:wheel', function(opt) {
        var zoom = canvas.getZoom() * Math.pow(0.999, opt.e.deltaY);
        if (zoom > 20) zoom = 20; if (zoom < 0.5) zoom = 0.5;
        canvas.zoomToPoint({ x: opt.e.offsetX, y: opt.e.offsetY }, zoom);
        opt.e.preventDefault(); opt.e.stopPropagation();
    });

    // Click-drag panning on the canvas (default drag; markers still get their
    // own click behavior since opt.target is set when a marker is clicked)
    canvas.on('mouse:down', function(opt) {
        if (opt.target) return;
        window.isDraggingPan = true;
        window.lastPosX = opt.e.clientX;
        window.lastPosY = opt.e.clientY;
        canvas.defaultCursor = 'grabbing';
    });

    canvas.on('mouse:move', function(opt) {
        if (window.isDraggingPan) {
            var vpt = canvas.viewportTransform;
            vpt[4] += opt.e.clientX - window.lastPosX;
            vpt[5] += opt.e.clientY - window.lastPosY;
            canvas.requestRenderAll(); window.lastPosX = opt.e.clientX; window.lastPosY = opt.e.clientY;
        }
    });

    canvas.on('mouse:up', function() {
        if (window.isDraggingPan) { canvas.setViewportTransform(canvas.viewportTransform); window.isDraggingPan = false; canvas.defaultCursor = 'default'; }
    });

    window.panCanvas = function(dx, dy) {
        var vpt = canvas.viewportTransform;
        vpt[4] += dx; vpt[5] += dy;
        canvas.setViewportTransform(vpt);
    };

    window.updateRowData = function(id, field, value) {
        var row = window.APP_STATE.extractedData.find(function(r) { return r._id === id; });
        if(row) {
            if(field === 'is_correct') {
                row.is_correct = value;
                var tr = document.getElementById("tr_" + id);
                if(value) tr.classList.add('row-correct'); else tr.classList.remove('row-correct');
            } else {
                row[field] = value;
            }
        }
    };

    window.deleteRow = function(id) {
        window.APP_STATE.extractedData = window.APP_STATE.extractedData.filter(function(r){ return r._id !== id; });
        window.renderChecklistTable();
    };

    document.getElementById('addRowBtn').onclick = function() {
        if(window.APP_STATE.images.length === 0) return alert("Upload images first.");
        var currentImg = window.APP_STATE.images[window.APP_STATE.currentIndex].name;
        var newRow = {
            _id: "MANUAL-" + Math.random().toString(36).substr(2, 9),
            image_name: currentImg,
            Category: "dimension", Component_Name: "", Dwg_View: "", View_Label: "Ungrouped",
            Pt_No: "", Dim_Type: "NEW", Dim_Description: "NEW DIM", Specified: "", Quantity: "", Tolerance: "", Measuring_Tools: "", is_correct: false
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

        var groupOrder = [];
        var groups = {};
        pageData.forEach(function(r) {
            var key = r.View_Label || 'Ungrouped';
            if (!groups[key]) { groups[key] = []; groupOrder.push(key); }
            groups[key].push(r);
        });

        groupOrder.forEach(function(groupKey) {
            var groupRows = groups[groupKey];

            var headerTr = document.createElement('tr');
            headerTr.className = "bg-blue-50 border-b border-t-2 border-blue-200";
            headerTr.innerHTML = '<td colspan="12" class="py-1 px-2 font-bold text-blue-800 text-xs">' + groupKey + '</td>';
            tbody.appendChild(headerTr);

            groupRows.forEach(function(r) {
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
            html += '<td class="py-1 px-1 border-r"><div class="flex items-center gap-1"><input type="text" class="font-mono text-xs" value="' + (r.Dwg_View || '') + '" onchange="window.updateRowData(\\'' + r._id + '\\', \\'Dwg_View\\', this.value)"><button class="text-orange-600 hover:text-orange-800 text-xs flex-shrink-0" title="Show on image" onclick="window.highlightRowCell(\\'' + r._id + '\\')">📍</button></div></td>';
            html += '<td class="py-1 px-1 border-r"><input type="text" value="' + (r.Pt_No || '') + '" onchange="window.updateRowData(\\'' + r._id + '\\', \\'Pt_No\\', this.value)"></td>';
            html += '<td class="py-1 px-1 border-r"><input type="text" class="font-bold text-indigo-700" value="' + (r.Dim_Type || '') + '" onchange="window.updateRowData(\\'' + r._id + '\\', \\'Dim_Type\\', this.value)"></td>';
            html += '<td class="py-1 px-1 border-r"><input type="text" value="' + (r.Dim_Description || '') + '" onchange="window.updateRowData(\\'' + r._id + '\\', \\'Dim_Description\\', this.value)"></td>';
            html += '<td class="py-1 px-1 border-r"><input type="text" value="' + (r.Specified || '') + '" onchange="window.updateRowData(\\'' + r._id + '\\', \\'Specified\\', this.value)"></td>';
            html += '<td class="py-1 px-1 border-r"><input type="text" value="' + (r.Quantity || '') + '" onchange="window.updateRowData(\\'' + r._id + '\\', \\'Quantity\\', this.value)"></td>';
            html += '<td class="py-1 px-1 border-r"><input type="text" class="text-red-600 font-mono" value="' + (r.Tolerance || '') + '" onchange="window.updateRowData(\\'' + r._id + '\\', \\'Tolerance\\', this.value)"></td>';
            // <--- ADD THIS LINE FOR MEASURING TOOLS:
            html += '<td class="py-1 px-1 border-r"><input type="text" value="' + (r.Measuring_Tools || '') + '" onchange="window.updateRowData(\\'' + r._id + '\\', \\'Measuring_Tools\\', this.value)"></td>';
            html += '<td class="py-1 px-1 text-center"><button class="text-red-500 hover:text-white hover:bg-red-500 px-2 py-1 rounded text-xs font-bold" onclick="window.deleteRow(\\'' + r._id + '\\')">X</button></td>';

            tr.innerHTML = html;
            tbody.appendChild(tr);
            });
        });
        if (window.APP_STATE.images.length > 0) { window.renderGridOverlay(); }
    };

    window.GRID_ROWS = ['H','G','F','E','D','C','B','A']; // top-to-bottom, matching the printed border
    window.GRID_COLS = 12;
    window.gridOverlayOn = true;

    window.parseGridRef = function(ref) {
        if (!ref) return null;
        var m = /([A-Ha-h])\s*-?\s*(\d{1,2})/.exec(ref);
        if (!m) return null;
        var rowIdx = window.GRID_ROWS.indexOf(m[1].toUpperCase());
        var colIdx = window.GRID_COLS - parseInt(m[2], 10);
        if (rowIdx < 0 || colIdx < 0 || colIdx >= window.GRID_COLS) return null;
        return { rowIdx: rowIdx, colIdx: colIdx };
    };

    window.cellGeometry = function(rowIdx, colIdx) {
        var imgObj = window.APP_STATE.images[window.APP_STATE.currentIndex];
        var imgW = imgObj._pxWidth, imgH = imgObj._pxHeight;
        if (!imgW || !imgH) return null;
        var left = window.imgOffsetX, top = window.imgOffsetY;
        var w = imgW * window.fabricScaleRatio, h = imgH * window.fabricScaleRatio;
        var colStep = w / window.GRID_COLS, rowStep = h / window.GRID_ROWS.length;
        return { x: left + colIdx * colStep, y: top + rowIdx * rowStep, w: colStep, h: rowStep };
    };

    window.renderGridOverlay = function() {
        Object.keys(window.canvasRects).forEach(function(key) {
            canvas.remove(window.canvasRects[key]);
        });
        window.canvasRects = {};
        if (!window.gridOverlayOn) { canvas.requestRenderAll(); return; }
        if (window.APP_STATE.images.length === 0) return;

        var imgObj = window.APP_STATE.images[window.APP_STATE.currentIndex];
        var imgW = imgObj._pxWidth, imgH = imgObj._pxHeight;
        if (!imgW || !imgH) return;

        var left = window.imgOffsetX, top = window.imgOffsetY;
        var w = imgW * window.fabricScaleRatio, h = imgH * window.fabricScaleRatio;
        var colStep = w / window.GRID_COLS;
        var rowStep = h / window.GRID_ROWS.length;

        // Shade cells that already have at least one extracted row - gives an
        // at-a-glance coverage map so gaps (unshaded areas with real content) stand out.
        var currentImg = imgObj.name;
        var cellItems = {};
        window.APP_STATE.extractedData.forEach(function(r) {
            if (r.image_name !== currentImg) return;
            var cell = window.parseGridRef(r.Dwg_View);
            if (!cell) return;
            var key = cell.rowIdx + '_' + cell.colIdx;
            if (!cellItems[key]) cellItems[key] = [];
            cellItems[key].push(r);
        });

        var MARKER_COLORS = { dimension: '#4f46e5', component: '#d97706', title_block: '#e11d48' };

        Object.keys(cellItems).forEach(function(key) {
            var parts = key.split('_');
            var ri = parseInt(parts[0], 10), ci = parseInt(parts[1], 10);
            var items = cellItems[key];
            var cols = Math.min(4, Math.ceil(Math.sqrt(items.length)));
            var rows = Math.ceil(items.length / cols);
            var padX = colStep / (cols + 1), padY = rowStep / (rows + 1);

            items.forEach(function(r, idx) {
                var mx, my, isPrecise;
                if (r.matched_bbox && r.matched_bbox.length === 4) {
                    // OCR - a real detected pixel position
                    var bx = r.matched_bbox[0], by = r.matched_bbox[1], bw = r.matched_bbox[2], bh = r.matched_bbox[3];
                    mx = window.imgOffsetX + (bx + bw / 2) * window.fabricScaleRatio;
                    my = window.imgOffsetY + (by + bh / 2) * window.fabricScaleRatio;
                    isPrecise = true;
                } else if (r.grid_pointer) {
                    // AI grid pointer - just a (row, col) answer, no pixel/bbox involved.
                    // Position is computed here, at render time, purely from that grid reference.
                    var gp = r.grid_pointer;
                    var gx0 = gp.rect[0], gy0 = gp.rect[1], gx1 = gp.rect[2], gy1 = gp.rect[3];
                    var subW = (gx1 - gx0) / gp.grid_size, subH = (gy1 - gy0) / gp.grid_size;
                    var cellCx = gx0 + (gp.col - 1) * subW + subW / 2;
                    var cellCy = gy0 + (gp.row - 1) * subH + subH / 2;
                    mx = window.imgOffsetX + cellCx * window.fabricScaleRatio;
                    my = window.imgOffsetY + cellCy * window.fabricScaleRatio;
                    isPrecise = true;
                } else {
                    var gc = idx % cols, gr = Math.floor(idx / cols);
                    mx = left + ci * colStep + padX * (gc + 1);
                    my = top + ri * rowStep + padY * (gr + 1);
                    isPrecise = false;
                }
                var color = MARKER_COLORS[r.Category] || '#6b7280';
                var boxSize = 10;

                var box = new fabric.Rect({
                    left: mx, top: my, width: boxSize, height: boxSize, originX: 'center', originY: 'center',
                    fill: r.is_correct ? 'rgba(22,163,74,0.25)' : 'rgba(255,255,255,0.4)',
                    stroke: r.is_correct ? '#16a34a' : color, strokeWidth: 2,
                    strokeDashArray: isPrecise ? null : [3, 2],
                    selectable: false, evented: true, hoverCursor: 'pointer'
                });
                box.on('mousedown', function() { window.selectRowFromMarker(r._id); });
                box.on('mouseover', function() { window.showMarkerTooltip(r, mx, my); });
                box.on('mouseout', function() { window.hideMarkerTooltip(); });

                canvas.add(box);
                window.canvasRects['marker_' + r._id] = box;
            });
        });

        for (var c = 1; c < window.GRID_COLS; c++) {
            var x = left + c * colStep;
            var line = new fabric.Line([x, top, x, top + h], { stroke: 'rgba(220,38,38,0.35)', strokeWidth: 1, selectable: false, evented: false });
            canvas.add(line); window.canvasRects['col_' + c] = line;
        }
        for (var rIdx = 1; rIdx < window.GRID_ROWS.length; rIdx++) {
            var y = top + rIdx * rowStep;
            var line = new fabric.Line([left, y, left + w, y], { stroke: 'rgba(220,38,38,0.35)', strokeWidth: 1, selectable: false, evented: false });
            canvas.add(line); window.canvasRects['row_' + rIdx] = line;
        }
        // Column numbers run high-to-low left-to-right (matches the printed border); row letters A-H top-to-bottom
        for (var ci2 = 0; ci2 < window.GRID_COLS; ci2++) {
            var label = window.GRID_COLS - ci2;
            var lx = left + ci2 * colStep + colStep / 2;
            var t1 = new fabric.Text(String(label), { left: lx, top: top - 14, fontSize: 11, fill: 'rgba(185,28,28,0.7)', selectable: false, evented: false, originX: 'center' });
            canvas.add(t1); window.canvasRects['colLabel_' + ci2] = t1;
        }
        for (var ri2 = 0; ri2 < window.GRID_ROWS.length; ri2++) {
            var ly = top + ri2 * rowStep + rowStep / 2;
            var t2 = new fabric.Text(window.GRID_ROWS[ri2], { left: left - 16, top: ly - 6, fontSize: 11, fill: 'rgba(185,28,28,0.7)', selectable: false, evented: false });
            canvas.add(t2); window.canvasRects['rowLabel_' + ri2] = t2;
        }
        canvas.requestRenderAll();
    };

    window.toggleGridOverlay = function() {
        window.gridOverlayOn = !window.gridOverlayOn;
        window.renderGridOverlay();
    };

    window.highlightCell = function(dwgView) {
        var cell = window.parseGridRef(dwgView);
        if (!cell) { logToScreen("Can't locate '" + dwgView + "' on the grid."); return; }
        var geo = window.cellGeometry(cell.rowIdx, cell.colIdx);
        if (!geo) return;
        if (window.canvasRects['flash']) canvas.remove(window.canvasRects['flash']);
        var flash = new fabric.Rect({
            left: geo.x, top: geo.y, width: geo.w, height: geo.h,
            fill: 'rgba(249,115,22,0.35)', stroke: '#f97316', strokeWidth: 2,
            selectable: false, evented: false
        });
        canvas.add(flash);
        canvas.bringToFront(flash);
        window.canvasRects['flash'] = flash;
        canvas.requestRenderAll();
        setTimeout(function() {
            if (window.canvasRects['flash'] === flash) {
                canvas.remove(flash);
                delete window.canvasRects['flash'];
                canvas.requestRenderAll();
            }
        }, 2000);
    };

    window.highlightRowCell = function(id) {
        var row = window.APP_STATE.extractedData.find(function(r) { return r._id === id; });
        if (row) window.highlightCell(row.Dwg_View);
    };

    window.showMarkerTooltip = function(r, x, y) {
        window.hideMarkerTooltip();
        var label = r.Category === 'dimension'
            ? (r.Dim_Description || r.Dim_Type || '') + ': ' + (r.Specified || '')
            : (r.Component_Name || r.Dim_Description || '');
        var text = new fabric.Text(label, {
            left: x + 8, top: y - 8, fontSize: 12, fill: 'white', backgroundColor: 'rgba(17,24,39,0.85)',
            padding: 4, selectable: false, evented: false
        });
        canvas.add(text);
        canvas.bringToFront(text);
        window.canvasRects['tooltip'] = text;
        canvas.requestRenderAll();
    };

    window.hideMarkerTooltip = function() {
        if (window.canvasRects['tooltip']) {
            canvas.remove(window.canvasRects['tooltip']);
            delete window.canvasRects['tooltip'];
            canvas.requestRenderAll();
        }
    };

    window.selectRowFromMarker = function(id) {
        var row = window.APP_STATE.extractedData.find(function(r) { return r._id === id; });
        if (!row) return;
        if (window.APP_STATE.categoryFilter !== 'all' && window.APP_STATE.categoryFilter !== row.Category) {
            window.APP_STATE.categoryFilter = 'all';
            window.renderChecklistTable();
        }
        var tr = document.getElementById('tr_' + id);
        if (tr) {
            tr.scrollIntoView({ behavior: 'smooth', block: 'center' });
            tr.classList.add('marker-flash');
            setTimeout(function() { tr.classList.remove('marker-flash'); }, 1500);
        }
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
            imgObj._pxWidth = img.width; imgObj._pxHeight = img.height;
            canvas.setBackgroundImage(img, canvas.renderAll.bind(canvas));

            window.renderChecklistTable();
            window.renderGridOverlay();
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
            if (resp.ocr_status) {
                var s = resp.ocr_status;
                if (!s.library_available) {
                    logToScreen("OCR: pytesseract not installed - relying entirely on AI grounding fallback.");
                } else if (!s.tokens_found) {
                    logToScreen("OCR: pytesseract installed but found 0 words - Tesseract binary likely not reachable. Relying on AI grounding fallback.");
                } else {
                    logToScreen("OCR matched " + s.ocr_matched_count + " of " + s.total_count + " rows.");
                }
                if (s.llm_grounding_calls > 0) {
                    logToScreen("AI grid-pointer pass: " + s.llm_grounding_calls + " cell call(s), recovered " + s.llm_grounding_matched_count + " more grid-cell positions.");
                }
                var stillApprox = s.total_count - s.ocr_matched_count - s.llm_grounding_matched_count;
                logToScreen(stillApprox + " row(s) remain approximate (dashed).");
            }
            window.renderChecklistTable();
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
    uvicorn.run(app, host="0.0.0.0", port=8084)
