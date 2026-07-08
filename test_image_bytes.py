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


@app.post("/api/extract")
async def extract_data(payload: ExtractRequest):
    session_id = payload.session_id

    system_instruction = """You are an expert QA Engineering AI for TATA STEEL GROWTH SHOP (TGS). Extract a comprehensive inspection checklist from

CRITICAL TGS-STYLE RULES (M-34597 Template):
1. EXTRACT ALL VISIBLE DIMENSIONS: Extract every OD, Length, Chamfer, Groove, Hole, Tap, PCD, Radius, and GD&T parameter.
2. GROUPING: Synchronize blocks by diameter zone (e.g., OD + its length + its chamfer + its radius must be grouped sequentially).
3. Dim_Description: MUST strictly be the feature name ONLY. Do not include values here. Use exact vocabulary: OUTER DIA, LENGTH, TOTAL LENGTH, CHAMFER...
4. Specified: Only the dimension/fit/thread (e.g., Ø320 r6, 1546, 2x45°, M16x30).
5. Tolerance: Extract the EXACT numerical limits (e.g., +0.027/+0.059). If the drawing only shows the fit class (e.g., 'h9', 'f7'), output '(num tol
6. Dim_Type MUST be one of these exact codes: OD, LD, CH, R, DH, INT, EXT, KS, GD.
7. Dwg_View: Identify Grid references (e.g., A1, B2) if visible on drawing borders.

ALSO extract physical named components (e.g. hopper, pulley, idler, conveyor,
walkway, monorail, sizer, scraper, feeder, magnetic separator) visible on the
drawing, using this additional schema per row:
- "Category": "dimension" or "component"
- "Component_Name": populated only for component rows (e.g. "Truck Dump Hopper")
For dimension rows, Component_Name stays empty. For component rows, dimension-
specific fields (Tolerance, Measuring_Tools, etc.) stay empty or "N/A".

ALSO extract specification table data (e.g. tables titled "CONVEYOR
CHARACTERISTICS", or sections like PULLEY, IDLER, DRIVE, MOTOR,
GEAR BOX/REDUCER, MISCELLANEOUS, STRUCTURE) using this additional
schema per row:
- "Category": "specification"
- "Component_Name": the section/ITEM header this row belongs to
  (e.g. "CONVEYOR", "MATERIAL", "BELTING", "PULLEY", "IDLER", "MOTOR")
- "Dim_Description": the PARTICULARS/label text exactly as written
  (e.g. "CAPACITY RATED/DESIGN (T.P.H)")
- "Specified": the SPECIFICATION/value column content exactly as written
  (e.g. "NBC-1", "45/1480", "HOT VULCANIZED")
- Dim_Type, Tolerance, Measuring_Tools, MC_No, Insp_Type: leave as "N/A"
  for specification rows — these fields only apply to dimension rows.
You MUST extract every visible row of every specification table on the
drawing, even if a table spans multiple columns or continues across a
page break. Do not summarize or skip rows — one JSON object per row.

OUTPUT FORMAT:
Return ONLY a valid JSON array of objects. Do not write markdown, do not write explanations.
Each object MUST have the following keys exactly:
"image_name", "Pt_No", "Dwg_View", "Dim_Type", "Dim_Description", "Specified", "Tolerance", "Measuring_Tools", "MC_No", "Insp_Type", "Category", "Component_Name"
"bbox" MUST be an array of exactly 4 floats [x, y, width, height],
each normalized between 0.0 and 1.0, representing the fraction of the
image's total width/height (NOT raw pixel coordinates). For example,
a feature whose bounding box starts at the horizontal midpoint and
covers 10% of the image width would have x=0.5, width=0.1. If you
cannot confidently determine a bbox, return an empty array [].
"""

    prompt_text = system_instruction + "\n\n=== CURRENT BLUEPRINTS TO ANALYZE ===\n"
    gcs_uris = []
    image_files_for_llm = []
    image_dimensions = {}  # image_name -> (width, height), looked up per-item later

    for img in payload.images:
        prompt_text += f"\nFilename: {img.name}\n"

        header_split = img.b64.split(',', 1)
        encoded = header_split[1] if len(header_split) == 2 else header_split[0]
        image_bytes = base64.b64decode(encoded)

        # Decode with PIL here, per image, so we always know THIS image's
        # real pixel size when we rescale its bbox later — never relies on
        # a stray variable from a previous loop iteration.
        pil_img = Image.open(io.BytesIO(image_bytes))
        image_dimensions[img.name] = pil_img.size  # (width, height)

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
            "bbox": []
        }]

    for item in parsed_data:
        item["_id"] = str(uuid.uuid4())
        if "bbox" not in item:
            item["bbox"] = []

        # Rescale normalized bbox [0-1] back to real pixel coordinates,
        # using the specific image this item belongs to — not whichever
        # image happened to be decoded last.
        bbox = item.get("bbox", [])
        img_name = item.get("image_name")
        dims = image_dimensions.get(img_name)

        if dims and isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox):
            img_width, img_height = dims
            x, y, w, h = bbox
            if max(x, y, w, h) <= 1.0:  # only rescale if actually normalized
                item["bbox"] = [
                    round(x * img_width),
                    round(y * img_height),
                    round(w * img_width),
                    round(h * img_height),
                ]

    # Prepare sanitized log (text prompt only, no raw image bytes)
    log_row = {
        "session_id": session_id,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_used": DEPLOYMENT,
        "prompt_used": prompt_text,
        "images": json.dumps(gcs_uris),
        "raw_llm_response": raw_response,
        "parsed_data": json.dumps(parsed_data),
    }
    log_to_bigquery(BQ_EXTRACTION_TABLE, [log_row])

    return {"data": parsed_data}
