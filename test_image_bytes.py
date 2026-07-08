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

    system_instruction = """You are an expert QA Engineering AI for TATA STEEL GROWTH SHOP (TGS). Extract a comprehensive inspection checklist from the drawing(s) below.

RULES:
1. Extract every visible OD, Length, Chamfer, Groove, Hole, Tap, PCD, Radius, GD&T dimension.
2. Group blocks by diameter zone (OD + length + chamfer + radius sequentially).
3. Dim_Description = feature name only (e.g. OUTER DIA, LENGTH, TOTAL LENGTH, CHAMFER), no values.
4. Specified = exact dimension/fit/thread as written (e.g. Ø320 r6, 1546, 2x45°, M16x30).
5. Tolerance = exact numeric limits (e.g. +0.027/+0.059), or fit class if that's all that's shown.
6. Dim_Type must be one of: OD, LD, CH, R, DH, INT, EXT, KS, GD.
7. Dwg_View = grid reference (A1, B2) if visible.

Also extract physical named components (hopper, pulley, idler, conveyor, walkway, monorail, sizer, scraper, feeder, magnetic separator):
- Category="component", Component_Name=the component (e.g. "Truck Dump Hopper"). Dimension fields stay "N/A".

Also extract every row of any specification table (e.g. "CONVEYOR CHARACTERISTICS", or sections like PULLEY, IDLER, DRIVE, MOTOR, GEAR BOX/REDUCER, MISCELLANEOUS, STRUCTURE):
- Category="specification", Component_Name=section/ITEM header (e.g. "PULLEY"), Dim_Description=PARTICULARS label, Specified=SPECIFICATION value. Other fields "N/A". Extract every row, even across multi-column tables — one object per row.

OUTPUT: Valid JSON array only, no markdown/explanation. Each object needs exactly these keys:
"image_name", "Pt_No", "Dwg_View", "Dim_Type", "Dim_Description", "Specified", "Tolerance", "Measuring_Tools", "MC_No", "Insp_Type", "Category", "Component_Name", "bbox"

"bbox" MUST be an array of exactly 4 integers [x, y, width, height] in
pixel coordinates, matching the exact image dimensions stated for that
filename below. If you cannot confidently determine a bbox, return [].
"""

    prompt_text = system_instruction + "\n\n=== CURRENT BLUEPRINTS TO ANALYZE ===\n"
    gcs_uris = []
    image_files_for_llm = []
    image_dimensions = {}  # image_name -> (width, height), for reference/debugging

    for img in payload.images:
        header_split = img.b64.split(',', 1)
        encoded = header_split[1] if len(header_split) == 2 else header_split[0]
        image_bytes = base64.b64decode(encoded)

        pil_img = Image.open(io.BytesIO(image_bytes))
        img_width, img_height = pil_img.size
        image_dimensions[img.name] = (img_width, img_height)

        prompt_text += f"\nFilename: {img.name} (image dimensions: {img_width}x{img_height} pixels)\n"

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

OUTPUT FORMAT:
Return ONLY a valid JSON array of objects. Do not write markdown, do not write explanations.
Each object MUST have the following keys exactly:
"image_name", "Pt_No", "Dwg_View", "Dim_Type", "Dim_Description", "Specified", "Tolerance", "Measuring_Tools", "MC_No", "Insp_Type", "Category", "Component_Name", "bbox"
"bbox" MUST be an array of exactly 4 integers [x, y, width, height] in
pixel coordinates, matching the exact image dimensions stated for that
filename below. If you cannot confidently determine a bbox, return [].
"""

Also extract every row of specification tables (e.g. "CONVEYOR CHARACTERISTICS", or sections PULLEY/IDLER/DRIVE/MOTOR/GEAR BOX). For each row you see in these tables:
- Category="specification"
- Component_Name=the section/ITEM this row belongs to (e.g. "PULLEY", "MOTOR")
- Dim_Description=the exact PARTICULARS label text from that row (e.g. "CAPACITY RATED/DESIGN (T.P.H)")
- Specified=the exact SPECIFICATION value from that row (e.g. "NBC-1", "45/1480")
Do not skip rows. Do not leave Dim_Description or Specified blank if the table has content there.
