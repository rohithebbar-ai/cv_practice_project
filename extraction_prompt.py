"""
COMPREHENSIVE EXTRACTION TEST — captures dimensions, tolerances,
component names, AND every small number/label visible on the drawing
(coordinates, elevations, grid refs, quantities, notes) — not just the
two categories from before.

This is a standalone test script (matches the proven-working temp-file
pattern) so you can validate the new prompt BEFORE merging it into
agent.py's extract_data().

Run: python .\comprehensive_extraction_test.py
"""

import json
import tempfile
import os
from google.oauth2 import service_account
from google.auth.transport.requests import AuthorizedSession

AUTH_URL = "https://genai-api-development-one-it-423929642383.asia-south1.run.app"
API_URL = "https://tslgenaiapidev.corp.tatasteel.com/genai"
SERVICE_ACCOUNT_PATH = "secrets/svc-genai-api-dev-oneit.json"

ADID = "ayfph2508h"
API_KEY = "SGB7QI6ZVDLCL6W1"
DEPLOYMENT = "gpt-5.2"

# Point this at a REAL drawing page file in local_uploads/
IMAGE_PATH = "local_uploads/PUT_YOUR_REAL_DRAWING_FILENAME_HERE.pdf_p1"


SYSTEM_INSTRUCTION = """You are an expert QA Engineering AI for TATA STEEL GROWTH SHOP (TGS). Extract EVERY piece of information visible on this engineering drawing — leave nothing out. This data will be used to train a machine learning model, so completeness and accuracy are critical.

RULE 0 — EXTRACT EVERYTHING, MULTIPLE CATEGORIES ARE MANDATORY:
This drawing may contain any combination of:
(a) Dimensional/tolerance data (diameters, lengths, chamfers, radii, GD&T)
(b) Named physical components (hopper, pulley, idler, conveyor, walkway, monorail, sizer, scraper, feeder, magnetic separator, etc.)
(c) Coordinate/reference markers (E-xxxx, N-xxxx, EL+xxx, grid references like A1, B2)
(d) Small numeric labels and callouts (bolt counts, spacing dimensions like "7 @ 3000 = 21000", quantities, part numbers)
(e) Title block fields (drawing number, revision, date, material, scale, sheet number, department, equipment area)
(f) Legend/key entries and their symbol definitions
(g) Notes, remarks, or text annotations anywhere on the drawing

You MUST extract items from EVERY category that is actually present. NEVER return an empty array if the drawing contains ANY visible content. If a category has nothing on this drawing, simply produce no rows for that category — but do not let that suppress rows from OTHER categories that ARE present.

EXTRACTION DETAIL RULES:
1. Extract every OD, Length, Chamfer, Groove, Hole, Tap, PCD, Radius, and GD&T parameter you can find, with EXACT numerical values as printed.
2. Extract every coordinate marker exactly as printed (e.g. "E-4061.407", "N-784.454", "EL+363.750").
3. Extract every small dimension/spacing number on the drawing, even ones that look minor (e.g. "3100", "2010", "5375", "7 @ 3000 = 21000").
4. Extract every named component with its label exactly as it appears (e.g. "TRUCK DUMP HOPPER", "TAIL PULLEY", "MONORAIL").
5. Extract the full title block: drawing number, revision number, date, material, scale, sheet number/size, department, equipment/area, detail, drawn/checked/approved names.
6. Extract every legend entry and what its symbol represents (e.g. "T.I. = TROUGHING IDLER").
7. Dim_Description MUST be the feature name ONLY — never include the value itself in this field. Use consistent vocabulary: OUTER DIA, LENGTH, TOTAL LENGTH, CHAMFER, RELIEF GROOVE, RADIUS, KEY WAY, COUNTER BORE, REAMED HOLE, DRILL SIZE, TAP SIZE, PCD, DIM, CONCENTRICITY, COORDINATE, SPACING, TITLE_BLOCK_FIELD, LEGEND_ENTRY, NOTE.
8. Specified: the actual dimension/value/fit/thread exactly as printed (e.g. Ø320 r6, 1546, 2x45°, M16x30, E-4061.407).
9. Tolerance: EXACT numerical limits if shown (e.g. +0.027/+0.059). If only a fit class is shown (e.g. 'h9'), output '(num tol to fill)'. If no tolerance applies (components, coordinates, title block, etc.), output 'N/A'.
10. Dim_Type: use OD, LD, CH, R, DH, INT, EXT, KS, GD for dimensional rows; use COORD for coordinate markers; use TITLE for title block fields; use LEGEND for legend entries; use COMP for components; use NOTE for text annotations.
11. Dwg_View: identify Grid references (e.g. A1, B2) if visible on drawing borders, or the named view (e.g. "PLAN", "ELEVATION (VIEW-XX)") if grid refs aren't present.
12. Category: "dimension", "component", "coordinate", "title_block", "legend", or "note" — matching the Dim_Type used.
13. Component_Name: populate ONLY for component rows with the component's visible label. Leave empty for all other row types.

OUTPUT FORMAT:
Return ONLY a valid JSON array of objects. Do not write markdown, do not write explanations, do not add any text before or after the array.
Each object MUST have the following keys exactly:
"image_name", "Pt_No", "Dwg_View", "Dim_Type", "Dim_Description", "Specified", "Tolerance", "Measuring_Tools", "MC_No", "Insp_Type", "Category", "Component_Name", "bbox"

"bbox" should be an array of 4 integers [x, y, width, height] if you can identify the feature's location. If you cannot confidently identify the bbox, use an empty array [] — this does NOT mean you should skip the row. Always include every row you found, regardless of bbox confidence.
"""


def main():
    creds = service_account.IDTokenCredentials.from_service_account_file(
        SERVICE_ACCOUNT_PATH,
        target_audience=AUTH_URL
    )
    session = AuthorizedSession(creds)

    prompt_text = SYSTEM_INSTRUCTION + "\n\n=== CURRENT BLUEPRINT TO ANALYZE ===\n"

    with open(IMAGE_PATH, "rb") as f:
        image_bytes = f.read()

    messages = [{"role": "user", "content": prompt_text}]
    payload = {
        "deployment_name": DEPLOYMENT,
        "temperature": "0.1",
        "adid": ADID,
        "apikey": API_KEY,
        "messages": json.dumps(messages),
        "max_tokens": "4000",
    }

    # Write to temp file and open as real file handle — this is the
    # proven-working pattern; raw in-memory bytes do NOT work with
    # this gateway.
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            files = {"files": ("page.jpg", f, "image/jpeg")}
            print(f"Sending image ({len(image_bytes)} bytes) with comprehensive extraction prompt...")
            resp = session.post(API_URL, headers={}, data=payload, files=files, timeout=300)
    finally:
        os.remove(tmp_path)

    print("STATUS CODE:", resp.status_code)

    try:
        parsed = resp.json()
        content = parsed["choices"][0]["message"]["content"]
        print("\n=== RAW MODEL CONTENT ===")
        print(content[:3000])

        # Try to parse as JSON array
        clean = content.strip()
        if clean.startswith("```json"):
            clean = clean[7:]
        if clean.startswith("```"):
            clean = clean[3:]
        if clean.endswith("```"):
            clean = clean[:-3]
        clean = clean.strip()

        rows = json.loads(clean)
        print(f"\n=== PARSED {len(rows)} ROWS ===")
        for row in rows:
            print(f"  [{row.get('Category', '?')}] {row.get('Dim_Description', row.get('Component_Name', '?'))}: {row.get('Specified', '')}")

    except Exception as e:
        print(f"\n(Could not parse response: {e})")
        print("RAW RESPONSE TEXT:")
        print(resp.text[:2000])


if __name__ == "__main__":
    main()
