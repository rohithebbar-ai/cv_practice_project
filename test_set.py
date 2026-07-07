"""
FINAL ISOLATION TEST — tiny JSON request (title block only, ~8 fields,
one object, not an array of many items) vs the same request in prose.

This separates two remaining hypotheses:
A) JSON output format itself is the problem, regardless of quantity.
B) It's specifically about extracting MANY items/an array that causes
   the bailout — a single small JSON object might work fine.

Run this script twice by toggling MODE below.
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

IMAGE_PATH = "local_uploads/PUT_YOUR_REAL_DRAWING_FILENAME_HERE.pdf_p1"

# Toggle this between "json" and "prose" and run both
MODE = "json"  # or "prose"

if MODE == "json":
    SYSTEM_INSTRUCTION = """Look at the title block in the bottom-right corner of this engineering drawing. Extract these 8 fields and return ONLY a single JSON object (not an array) with these exact keys: "drawing_number", "revision", "date", "material", "scale", "sheet_number", "department", "equipment_area". If a field isn't visible, use null for that key."""
else:
    SYSTEM_INSTRUCTION = """Look at the title block in the bottom-right corner of this engineering drawing. Tell me: the drawing number, revision, date, material, scale, sheet number, department, and equipment/area. Just describe what you find in plain sentences."""


def main():
    creds = service_account.IDTokenCredentials.from_service_account_file(
        SERVICE_ACCOUNT_PATH,
        target_audience=AUTH_URL
    )
    session = AuthorizedSession(creds)

    with open(IMAGE_PATH, "rb") as f:
        image_bytes = f.read()

    messages = [{"role": "user", "content": SYSTEM_INSTRUCTION}]
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
            print(f"MODE={MODE} | Sending {len(image_bytes)} bytes...")
            resp = session.post(API_URL, headers={}, data=payload, files=files, timeout=300)
    finally:
        os.remove(tmp_path)

    print("STATUS CODE:", resp.status_code)
    parsed = resp.json()
    msg = parsed["choices"][0]["message"]
    usage = parsed.get("usage", {})

    print("prompt_tokens:", usage.get("prompt_tokens"))
    print("completion_tokens:", usage.get("completion_tokens"))
    print("\n=== CONTENT ===")
    print(msg.get("content", ""))


if __name__ == "__main__":
    main()
