"""
LOOSE JSON TEST — ask for JSON output, but do NOT enforce a rigid
schema (no fixed keys, no bbox, no strict category list). Let the
model decide its own structure for whatever it finds.

Purpose: isolate whether it's specifically the RIGID SCHEMA (fixed
keys, fixed categories) causing the empty-array bailout, or whether
ANY JSON-formatting request causes it regardless of schema strictness.
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


# Deliberately loose — no fixed keys, no bbox, no strict category enum
SYSTEM_INSTRUCTION = """Look at this engineering drawing carefully. Extract everything you can see: dimensions, numbers, labels, component names, coordinates, title block info, notes — anything at all that appears on the drawing.

Return your findings as a JSON array. Each item can have whatever fields make sense for that piece of information (for example a dimension might have "type", "value", "tolerance"; a component might just have "name"; a title block field might have "field" and "value"). Use your own judgment on structure — there is no fixed schema to follow.

Return ONLY the JSON array, nothing else.
"""


def main():
    creds = service_account.IDTokenCredentials.from_service_account_file(
        SERVICE_ACCOUNT_PATH,
        target_audience=AUTH_URL
    )
    session = AuthorizedSession(creds)

    prompt_text = SYSTEM_INSTRUCTION

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

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            files = {"files": ("page.jpg", f, "image/jpeg")}
            print(f"Sending {len(image_bytes)} bytes, LOOSE JSON (no fixed schema)...")
            resp = session.post(API_URL, headers={}, data=payload, files=files, timeout=300)
    finally:
        os.remove(tmp_path)

    print("STATUS CODE:", resp.status_code)

    parsed = resp.json()
    msg = parsed["choices"][0]["message"]
    usage = parsed.get("usage", {})

    print("\n=== DIAGNOSTICS ===")
    print("refusal field:", msg.get("refusal"))
    print("finish_reason:", parsed["choices"][0].get("finish_reason"))
    print("prompt_tokens:", usage.get("prompt_tokens"))
    print("completion_tokens:", usage.get("completion_tokens"))

    content = msg.get("content", "")
    print("\n=== RAW CONTENT ===")
    print(content[:4000])


if __name__ == "__main__":
    main()
