"""
ISOLATION TEST — same image, same extraction goal, but:
1. NO bbox requirement at all (removed entirely from prompt + schema)
2. Prints full usage/token info and the 'refusal' field for diagnostics
3. Slightly simpler schema to reduce prompt complexity

Purpose: find out if bbox is the suppressor, or if strict JSON-array
extraction itself is the problem regardless of bbox.
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


# NOTE: bbox completely removed from schema and instructions
SYSTEM_INSTRUCTION = """You are an expert QA Engineering AI. Extract every piece of information visible on this engineering drawing: dimensions, tolerances, named components (hopper, pulley, idler, conveyor, walkway, monorail, etc.), coordinate markers, spacing numbers, title block fields, legend entries, and notes.

Return ONLY a valid JSON array of objects. Do not write markdown, do not write explanations.
Each object MUST have these keys exactly:
"image_name", "Category", "Description", "Value"

- "Category": one of "dimension", "component", "coordinate", "title_block", "legend", "note"
- "Description": the feature/field name (e.g. "OUTER DIA", "TRUCK DUMP HOPPER", "DRG NO")
- "Value": the actual value/label as printed (e.g. "320", "E-4061.407", "GAD-38-01-02-03-307-009")

Extract EVERY item you can find. Do not return an empty array if the drawing has any visible content.
"""


def main():
    creds = service_account.IDTokenCredentials.from_service_account_file(
        SERVICE_ACCOUNT_PATH,
        target_audience=AUTH_URL
    )
    session = AuthorizedSession(creds)

    prompt_text = SYSTEM_INSTRUCTION + "\n\n=== BLUEPRINT TO ANALYZE ===\n"

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
            print(f"Sending {len(image_bytes)} bytes, NO bbox requirement...")
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
    print(content[:3000])

    try:
        clean = content.strip()
        if clean.startswith("```json"):
            clean = clean[7:]
        if clean.startswith("```"):
            clean = clean[3:]
        if clean.endswith("```"):
            clean = clean[:-3]
        rows = json.loads(clean.strip())
        print(f"\n=== PARSED {len(rows)} ROWS ===")
        for row in rows[:20]:
            print(f"  [{row.get('Category')}] {row.get('Description')}: {row.get('Value')}")
    except Exception as e:
        print(f"(Could not parse: {e})")


if __name__ == "__main__":
    main()
