"""
ISOLATED TEST — raw bytes (in-memory), NOT a file handle, NOT a temp file.
Everything else matches the proven-working genai_vision.py exactly:
- correct adid (ayfph2508h)
- 300s timeout (not 120s)
- same auth, same URL, same payload structure

Purpose: find out definitively whether raw bytes ever worked fine, and the
earlier "I can't see any image" failures were actually caused by the adid
typo / short timeout all along — OR whether file-handle-vs-bytes is a real,
separate issue with this gateway.

Point RAW_IMAGE_PATH at one of the REAL drawing page files in local_uploads/
(a .pdf_p1 file), not test.jpg, so this also doubles as a real-content test.
"""

import json
from google.oauth2 import service_account
from google.auth.transport.requests import AuthorizedSession

AUTH_URL = "https://genai-api-development-one-it-423929642383.asia-south1.run.app"
API_URL = "https://tslgenaiapidev.corp.tatasteel.com/genai"
SERVICE_ACCOUNT_PATH = "secrets/svc-genai-api-dev-oneit.json"

ADID = "ayfph2508h"        # confirmed correct
API_KEY = "SGB7QI6ZVDLCL6W1"
DEPLOYMENT = "gpt-5.2"

# Point this at a REAL drawing page file already in local_uploads/
RAW_IMAGE_PATH = "local_uploads/SESS-dp0gmeekp_GA___ASSIGNMENT_DRAWING_OF_CONV.NBC-1_R4__08.02.2021-Sheet 2.pdf_p1"


def main():
    creds = service_account.IDTokenCredentials.from_service_account_file(
        SERVICE_ACCOUNT_PATH,
        target_audience=AUTH_URL
    )
    session = AuthorizedSession(creds)

    messages = [
        {"role": "user", "content": "Describe what you see in the attached engineering drawing in 2-3 sentences."}
    ]

    payload = {
        "deployment_name": DEPLOYMENT,
        "temperature": "0.1",
        "adid": ADID,
        "apikey": API_KEY,
        "messages": json.dumps(messages),
        "max_tokens": "200",
    }

    # THE VARIABLE UNDER TEST: read bytes into memory, do NOT keep a file
    # handle open, do NOT write to a temp file. Pure in-memory bytes,
    # exactly like agent.py's call_llm() currently does.
    with open(RAW_IMAGE_PATH, "rb") as f:
        image_bytes = f.read()
    # file is now closed - image_bytes is a plain bytes object in memory

    files = {"files": ("page.jpg", image_bytes, "image/jpeg")}

    print(f"Sending RAW BYTES (not file handle), size={len(image_bytes)} bytes...")
    print("Using timeout=300 (matching the proven-working config)...")

    resp = session.post(API_URL, headers={}, data=payload, files=files, timeout=300)

    print("STATUS CODE:", resp.status_code)
    print("RAW RESPONSE:")
    print(resp.text[:2000])

    try:
        parsed = resp.json()
        content = parsed["choices"][0]["message"]["content"]
        print("\n=== MODEL'S ANSWER ===")
        print(content)
    except Exception as e:
        print(f"(Could not extract content: {e})")


if __name__ == "__main__":
    main()
