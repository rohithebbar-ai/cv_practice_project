"""
ISOLATION TEST — text-only, same credentials, no file attachment.

Purpose: confirm whether adid/apikey work at all (text-only call),
to isolate whether the earlier "Unauthorized!" was caused specifically
by the files= parameter/multipart encoding, or by the credentials
themselves being invalid.
"""

import json
from google.oauth2 import service_account
from google.auth.transport.requests import AuthorizedSession

AUTH_URL = "https://genai-api-development-one-it-423929642383.asia-south1.run.app"
API_URL = "https://tslgenaiapidev.corp.tatasteel.com/genai"
SERVICE_ACCOUNT_PATH = "secrets/svc-genai-api-dev-oneit.json"

ADID = "ayfph2508"
API_KEY = "SGB7QI6ZVDLCL6W1"
DEPLOYMENT = "gpt-5.2"


def main():
    creds = service_account.IDTokenCredentials.from_service_account_file(
        SERVICE_ACCOUNT_PATH,
        target_audience=AUTH_URL
    )
    session = AuthorizedSession(creds)

    messages = [
        {"role": "user", "content": "Say hello in one sentence."}
    ]

    payload = {
        "deployment_name": DEPLOYMENT,
        "temperature": "0.1",
        "adid": ADID,
        "apikey": API_KEY,
        "messages": json.dumps(messages),
        "max_tokens": "100",
    }

    print("Sending TEXT-ONLY request (no files= parameter)...")
    # NOTE: no files= here at all — this keeps the request as plain
    # application/x-www-form-urlencoded, same as your working text calls.
    resp = session.post(API_URL, headers={}, data=payload, timeout=60)

    print("STATUS CODE:", resp.status_code)
    print("RAW RESPONSE:")
    print(resp.text[:2000])

    try:
        print("\nPARSED JSON:")
        print(json.dumps(resp.json(), indent=2)[:2000])
    except Exception as e:
        print(f"(Could not parse as JSON: {e})")


if __name__ == "__main__":
    main()
