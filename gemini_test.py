import os
import cv2
import json
import time
import base64
import requests
import logging
import pandas as pd
import numpy as np
from datetime import datetime

from google.oauth2 import service_account
from google.auth.transport.requests import AuthorizedSession
from dotenv import load_dotenv

load_dotenv()

SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE")
url1 = os.getenv("URL1")
url2 = os.getenv("URL2")

# ========================================================
# AUTH SESSION
# ========================================================

credentials = service_account.IDTokenCredentials.from_service_account_file(SERVICE_ACCOUNT_FILE, target_audience=url1)

authed_session = AuthorizedSession(credentials)

# ========================================================
# GEMINI CALL
# ========================================================

def call_gemini_llm(sys_prompt, user_prompt, file_path, max_tokens=12000):

    for attempt in range(3):
        try:
            payload = {
                "deployment_name": "gemini-3.5-flash",
                "temperature": "0.0",
                "adid": os.getenv("P_No"),
                "apikey": os.getenv("API_KEY"),
                "grounding": "0",
                "max_tokens": str(max_tokens),
                "messages": json.dumps([
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ])
            }

            file_extension = os.path.splitext(file_path)[1].lower()

            mime_type = ''
            if file_extension in ['.jpg', '.jpeg']:
                mime_type = 'image/jpeg'
            else:
                raise ValueError("Unsupported file type")

            files = [('file', (os.path.basename(file_path), open(file_path, 'rb'), mime_type))]
            print(files)
            headers = {}

            response = authed_session.post(url2, headers=headers, data=payload, files=files, timeout = 180)
            # response = authed_session.post(url2, data=payload)
            # response = authed_session.post(url2, headers=headers, data=payload, files=files)

            if response.status_code == 200:
                break

            logging.warning(f"Retry {attempt+1} - Status: {response.status_code}")
            time.sleep(2)

        except Exception as e:
            logging.error(f"Retry {attempt+1} failed: {e}")
            time.sleep(2)

    if response.status_code != 200:
        raise Exception(f"API Error: {response.status_code}")

    response_json = response.json()

    try:
        return response_json["candidates"][0]["content"]["parts"][0]["text"]
    except:
        return ""


if __name__ == "__main__":
    sys_prompt = "You are a helpful assistant."
    user_prompt = "Describe exactly what you see in this engineering drawing. Include any titles, section labels, dimension numbers, drawing numbers, or table content you can read."
    result = call_gemini_llm(sys_prompt, user_prompt, "minimal_logs/MIN-f0e1a003_rendered.jpg")
    print(result)
