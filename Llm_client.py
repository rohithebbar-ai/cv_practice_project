from google.oauth2 import service_account
from google.auth.transport.requests import AuthorizedSession

GENAI_AUTH_URL = "https://genai-api-development-one-it-423929642383.asia-south1.run.app"
GENAI_SERVICE_ACCOUNT = "secrets/svc-genai-api-dev-oneit.json"

_creds = service_account.IDTokenCredentials.from_service_account_file(
    GENAI_SERVICE_ACCOUNT,
    target_audience=GENAI_AUTH_URL
)
_authed_session = AuthorizedSession(_creds)

def call_llm(prompt_content: List[Dict[str, Any]]) -> str:
    logger.info("Preparing LLM payload...")
    payload = LLM_API_PAYLOAD_TEMPLATE.copy()
    messages = [{"role": "user", "content": prompt_content}]
    payload['messages'] = json.dumps(messages)

    try:
        response = _authed_session.post(
            LLM_API_URL,
            headers=LLM_API_HEADERS,
            data=payload,
            timeout=300
        )
        response.raise_for_status()
        raw_response_text = response.text

        llm_output = raw_response_text
        try:
            parsed_json = json.loads(raw_response_text)
            if isinstance(parsed_json, list) and len(parsed_json) > 0 and 'content' in parsed_json[0]:
                llm_output = parsed_json[0]['content']
            elif isinstance(parsed_json, dict):
                if 'choices' in parsed_json and len(parsed_json['choices']) > 0 and 'message' in parsed_json['choices'][0]:
                    llm_output = parsed_json['choices'][0]['message']['content']
                elif 'response' in parsed_json:
                    llm_output = parsed_json['response']
                elif 'result' in parsed_json:
                    llm_output = parsed_json['result']
        except json.JSONDecodeError:
            pass

        return str(llm_output)
    except Exception as e:
        logger.error("LLM Call Failed: " + str(e))
        return "[]"
