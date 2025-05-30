import os
import json
import requests
from google.oauth2 import service_account
import google.auth.transport.requests
from dotenv import load_dotenv

load_dotenv()

SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE_PATH")
SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
VISION_API_URL = "https://vision.googleapis.com/v1/images:annotate"


def _get_access_token() -> str:
    """
    Loads the service account and returns a valid access token.
    """
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    auth_req = google.auth.transport.requests.Request()
    credentials.refresh(auth_req)
    return credentials.token


def extract_text_from_base64(base64_image: str) -> str:
    """
    Sends a base64-encoded image to Google Vision API and returns extracted text.

    Raises:
        RuntimeError if Vision API call fails or response is invalid.
    """
    access_token = _get_access_token()

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    body = {
        "requests": [
            {
                "image": {"content": base64_image},
                "features": [{"type": "TEXT_DETECTION"}],
            }
        ]
    }

    response = requests.post(VISION_API_URL, headers=headers, data=json.dumps(body))

    if response.status_code != 200:
        raise RuntimeError(
            f"Vision API error: {response.status_code} - {response.text}"
        )

    result = response.json()
    try:
        return result["responses"][0]["fullTextAnnotation"]["text"]
    except (KeyError, IndexError):
        return ""


if __name__ == "__main__":
    import base64

    image_path = r"sample_images/images.png"
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    sample_response = extract_text_from_base64(base64_image=base64_image)
    print(sample_response)