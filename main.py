from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.services.ocr_service import extract_text_from_base64
from app.services.llm_service import get_prescription_details
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
try:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    print("✅ gemini client started successfully")
except Exception as e:
    print(f"❌ error creating gemini client: {e}")
    exit(1)

app = FastAPI()


# Request models
class ImageBase64Request(BaseModel):
    image_base64: str


class TextRequest(BaseModel):
    text: str


# Endpoint 1: Extract text from base64 image
@app.post("/extract-text")
def extract_text(request: ImageBase64Request):
    try:
        text = extract_text_from_base64(request.image_base64)
        return {"extracted_text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint 2: Get prescription details from text
@app.post("/get-prescription-details")
def get_details(request: TextRequest):
    try:
        result = get_prescription_details(client, request.text)
        return {"prescription_details": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint 3: Combined - extract text and get prescription details
@app.post("/process-prescription")
def process_prescription(request: ImageBase64Request):
    try:
        text = extract_text_from_base64(request.image_base64)
        result = get_prescription_details(client, text)
        return {"prescription_details": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
