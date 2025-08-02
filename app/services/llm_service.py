from google.genai import types
from google.genai.client import Client
from google import genai
import os 
from dotenv import load_dotenv

load_dotenv()


def get_prescription_details(client: Client, prescription_text: str) -> str:
    """
    This function performs a web search using the Google GenAI API and returns
    the text of the search result.

    Args:
    query (str): The search query to be performed.

    Returns:
    str: The search result text from Google GenAI, or an error message.
    """
    try:
        query = f"""
            You are a medical assistant helping patients understand their digital prescriptions.

            Your task is to read the extracted prescription text provided below, and generate a structured, layman-friendly summary. The summary should help the patient understand their diagnosis, prescribed medicines, and any instructions.

            Instructions:
            - Clearly list the medicines with dosage, strength, and purpose as mentioned in the prescription text 
            - Do NOT add any medicines or assumptions not found in the text.
            - If any part of the prescription is unclear or missing, state it honestly.
            - Keep the summary precise, medically accurate, and easy to follow.

            Extracted Prescription Text:
            \"\"\"
            {prescription_text}
            \"\"\"
        """

        response = client.models.generate_content(
            model=os.getenv("GEMINI_MODEL"),
            contents=query,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearchRetrieval())]
            ),
        )
        return response.candidates[0].content.parts[0].text
    except Exception as e:
        # Log the exception or handle it as needed
        return f"[ERROR] An error occurred during prescription evaluation: {str(e)}"


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    result = get_prescription_details(client, "")
    print(result)
