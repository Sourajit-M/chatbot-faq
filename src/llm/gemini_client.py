from google import genai
from src.config import settings


class GeminiClient:
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.model_name = model_name

    def generate_response(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model_name, contents=prompt
        )
        return response.text.strip()