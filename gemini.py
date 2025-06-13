
from typing import List
import google.generativeai as genai

from .base import LLMProvider

class GeminiProvider(LLMProvider):
    """Google Gemini LLM provider implementation."""
    api_key: str
    model_name: str

    def __post_init__(self):
        genai.configure(api_key=self.api_key)

    def generate_text(self, prompt: str, **kwargs) -> str:
        model = genai.GenerativeModel(self.model_name)
        response = model.generate_content(prompt, **kwargs)
        return response.text

    def generate_embedding(self, text: str) -> List[float]:
        # Gemini embedding model is typically 'embedding-001'
        model = genai.GenerativeModel('embedding-001')
        response = model.embed_content(content=text, task_type="retrieval_query")
        return response["embedding"]

    def get_model_name(self) -> str:
        return self.model_name


