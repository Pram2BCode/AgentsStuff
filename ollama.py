
from typing import List
import requests

from .base import LLMProvider

class OllamaProvider(LLMProvider):
    """Ollama LLM provider implementation."""
    base_url: str
    model_name: str

    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = "llama2"):
        super().__init__(base_url=base_url, model_name=model_name)
        
        # Ensure the base_url has a protocol
        if not (self.base_url.startswith("http://") or self.base_url.startswith("https://")):
            self.base_url = "http://" + self.base_url # Ollama typically runs on http

    def generate_text(self, prompt: str, **kwargs) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {"model": self.model_name, "prompt": prompt, **kwargs}
        response = requests.post(url, json=payload)
        response.raise_for_status()
        # Ollama's API returns a stream of JSON objects, we need to parse them
        # For simplicity, we'll assume a single response for now.
        # In a real application, you'd iterate through response.iter_lines()
        return response.json()["response"]

    def generate_embedding(self, text: str) -> List[float]:
        url = f"{self.base_url}/api/embeddings"
        payload = {"model": self.model_name, "prompt": text}
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()["embedding"]

    def get_model_name(self) -> str:
        return self.model_name


