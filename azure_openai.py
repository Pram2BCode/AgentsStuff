import os
from openai import AzureOpenAI, APIConnectionError
from typing import List
from .base import LLMProvider

class AzureOpenAIProvider(LLMProvider):
    """LLM Provider for Azure OpenAI Service."""
    api_key: str
    azure_endpoint: str
    api_version: str
    deployment_name: str
    client: AzureOpenAI = None # Define client as a field

    def __post_init__(self):
        # Ensure the endpoint has a protocol
        if not (self.azure_endpoint.startswith("http://") or self.azure_endpoint.startswith("https://")):
            self.azure_endpoint = "https://" + self.azure_endpoint
        self.client = AzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.azure_endpoint,
            api_version=self.api_version
        )
        self.model_name = self.deployment_name

    def generate_text(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except APIConnectionError as e:
            raise ConnectionError(f"Could not connect to Azure OpenAI: {e}") from e

    def generate_embedding(self, text: str) -> List[float]:
        try:
            response = self.client.embeddings.create(input=text, model=self.deployment_name)
            return response.data[0].embedding
        except APIConnectionError as e:
            raise ConnectionError(f"Could not connect to Azure OpenAI: {e}") from e

    def get_model_name(self) -> str:
        return self.model_name


