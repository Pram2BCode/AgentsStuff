import unittest
import os
from unittest.mock import MagicMock, patch

from src.llm_providers.base import LLMProvider
from src.llm_providers.azure_openai import AzureOpenAIProvider
from src.llm_providers.ollama import OllamaProvider
from src.llm_providers.gemini import GeminiProvider
from src.llm_providers.provider_factory import get_llm_provider

class TestLLMProviders(unittest.TestCase):

    def test_azure_openai_provider(self):
        with patch("openai.AzureOpenAI") as mock_azure_openai:
            mock_client_instance = MagicMock()
            mock_azure_openai.return_value = mock_client_instance
            mock_client_instance.chat.completions.create.return_value.choices[0].message.content = "Test response"
            mock_client_instance.embeddings.create.return_value.data[0].embedding = [0.1, 0.2, 0.3]

            provider = AzureOpenAIProvider("fake_key", "fake_endpoint", "fake_version", "fake_deployment")
            self.assertEqual(provider.generate_text("test"), "Test response")
            self.assertEqual(provider.generate_embedding("test"), [0.1, 0.2, 0.3])
            self.assertEqual(provider.get_model_name(), "fake_deployment")

    def test_ollama_provider(self):
        with patch("requests.post") as mock_post:
            mock_post.return_value.json.return_value = {"response": "Ollama response"}
            provider = OllamaProvider("http://localhost:11434", "llama2")
            self.assertEqual(provider.generate_text("test"), "Ollama response")
            
            mock_post.return_value.json.return_value = {"embedding": [0.4, 0.5, 0.6]}
            self.assertEqual(provider.generate_embedding("test"), [0.4, 0.5, 0.6])
            self.assertEqual(provider.get_model_name(), "llama2")

    def test_gemini_provider(self):
        with patch("google.generativeai.GenerativeModel") as mock_gemini_model:
            mock_model_instance = MagicMock()
            mock_gemini_model.return_value = mock_model_instance
            mock_model_instance.generate_content.return_value.text = "Gemini response"
            mock_model_instance.embed_content.return_value = {"embedding": [0.7, 0.8, 0.9]}

            provider = GeminiProvider("fake_key", "gemini-pro")
            self.assertEqual(provider.generate_text("test"), "Gemini response")
            self.assertEqual(provider.generate_embedding("test"), [0.7, 0.8, 0.9])
            self.assertEqual(provider.get_model_name(), "gemini-pro")

    @patch("src.llm_providers.azure_openai.AzureOpenAIProvider")
    @patch("src.llm_providers.ollama.OllamaProvider")
    @patch("src.llm_providers.gemini.GeminiProvider")
    def test_provider_factory(self, mock_gemini, mock_ollama, mock_azure):

            mock_azure.return_value = MagicMock(spec=AzureOpenAIProvider)
            mock_ollama.return_value = MagicMock(spec=OllamaProvider)
            mock_gemini.return_value = MagicMock(spec=GeminiProvider)

            provider = get_llm_provider("azure_openai", api_key="key", azure_endpoint="endpoint", api_version="version", deployment_name="deployment")
            self.assertIsInstance(provider, AzureOpenAIProvider)

            provider = get_llm_provider("ollama", base_url="url", model_name="model")
            self.assertIsInstance(provider, OllamaProvider)

            provider = get_llm_provider("gemini", api_key="key", model_name="model")
            self.assertIsInstance(provider, GeminiProvider)

            with self.assertRaises(ValueError):
                get_llm_provider("unknown_provider")

if __name__ == "__main__":
    unittest.main()


