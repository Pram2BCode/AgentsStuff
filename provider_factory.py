from .azure_openai import AzureOpenAIProvider
from .ollama import OllamaProvider
from .gemini import GeminiProvider
from .base import LLMProvider

def get_llm_provider(provider_name: str, **kwargs) -> LLMProvider:
    """Factory function to get an LLM provider instance."""
    if provider_name == "azure_openai":
        return AzureOpenAIProvider(**kwargs)
    elif provider_name == "ollama":
        return OllamaProvider(**kwargs)
    elif provider_name == "gemini":
        return GeminiProvider(**kwargs)
    else:
        raise ValueError(f"Unknown LLM provider: {provider_name}")


