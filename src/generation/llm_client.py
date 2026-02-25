"""
LLM client wrapper for interacting with language models.
"""
from openai import OpenAI
import os
from typing import Optional


class LLMClient:
    """Wrapper for LLM inference via OpenAI-compatible API."""
    
    def __init__(
        self, 
        api_key: Optional[str] = "ollama",
        base_url: str = "http://localhost:11434/v1",
        model: str = "llama3.2"
    ):
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OLLAMA_API_KEY", "ollama"),
            base_url=base_url,
        )
        self.model = model
    
    def generate(self, messages: list[dict], max_tokens: int = 1024, temperature: float = 0) -> str:
        """Generate a response from the LLM."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
    
    def embed(self, texts: list[str], model: str = "text-embedding-3-large") -> list[list[float]]:
        """Generate embeddings for texts."""
        response = self.client.embeddings.create(
            model=model,
            input=texts,
        )
        return [item.embedding for item in response.data]
