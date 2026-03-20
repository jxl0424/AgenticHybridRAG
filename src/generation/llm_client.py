"""
LLM client wrapper for interacting with language models.
"""
from openai import OpenAI
import os
from typing import Optional, Any


class LLMClient:
    """Wrapper for LLM inference via OpenAI-compatible API."""
    
    def __init__(
        self, 
        api_key: Optional[str] = "ollama",
        base_url: str = "http://localhost:11434/v1",
        model: str = "qwen2.5:7b-instruct"
    ):
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OLLAMA_API_KEY", "ollama"),
            base_url=base_url,
        )
        self.model = model
        self.total_tokens_used = 0
        self.prompt_tokens_used = 0
        self.completion_tokens_used = 0
    
    def reset_usage(self):
        """Reset token usage counters."""
        self.total_tokens_used = 0
        self.prompt_tokens_used = 0
        self.completion_tokens_used = 0
    
    
    def generate(self, messages: list[dict], max_tokens: int = 1024, temperature: float = 0) -> str:
        """Generate a response from the LLM."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            
            if hasattr(response, 'usage') and response.usage:
                self.prompt_tokens_used += getattr(response.usage, 'prompt_tokens', 0)
                self.completion_tokens_used += getattr(response.usage, 'completion_tokens', 0)
                self.total_tokens_used += getattr(response.usage, 'total_tokens', 0)
                
            if not response.choices:
                return ""
            content = response.choices[0].message.content
            return content.strip() if content else ""
        except Exception as e:
            print(f"LLM generation error: {e}")
            return ""

    def chat_completion(
        self, 
        messages: list[dict], 
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[Any] = "auto",
        max_tokens: int = 2048, 
        temperature: float = 0
    ) -> Any:
        """
        Generate a full chat completion response, supporting tools.
        Returns the raw message object (which includes .content and .tool_calls).
        """
        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = tool_choice
                
            response = self.client.chat.completions.create(**kwargs)
            
            if hasattr(response, 'usage') and response.usage:
                self.prompt_tokens_used += getattr(response.usage, 'prompt_tokens', 0)
                self.completion_tokens_used += getattr(response.usage, 'completion_tokens', 0)
                self.total_tokens_used += getattr(response.usage, 'total_tokens', 0)
                
            return response.choices[0].message if response.choices else None
        except Exception as e:
            print(f"LLM chat completion error: {e}")
            return None
