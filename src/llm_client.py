from abc import ABC, abstractmethod
from typing import List, Optional
import openai
from anthropic import Anthropic
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
import structlog

logger = structlog.get_logger(__name__)


class LLMClient(ABC):
    """Abstract base for LLM clients"""

    @abstractmethod
    def complete(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        stop_sequences: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate completion for prompt"""
        pass


class OpenAIClient(LLMClient):
    """Production OpenAI client with retries"""

    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        # Initialize with explicit parameters only, avoid proxy issues
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def complete(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        stop_sequences: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate completion with retries"""
        try:
            # Use provided system prompt or default
            system_content = system_prompt or "You are a helpful AI agent."
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_sequences,
            )

            content = response.choices[0].message.content
            if content is None:
                content = "Error: OpenAI returned empty content"

            logger.debug(
                "LLM completion",
                prompt_preview=prompt[:100] + "..." if len(prompt) > 100 else prompt,
                response_preview=content[:100] + "..." if content and len(content) > 100 else content,
                response_length=len(content) if content else 0,
                model=self.model,
                temperature=temperature,
                max_tokens=max_tokens,
                stop_sequences=stop_sequences
            )
            return content

        except Exception as e:
            logger.error("OpenAI API error", error=str(e))
            raise


class AnthropicClient(LLMClient):
    """Production Anthropic client with retries"""

    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        self.client = Anthropic(api_key=api_key)
        self.model = model

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def complete(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        stop_sequences: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate completion with retries"""
        try:
            # Build messages - Anthropic handles system prompts differently
            if system_prompt:
                # For Anthropic, we'll prepend system prompt to the user message
                full_prompt = f"{system_prompt}\n\n{prompt}"
            else:
                full_prompt = prompt
                
            response = self.client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                stop_sequences=stop_sequences or [],
            )

            # Handle different content types
            content = response.content[0]
            if hasattr(content, 'text'):
                content = content.text
            else:
                content = str(content)
            logger.debug(
                "LLM completion",
                prompt_preview=prompt[:50],
                response_preview=content[:50],
            )
            return content

        except Exception as e:
            logger.error("Anthropic API error", error=str(e))
            raise


def create_llm_client(provider: str, api_key: str) -> LLMClient:
    """Factory for LLM clients"""
    if provider == "openai":
        return OpenAIClient(api_key)
    elif provider == "anthropic":
        return AnthropicClient(api_key)
    else:
        raise ValueError(f"Unknown provider: {provider}")
