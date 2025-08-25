"""LLM client for reasoning generation."""

import asyncio
import json
import logging
import os
from enum import Enum
from typing import List, Optional, Union

import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class LLMBackend(str, Enum):
    """Supported LLM backends."""

    OLLAMA = "ollama"
    OPENROUTER = "openrouter"
    VLLM = "vllm"
    CUSTOM = "custom"


class LLMClient:
    """Unified client for LLM interactions."""

    def __init__(self):
        """Initialize LLM client with configured backends."""
        self.backends = self._parse_backends()
        self.default_backend = os.getenv("ROUTER_DEFAULT_BACKEND", "ollama")

        # Backend configurations
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.ollama_model = os.getenv("OLLAMA_DEFAULT_MODEL", "llama3.1:8b")

        self.openrouter_key = os.getenv("OPENROUTER_API_KEY")
        self.openrouter_model = os.getenv(
            "OPENROUTER_DEFAULT_MODEL", "meta-llama/llama-3.1-8b-instruct"
        )

        self.custom_url = os.getenv("CUSTOM_LLM_URL")

        # Request session
        self._session = None
        # Timeouts
        self.request_timeout_seconds = float(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "120"))

    def _parse_backends(self) -> List[LLMBackend]:
        """Parse enabled backends from environment."""
        backends_str = os.getenv("LLM_BACKENDS", "ollama")
        backends = []

        for backend in backends_str.split(","):
            backend = backend.strip().lower()
            if backend in [b.value for b in LLMBackend]:
                backends.append(LLMBackend(backend))

        return backends or [LLMBackend.OLLAMA]

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        backend: Optional[Union[str, LLMBackend]] = None,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Generate response from LLM.

        Args:
            prompt: The user prompt
            model: Model to use (overrides default)
            backend: Backend to use (overrides default)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            system_prompt: System prompt to prepend
            **kwargs: Additional backend-specific parameters

        Returns:
            Generated text response
        """
        # Determine backend
        if backend is None:
            backend = LLMBackend(self.default_backend)
        elif isinstance(backend, str):
            backend = LLMBackend(backend)

        # Route to appropriate backend
        if backend == LLMBackend.OLLAMA:
            return await self._generate_ollama(
                prompt, model, temperature, max_tokens, system_prompt, **kwargs
            )
        elif backend == LLMBackend.OPENROUTER:
            return await self._generate_openrouter(
                prompt, model, temperature, max_tokens, system_prompt, **kwargs
            )
        elif backend == LLMBackend.CUSTOM:
            return await self._generate_custom(
                prompt, model, temperature, max_tokens, system_prompt, **kwargs
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _generate_ollama(
        self,
        prompt: str,
        model: Optional[str],
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str],
        **kwargs,
    ) -> str:
        """Generate using Ollama backend."""
        session = await self._get_session()

        model = model or self.ollama_model

        # Build request
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            **kwargs,
        }

        try:
            async with session.post(
                f"{self.ollama_host}/api/chat",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.request_timeout_seconds),
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["message"]["content"]
                else:
                    error = await response.text()
                    raise Exception(f"Ollama error {response.status}: {error}")

        except asyncio.TimeoutError:
            logger.error("Ollama request timed out")
            raise
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _generate_openrouter(
        self,
        prompt: str,
        model: Optional[str],
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str],
        **kwargs,
    ) -> str:
        """Generate using OpenRouter backend."""
        if not self.openrouter_key:
            raise ValueError("OPENROUTER_API_KEY not configured")

        session = await self._get_session()
        model = model or self.openrouter_model

        # Build request
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        headers = {
            "Authorization": f"Bearer {self.openrouter_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/mcp-sgr/mcp-sgr",
            "X-Title": "MCP-SGR",
        }

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }

        try:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.request_timeout_seconds),
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["choices"][0]["message"]["content"]
                else:
                    error = await response.text()
                    raise Exception(f"OpenRouter error {response.status}: {error}")

        except Exception as e:
            logger.error(f"OpenRouter generation failed: {e}")
            raise

    async def _generate_custom(
        self,
        prompt: str,
        model: Optional[str],
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str],
        **kwargs,
    ) -> str:
        """Generate using custom backend."""
        if not self.custom_url:
            raise ValueError("CUSTOM_LLM_URL not configured")

        session = await self._get_session()

        # Build request (assumes OpenAI-compatible API)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }

        if model:
            payload["model"] = model

        try:
            async with session.post(
                self.custom_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.request_timeout_seconds),
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    # Try different response formats
                    if "choices" in data:
                        return data["choices"][0]["message"]["content"]
                    elif "response" in data:
                        return data["response"]
                    elif "text" in data:
                        return data["text"]
                    else:
                        return json.dumps(data)
                else:
                    error = await response.text()
                    raise Exception(f"Custom LLM error {response.status}: {error}")

        except Exception as e:
            logger.error(f"Custom LLM generation failed: {e}")
            raise

    async def close(self):
        """Close the client and cleanup resources."""
        if self._session and not self._session.closed:
            await self._session.close()

    def __del__(self):
        """Cleanup on deletion."""
        if self._session and not self._session.closed:
            try:
                asyncio.create_task(self._session.close())
            except RuntimeError:
                # Event loop might be closed
                pass
