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
        # Prefer explicit override; else choose based on available credentials/backends
        explicit_backend = os.getenv("ROUTER_DEFAULT_BACKEND")
        if explicit_backend:
            self.default_backend = explicit_backend
        else:
            # If OpenRouter key present and backend enabled, prefer OpenRouter
            if os.getenv("OPENROUTER_API_KEY") and any(b.value == "openrouter" for b in self.backends):
                self.default_backend = "openrouter"
            elif any(b.value == "ollama" for b in self.backends):
                self.default_backend = "ollama"
            elif any(b.value == "custom" for b in self.backends):
                self.default_backend = "custom"
            else:
                # Fallback
                self.default_backend = "openrouter" if os.getenv("OPENROUTER_API_KEY") else "ollama"

        # Backend configurations
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.ollama_model = os.getenv("OLLAMA_DEFAULT_MODEL", "llama3.1:8b")

        self.openrouter_key = os.getenv("OPENROUTER_API_KEY")
        self.openrouter_model = os.getenv(
            "OPENROUTER_DEFAULT_MODEL", "meta-llama/llama-3.1-8b-instruct"
        )

        self.custom_url = os.getenv("CUSTOM_LLM_URL")

        # Connection pooling and session management
        self._session = None
        self._session_lock = asyncio.Lock()
        
        # Connection pool settings
        self.max_connections = int(os.getenv("LLM_MAX_CONNECTIONS", "100"))
        self.max_connections_per_host = int(os.getenv("LLM_MAX_CONNECTIONS_PER_HOST", "30"))
        
        # Timeouts and retry settings
        self.request_timeout_seconds = float(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "120"))
        self.connection_timeout = float(os.getenv("LLM_CONNECTION_TIMEOUT", "10"))
        self.retry_attempts = int(os.getenv("LLM_RETRY_ATTEMPTS", "3"))
        
        # Performance monitoring
        self._request_count = 0
        self._error_count = 0
        self._total_response_time = 0.0

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
        """Get or create aiohttp session with connection pooling."""
        async with self._session_lock:
            if self._session is None or self._session.closed:
                # Create connection pool with limits
                connector = aiohttp.TCPConnector(
                    limit=self.max_connections,
                    limit_per_host=self.max_connections_per_host,
                    ttl_dns_cache=300,  # DNS cache TTL
                    use_dns_cache=True,
                    keepalive_timeout=30,
                    enable_cleanup_closed=True
                )
                
                # Create timeout configuration
                timeout = aiohttp.ClientTimeout(
                    total=self.request_timeout_seconds,
                    connect=self.connection_timeout,
                    sock_connect=self.connection_timeout,
                    sock_read=30
                )
                
                self._session = aiohttp.ClientSession(
                    connector=connector,
                    timeout=timeout,
                    headers={
                        'User-Agent': 'MCP-SGR/0.1.0 (LLM Client)',
                        'Accept': 'application/json',
                        'Accept-Encoding': 'gzip, deflate'
                    }
                )
                
                logger.debug("Created new HTTP session with connection pooling")
            
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

    async def generate_batch(
        self,
        prompts: List[str],
        model: Optional[str] = None,
        backend: Optional[Union[str, LLMBackend]] = None,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        system_prompt: Optional[str] = None,
        max_concurrent: int = 5,
        **kwargs,
    ) -> List[str]:
        """Generate responses for multiple prompts concurrently.
        
        Args:
            prompts: List of prompts to process
            max_concurrent: Maximum number of concurrent requests
            **kwargs: Additional parameters passed to generate()
            
        Returns:
            List of generated responses in the same order as prompts
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_prompt(prompt: str) -> str:
            async with semaphore:
                return await self.generate(
                    prompt=prompt,
                    model=model,
                    backend=backend,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system_prompt=system_prompt,
                    **kwargs
                )
        
        # Process all prompts concurrently
        tasks = [process_prompt(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error strings
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch processing failed for prompt {i}: {result}")
                processed_results.append(f"Error: {str(result)}")
            else:
                processed_results.append(result)
        
        return processed_results

    def get_performance_stats(self) -> dict:
        """Get performance statistics for the LLM client."""
        if self._request_count == 0:
            avg_response_time = 0.0
        else:
            avg_response_time = self._total_response_time / self._request_count
        
        error_rate = (self._error_count / self._request_count * 100) if self._request_count > 0 else 0.0
        
        return {
            "total_requests": self._request_count,
            "total_errors": self._error_count,
            "error_rate_percent": round(error_rate, 2),
            "average_response_time_seconds": round(avg_response_time, 3),
            "session_open": self._session is not None and not self._session.closed
        }

    def reset_performance_stats(self):
        """Reset performance statistics."""
        self._request_count = 0
        self._error_count = 0
        self._total_response_time = 0.0

    async def health_check(self, timeout: float = 10.0) -> dict:
        """Perform health check on configured backends.
        
        Args:
            timeout: Timeout for health check requests
            
        Returns:
            Dictionary with health status for each backend
        """
        health_status = {}
        
        for backend in self.backends:
            try:
                if backend == LLMBackend.OLLAMA:
                    status = await self._health_check_ollama(timeout)
                elif backend == LLMBackend.OPENROUTER:
                    status = await self._health_check_openrouter(timeout)
                elif backend == LLMBackend.CUSTOM:
                    status = await self._health_check_custom(timeout)
                else:
                    status = {"status": "unknown", "message": "Backend not implemented"}
                
                health_status[backend.value] = status
                
            except Exception as e:
                health_status[backend.value] = {
                    "status": "error",
                    "message": str(e)
                }
        
        return health_status

    async def _health_check_ollama(self, timeout: float) -> dict:
        """Health check for Ollama backend."""
        session = await self._get_session()
        
        try:
            async with session.get(
                f"{self.ollama_host}/api/tags",
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    models = [model["name"] for model in data.get("models", [])]
                    return {
                        "status": "healthy",
                        "available_models": models,
                        "default_model": self.ollama_model
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "message": f"HTTP {response.status}"
                    }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    async def _health_check_openrouter(self, timeout: float) -> dict:
        """Health check for OpenRouter backend."""
        if not self.openrouter_key:
            return {
                "status": "unconfigured",
                "message": "No API key configured"
            }
        
        session = await self._get_session()
        
        try:
            async with session.get(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {self.openrouter_key}"},
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                if response.status == 200:
                    return {
                        "status": "healthy",
                        "default_model": self.openrouter_model
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "message": f"HTTP {response.status}"
                    }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    async def _health_check_custom(self, timeout: float) -> dict:
        """Health check for custom backend."""
        if not self.custom_url:
            return {
                "status": "unconfigured",
                "message": "No custom URL configured"
            }
        
        session = await self._get_session()
        
        try:
            # Try a simple health check endpoint
            health_url = f"{self.custom_url.rstrip('/')}/health"
            async with session.get(
                health_url,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                if response.status == 200:
                    return {"status": "healthy"}
                else:
                    return {
                        "status": "unhealthy", 
                        "message": f"HTTP {response.status}"
                    }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    async def _track_request_performance(self, start_time: float, success: bool):
        """Track request performance metrics."""
        response_time = asyncio.get_event_loop().time() - start_time
        
        self._request_count += 1
        self._total_response_time += response_time
        
        if not success:
            self._error_count += 1
        
        # Log slow requests
        if response_time > 30.0:  # 30 seconds threshold
            logger.warning(f"Slow LLM request: {response_time:.2f}s")

    async def close(self):
        """Close the client and cleanup resources."""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.debug("Closed HTTP session")

    def __del__(self):
        """Cleanup on deletion."""
        if self._session and not self._session.closed:
            try:
                asyncio.create_task(self._session.close())
            except RuntimeError:
                # Event loop might be closed
                pass
