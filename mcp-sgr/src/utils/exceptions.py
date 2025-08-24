"""Custom exceptions for MCP-SGR."""


class SGRError(Exception):
    """Base exception for SGR errors."""
    pass


class SchemaValidationError(SGRError):
    """Raised when schema validation fails."""
    pass


class LLMError(SGRError):
    """Raised when LLM operations fail."""
    pass


class CacheError(SGRError):
    """Raised when cache operations fail."""
    pass


class RouterError(SGRError):
    """Raised when routing operations fail."""
    pass


class AgentCallError(SGRError):
    """Raised when agent calls fail."""
    pass


class ConfigurationError(SGRError):
    """Raised when configuration is invalid."""
    pass