"""PII redaction utilities."""

import re
import logging
from typing import Any, Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class RedactionPattern:
    """Pattern for redacting PII."""
    name: str
    pattern: re.Pattern
    replacement: str = "[REDACTED]"
    hash_it: bool = False  # Whether to include hash for tracking


class PIIRedactor:
    """Redacts personally identifiable information."""
    
    # Common PII patterns
    PATTERNS = {
        "email": RedactionPattern(
            name="email",
            pattern=re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            replacement="[EMAIL]",
            hash_it=True
        ),
        # Ensure credit cards are detected before generic phone patterns
        "credit_card": RedactionPattern(
            name="credit_card",
            pattern=re.compile(r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|6(?:011|5[0-9]{2})[0-9]{12})\b'),
            replacement="[CREDIT_CARD]",
            hash_it=False
        ),
        "phone": RedactionPattern(
            name="phone",
            # Broader phone pattern to catch short local formats like 555-1234 as well
            pattern=re.compile(r'(?:\+?\d{1,3}[-.\s]?)?(?:\(\d{3}\)|\d{3})?[-.\s]?\d{3}[-.\s]?\d{4}|\b\d{3}[-.\s]?\d{4}\b'),
            replacement="[PHONE]",
            hash_it=True
        ),
        "ssn": RedactionPattern(
            name="ssn",
            pattern=re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            replacement="[SSN]",
            hash_it=False
        ),
        "ip_address": RedactionPattern(
            name="ip_address",
            pattern=re.compile(r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'),
            replacement="[IP_ADDRESS]",
            hash_it=True
        ),
        "api_key": RedactionPattern(
            name="api_key",
            pattern=re.compile(r'\b(?:api[_-]?key|apikey|api[_-]?secret|access[_-]?token)["\']?\s*[:=]\s*["\']?([A-Za-z0-9+/=_-]{20,})["\']?\b', re.IGNORECASE),
            replacement="[API_KEY]",
            hash_it=False
        ),
        "aws_key": RedactionPattern(
            name="aws_key",
            pattern=re.compile(r'\b(?:AKIA[0-9A-Z]{16}|aws_access_key_id\s*=\s*[A-Z0-9]{20})\b'),
            replacement="[AWS_KEY]",
            hash_it=False
        ),
        "date_of_birth": RedactionPattern(
            name="date_of_birth",
            pattern=re.compile(r'\b(?:DOB|date[_\s]?of[_\s]?birth)[:\s]+(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b', re.IGNORECASE),
            replacement="[DOB]",
            hash_it=False
        )
    }
    
    def __init__(self, enabled: bool = True, patterns: Optional[List[str]] = None):
        """Initialize redactor.
        
        Args:
            enabled: Whether redaction is enabled
            patterns: List of pattern names to use (default: all)
        """
        self.enabled = enabled
        
        # Select patterns to use
        if patterns:
            self.active_patterns = {
                name: pattern 
                for name, pattern in self.PATTERNS.items() 
                if name in patterns
            }
        else:
            self.active_patterns = self.PATTERNS.copy()
        
        logger.info(f"PII Redactor initialized with {len(self.active_patterns)} patterns")
    
    def redact_text(self, text: str) -> Tuple[str, Dict[str, int]]:
        """Redact PII from text.
        
        Args:
            text: Text to redact
            
        Returns:
            Tuple of (redacted_text, redaction_counts)
        """
        if not self.enabled or not text:
            return text, {}
        
        redacted = text
        counts = {}
        
        for name, pattern in self.active_patterns.items():
            matches = pattern.pattern.findall(redacted)
            if matches:
                counts[name] = len(matches)
                
                if pattern.hash_it:
                    # Replace with hash for tracking
                    for match in set(matches):
                        hash_suffix = hashlib.md5(str(match).encode()).hexdigest()[:8]
                        replacement = f"{pattern.replacement}:{hash_suffix}"
                        redacted = redacted.replace(match, replacement)
                else:
                    # Simple replacement
                    redacted = pattern.pattern.sub(pattern.replacement, redacted)
        
        return redacted, counts
    
    def redact_dict(self, data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, int]]:
        """Redact PII from dictionary recursively.
        
        Args:
            data: Dictionary to redact
            
        Returns:
            Tuple of (redacted_dict, redaction_counts)
        """
        if not self.enabled:
            return data, {}
        
        total_counts = {}
        
        def _redact_recursive(obj: Any) -> Any:
            if isinstance(obj, str):
                redacted, counts = self.redact_text(obj)
                for k, v in counts.items():
                    total_counts[k] = total_counts.get(k, 0) + v
                return redacted
            elif isinstance(obj, dict):
                return {k: _redact_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_redact_recursive(item) for item in obj]
            else:
                return obj
        
        redacted_data = _redact_recursive(data)
        return redacted_data, total_counts
    
    def redact_logs(self, log_entry: str) -> str:
        """Redact PII from log entries.
        
        Args:
            log_entry: Log entry to redact
            
        Returns:
            Redacted log entry
        """
        if not self.enabled:
            return log_entry
        
        redacted, _ = self.redact_text(log_entry)
        return redacted
    
    def add_custom_pattern(
        self,
        name: str,
        pattern: str,
        replacement: str = "[CUSTOM]",
        hash_it: bool = False
    ):
        """Add a custom redaction pattern.
        
        Args:
            name: Pattern name
            pattern: Regex pattern string
            replacement: Replacement text
            hash_it: Whether to add hash suffix
        """
        try:
            compiled = re.compile(pattern)
            self.active_patterns[name] = RedactionPattern(
                name=name,
                pattern=compiled,
                replacement=replacement,
                hash_it=hash_it
            )
            logger.info(f"Added custom pattern: {name}")
        except re.error as e:
            logger.error(f"Invalid regex pattern for {name}: {e}")
    
    def remove_pattern(self, name: str):
        """Remove a redaction pattern.
        
        Args:
            name: Pattern name to remove
        """
        if name in self.active_patterns:
            del self.active_patterns[name]
            logger.info(f"Removed pattern: {name}")
    
    def get_active_patterns(self) -> List[str]:
        """Get list of active pattern names."""
        return list(self.active_patterns.keys())
    
    def validate_no_pii(self, text: str) -> Tuple[bool, List[str]]:
        """Validate that text contains no PII.
        
        Args:
            text: Text to validate
            
        Returns:
            Tuple of (is_clean, found_pii_types)
        """
        if not self.enabled:
            return True, []
        
        found_types = []
        
        for name, pattern in self.active_patterns.items():
            if pattern.pattern.search(text):
                found_types.append(name)
        
        return len(found_types) == 0, found_types