"""Model router for intelligent backend selection."""

import ast
import logging
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


class TaskType(str, Enum):
    """Types of tasks for routing."""

    ANALYSIS = "analysis"
    PLANNING = "planning"
    DECISION = "decision"
    CODE_GENERATION = "code_generation"
    SUMMARIZATION = "summarization"
    SEARCH = "search"
    GENERAL = "general"


@dataclass
class RoutingRule:
    """A routing rule definition."""

    condition: str  # When clause
    backend: str  # Which backend to use
    model: Optional[str] = None  # Specific model override

    def _resolve_field(self, context: Dict[str, Any], dotted: str) -> Any:
        obj: Any = context
        for part in dotted.split("."):
            if isinstance(obj, dict) and part in obj:
                obj = obj[part]
            else:
                return None
        return obj

    def matches(self, context: Dict[str, Any]) -> bool:
        """Check if rule matches given context."""
        try:
            # Equality check: field == "value"
            if "==" in self.condition:
                left, right = self.condition.split("==", 1)
                field = left.strip()
                equality_value = right.strip().strip('"').strip("'")
                current_val = self._resolve_field(context, field)
                return str(current_val) == equality_value

            # Membership check: field in [..]
            if " in " in self.condition:
                left, right = self.condition.split(" in ", 1)
                field = left.strip()
                try:
                    membership_values = ast.literal_eval(right.strip())
                except Exception as e:
                    logger.error(f"Invalid membership list in condition '{self.condition}': {e}")
                    return False
                current_val = self._resolve_field(context, field)
                return current_val in membership_values

            # Numeric comparisons
            for op in ("<=", ">=", "<", ">"):
                token = f" {op} "
                if token in self.condition:
                    left, right = self.condition.split(token, 1)
                    field = left.strip()
                    try:
                        numeric_value = float(right.strip())
                    except ValueError:
                        logger.error(f"Right-hand side is not numeric in condition '{self.condition}'")
                        return False
                    current_val = self._resolve_field(context, field)
                    try:
                        current_num = float(current_val)  # type: ignore[arg-type]
                    except (TypeError, ValueError):
                        logger.error(
                            f"Non-numeric value for comparison in field '{field}': {current_val}"
                        )
                        return False
                    if op == "<":
                        return current_num < numeric_value
                    if op == ">":
                        return current_num > numeric_value
                    if op == "<=":
                        return current_num <= numeric_value
                    if op == ">=":
                        return current_num >= numeric_value

            return False
        except (ValueError, TypeError, SyntaxError) as e:
            logger.error(f"Error evaluating rule condition '{self.condition}': {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error in rule evaluation: {e}", exc_info=True)
            return False


class ModelRouter:
    """Routes requests to appropriate models based on rules."""

    def __init__(self, policy_file: Optional[str] = None):
        """Initialize router with policy file."""
        self.policy_file = policy_file or os.getenv(
            "ROUTER_POLICY_FILE", "./config/router_policy.yaml"
        )
        self.default_backend = os.getenv("ROUTER_DEFAULT_BACKEND", "ollama")
        self.rules: List[RoutingRule] = []
        self.retry_config = {"max_attempts": 2, "backoff": 0.8}

        self._load_policy()

    def _load_policy(self):
        """Load routing policy from file."""
        policy_path = Path(self.policy_file)

        if not policy_path.exists():
            logger.info(f"No policy file found at {self.policy_file}, using defaults")
            self._load_default_policy()
            return

        try:
            with open(policy_path, "r") as f:
                policy = yaml.safe_load(f)

            # Load rules
            if "router" in policy and "rules" in policy["router"]:
                for rule_def in policy["router"]["rules"]:
                    rule = RoutingRule(
                        condition=rule_def.get("when", "true"),
                        backend=rule_def.get("use", self.default_backend),
                        model=rule_def.get("model"),
                    )
                    self.rules.append(rule)

            # Load retry config
            if "router" in policy and "retry" in policy["router"]:
                self.retry_config.update(policy["router"]["retry"])

            logger.info(f"Loaded {len(self.rules)} routing rules from {self.policy_file}")

        except Exception as e:
            logger.error(f"Failed to load policy file: {e}")
            self._load_default_policy()

    def _load_default_policy(self):
        """Load default routing policy."""
        self.rules = [
            RoutingRule(
                condition='task_type == "code_generation"',
                backend="ollama",
                model="qwen2.5-coder:7b",
            ),
            RoutingRule(
                condition='task_type in ["analysis", "summarization"]',
                backend="ollama",
                model="llama3.1:8b",
            ),
            RoutingRule(condition="tokens > 8000", backend="openrouter"),
            RoutingRule(condition='risk == "high"', backend="openrouter"),
        ]

    def select_backend(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Select backend based on context and rules.

        Args:
            context: Request context including task_type, tokens, etc.

        Returns:
            Dict with backend, model, and retry config
        """
        # Evaluate rules in order
        for rule in self.rules:
            if rule.matches(context):
                logger.debug(f"Rule matched: {rule.condition} -> {rule.backend}")
                return {"backend": rule.backend, "model": rule.model, "retry": self.retry_config}

        # Default fallback
        logger.debug(f"No rule matched, using default backend: {self.default_backend}")
        return {"backend": self.default_backend, "model": None, "retry": self.retry_config}

    def estimate_tokens(self, text: str, multiplier: float = 1.3) -> int:
        """Estimate token count for text.

        Args:
            text: Input text
            multiplier: Safety multiplier

        Returns:
            Estimated token count
        """
        # Simple estimation: ~4 characters per token
        base_estimate = len(text) / 4
        return int(base_estimate * multiplier)

    def detect_task_type(self, task: str, schema_type: Optional[str] = None) -> TaskType:
        """Detect task type from task description.

        Args:
            task: Task description
            schema_type: Explicitly provided schema type

        Returns:
            Detected task type
        """
        if schema_type and schema_type in [t.value for t in TaskType]:
            return TaskType(schema_type)

        # Simple keyword-based detection
        task_lower = task.lower()

        if any(
            word in task_lower
            for word in [
                "analyze",
                "understand",
                "identify",
                "assess",
                "анализ",
                "понять",
                "оценить",
                "разобрать",
            ]
        ):
            return TaskType.ANALYSIS
        elif any(
            word in task_lower
            for word in [
                "plan",
                "strategy",
                "approach",
                "steps",
                "план",
                "стратегия",
                "подход",
                "шаги",
            ]
        ):
            return TaskType.PLANNING
        elif any(
            word in task_lower
            for word in ["decide", "choose", "select", "compare", "решить", "выбрать", "сравнить"]
        ):
            return TaskType.DECISION
        elif any(
            word in task_lower
            for word in [
                "code",
                "implement",
                "program",
                "function",
                "код",
                "реализовать",
                "функция",
            ]
        ):
            return TaskType.CODE_GENERATION
        elif any(
            word in task_lower
            for word in [
                "summarize",
                "summary",
                "brief",
                "overview",
                "суммариз",
                "резюме",
                "кратко",
                "обзор",
            ]
        ):
            return TaskType.SUMMARIZATION
        elif any(
            word in task_lower
            for word in ["search", "find", "locate", "lookup", "искать", "найти", "найди", "поиск"]
        ):
            return TaskType.SEARCH
        else:
            return TaskType.GENERAL

    def create_routing_context(
        self,
        task: str,
        schema_type: Optional[str] = None,
        budget: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create routing context from request.

        Args:
            task: Task description
            schema_type: Schema type if known
            budget: Reasoning budget
            metadata: Additional metadata

        Returns:
            Context dict for routing decisions
        """
        context = {
            "task_type": self.detect_task_type(task, schema_type).value,
            "tokens": self.estimate_tokens(task),
            "budget": budget or "lite",
            "risk": "low",  # Default, could be enhanced
        }

        # Add metadata fields
        if metadata:
            context.update(metadata)

        # Infer risk level
        risk_keywords = ["production", "critical", "security", "payment", "auth"]
        if any(word in task.lower() for word in risk_keywords):
            context["risk"] = "high"

        return context

    def get_retry_config(self) -> Dict[str, Any]:
        """Get retry configuration."""
        return self.retry_config.copy()