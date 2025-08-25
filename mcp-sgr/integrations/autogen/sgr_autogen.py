"""AutoGen integration for MCP-SGR.

Provides SGR-enhanced agents and conversation patterns for AutoGen.
"""

import asyncio
from typing import Any, Dict, List, Optional, Union, Callable

# Note: This is a template. Install autogen with: pip install pyautogen
try:
    from autogen import Agent, ConversableAgent, AssistantAgent, UserProxyAgent
except ImportError:
    # Stub classes for when autogen is not installed
    class Agent:
        pass
    class ConversableAgent:
        def __init__(self, *args, **kwargs):
            pass
        def generate_reply(self, *args, **kwargs):
            return ""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.tools import apply_sgr_tool, wrap_agent_call_tool
from src.utils.llm_client import LLMClient
from src.utils.cache import CacheManager
from src.utils.telemetry import TelemetryManager


class SGRAgent(ConversableAgent):
    """AutoGen agent enhanced with SGR reasoning capabilities."""
    
    def __init__(
        self,
        name: str,
        system_message: str = "",
        schema_type: str = "auto",
        budget: str = "lite",
        sgr_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Initialize SGR-enhanced AutoGen agent.
        
        Args:
            name: Agent name
            system_message: System prompt
            schema_type: SGR schema type to use
            budget: SGR budget (none/lite/full)
            sgr_config: Additional SGR configuration
            **kwargs: Other ConversableAgent parameters
        """
        super().__init__(name=name, system_message=system_message, **kwargs)
        
        self.schema_type = schema_type
        self.budget = budget
        self.sgr_config = sgr_config or {}
        
        # Initialize SGR components
        self.llm_client = LLMClient()
        self.cache_manager = CacheManager()
        self.telemetry = TelemetryManager()
        
        # Flag to track initialization
        self._sgr_initialized = False
        
    async def _ensure_sgr_initialized(self):
        """Ensure SGR components are initialized."""
        if not self._sgr_initialized:
            await self.cache_manager.initialize()
            await self.telemetry.initialize()
            self._sgr_initialized = True
    
    def generate_reply(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[Any] = None,
    ) -> Union[str, Dict, None]:
        """Generate reply with SGR pre/post analysis.
        
        Args:
            messages: Conversation history
            sender: Message sender
            config: Configuration
            
        Returns:
            Enhanced reply with SGR reasoning
        """
        # Run async method in sync context
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(
                self._generate_reply_with_sgr(messages, sender, config)
            )
        finally:
            loop.close()
    
    async def _generate_reply_with_sgr(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[Any] = None,
    ) -> Union[str, Dict, None]:
        """Async generate reply with SGR enhancement."""
        await self._ensure_sgr_initialized()
        
        # Extract the task from the last message
        if not messages or len(messages) == 0:
            return None
            
        last_message = messages[-1]
        task = last_message.get("content", "")
        
        # Apply SGR analysis
        sgr_result = await apply_sgr_tool(
            arguments={
                "task": task,
                "schema_type": self.schema_type,
                "budget": self.budget,
                "context": {
                    "conversation_history": messages[:-1],
                    "sender": sender.name if sender else None,
                    **self.sgr_config
                }
            },
            llm_client=self.llm_client,
            cache_manager=self.cache_manager,
            telemetry=self.telemetry
        )
        
        # Generate base reply using parent method
        base_reply = super().generate_reply(messages, sender, config)
        
        # Enhance reply with SGR insights
        if isinstance(base_reply, str):
            enhanced_reply = self._enhance_reply_with_reasoning(
                base_reply, sgr_result
            )
            return enhanced_reply
        
        return base_reply
    
    def _enhance_reply_with_reasoning(
        self,
        base_reply: str,
        sgr_result: Dict[str, Any]
    ) -> str:
        """Enhance reply with SGR reasoning insights.
        
        Args:
            base_reply: Original reply
            sgr_result: SGR analysis results
            
        Returns:
            Enhanced reply
        """
        confidence = sgr_result.get("confidence", 0)
        reasoning = sgr_result.get("reasoning", {})
        actions = sgr_result.get("suggested_actions", [])
        
        # Add confidence indicator if low
        if confidence < 0.6:
            base_reply = f"[Confidence: {confidence:.2f}] {base_reply}"
        
        # Add suggested actions if any
        if actions and self.sgr_config.get("include_actions", True):
            actions_text = "\n\nSuggested next steps:\n" + "\n".join(
                f"- {action}" for action in actions[:3]
            )
            base_reply += actions_text
        
        return base_reply


class SGRGroupChatManager:
    """Manages group chat with SGR-enhanced analysis."""
    
    def __init__(
        self,
        agents: List[Agent],
        sgr_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize SGR group chat manager.
        
        Args:
            agents: List of agents in the group
            sgr_config: SGR configuration
        """
        self.agents = agents
        self.sgr_config = sgr_config or {
            "analyze_interactions": True,
            "suggest_next_speaker": True,
            "track_confidence": True
        }
        
        # Initialize SGR components
        self.llm_client = LLMClient()
        self.cache_manager = CacheManager()
        self.telemetry = TelemetryManager()
        
    async def analyze_conversation(
        self,
        messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze group conversation with SGR.
        
        Args:
            messages: Conversation history
            
        Returns:
            Analysis results
        """
        await self.cache_manager.initialize()
        await self.telemetry.initialize()
        
        # Create analysis task
        task = f"""Analyze this group conversation:
        
Participants: {[agent.name for agent in self.agents]}
Messages: {len(messages)}

Key topics discussed:
{self._summarize_messages(messages)}

Analyze the conversation dynamics, progress, and suggest next steps.
"""
        
        result = await apply_sgr_tool(
            arguments={
                "task": task,
                "schema_type": "analysis",
                "budget": "full",
                "context": {
                    "conversation_type": "group_chat",
                    "participant_count": len(self.agents)
                }
            },
            llm_client=self.llm_client,
            cache_manager=self.cache_manager,
            telemetry=self.telemetry
        )
        
        return result
    
    def _summarize_messages(self, messages: List[Dict[str, Any]]) -> str:
        """Create a brief summary of messages.
        
        Args:
            messages: List of messages
            
        Returns:
            Summary string
        """
        summary_lines = []
        for i, msg in enumerate(messages[-5:]):  # Last 5 messages
            sender = msg.get("name", "Unknown")
            content = msg.get("content", "")[:100] + "..."
            summary_lines.append(f"{sender}: {content}")
        
        return "\n".join(summary_lines)


# Convenience functions
def create_sgr_assistant(
    name: str,
    system_message: str,
    schema_type: str = "auto",
    **kwargs
) -> SGRAgent:
    """Create an SGR-enhanced assistant agent.
    
    Args:
        name: Agent name
        system_message: System prompt
        schema_type: SGR schema type
        **kwargs: Additional parameters
        
    Returns:
        SGR-enhanced assistant
    """
    return SGRAgent(
        name=name,
        system_message=system_message,
        schema_type=schema_type,
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        **kwargs
    )


def wrap_autogen_agent(
    agent: Agent,
    schema_type: str = "auto",
    budget: str = "lite"
) -> Agent:
    """Wrap existing AutoGen agent with SGR capabilities.
    
    Args:
        agent: AutoGen agent to wrap
        schema_type: SGR schema type
        budget: SGR budget
        
    Returns:
        Wrapped agent
    """
    # Store original generate_reply
    original_generate_reply = agent.generate_reply
    
    # Create wrapper
    async def sgr_generate_reply(messages, sender, config):
        # Initialize SGR components
        llm_client = LLMClient()
        cache_manager = CacheManager()
        telemetry = TelemetryManager()
        
        await cache_manager.initialize()
        await telemetry.initialize()
        
        # Wrap the original method
        result = await wrap_agent_call_tool(
            arguments={
                "agent_endpoint": lambda req: original_generate_reply(
                    req["messages"], req["sender"], req["config"]
                ),
                "agent_request": {
                    "messages": messages,
                    "sender": sender,
                    "config": config
                },
                "sgr_config": {
                    "schema_type": schema_type,
                    "budget": budget,
                    "pre_analysis": True,
                    "post_analysis": True
                }
            },
            llm_client=llm_client,
            cache_manager=cache_manager,
            telemetry=telemetry
        )
        
        return result.get("agent_response", "")
    
    # Replace method
    def wrapped_generate_reply(messages, sender, config):
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(
                sgr_generate_reply(messages, sender, config)
            )
        finally:
            loop.close()
    
    agent.generate_reply = wrapped_generate_reply
    return agent