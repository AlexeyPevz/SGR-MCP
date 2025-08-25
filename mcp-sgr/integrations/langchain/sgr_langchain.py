"""LangChain integration for MCP-SGR.

Provides SGR-enhanced runnables and chains for LangChain applications.
"""

from typing import Any, Dict, List, Optional, Type
from langchain.schema.runnable import Runnable, RunnableConfig
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.schema import BaseMessage, HumanMessage, AIMessage

from ...src.tools import apply_sgr_tool, wrap_agent_call_tool
from ...src.utils.llm_client import LLMClient
from ...src.utils.cache import CacheManager
from ...src.utils.telemetry import TelemetryManager


class SGRRunnable(Runnable):
    """LangChain Runnable that applies SGR analysis to any task."""
    
    def __init__(
        self,
        schema_type: str = "auto",
        budget: str = "lite",
        llm_client: Optional[LLMClient] = None
    ):
        self.schema_type = schema_type
        self.budget = budget
        self.llm_client = llm_client or LLMClient()
        self.cache_manager = CacheManager()
        self.telemetry = TelemetryManager()
        
    async def ainvoke(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Async invoke SGR analysis."""
        await self.cache_manager.initialize()
        await self.telemetry.initialize()
        
        task = input if isinstance(input, str) else str(input)
        
        result = await apply_sgr_tool(
            arguments={
                "task": task,
                "schema_type": self.schema_type,
                "budget": self.budget,
                "context": kwargs.get("context", {})
            },
            llm_client=self.llm_client,
            cache_manager=self.cache_manager,
            telemetry=self.telemetry
        )
        
        return result
    
    def invoke(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Sync invoke (runs async in loop)."""
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.ainvoke(input, config, **kwargs))
        finally:
            loop.close()


class SGRChainWrapper:
    """Wraps any LangChain chain with SGR pre/post analysis."""
    
    def __init__(
        self,
        chain: Runnable,
        sgr_config: Optional[Dict[str, Any]] = None
    ):
        self.chain = chain
        self.sgr_config = sgr_config or {
            "schema_type": "auto",
            "budget": "lite",
            "pre_analysis": True,
            "post_analysis": True
        }
        self.llm_client = LLMClient()
        self.cache_manager = CacheManager()
        self.telemetry = TelemetryManager()
    
    async def arun(self, input: Any, **kwargs) -> Dict[str, Any]:
        """Run chain with SGR wrapper."""
        await self.cache_manager.initialize()
        await self.telemetry.initialize()
        
        # Simulate agent endpoint for wrap_agent_call
        async def chain_as_agent(request):
            return await self.chain.ainvoke(request["input"])
        
        result = await wrap_agent_call_tool(
            arguments={
                "agent_endpoint": chain_as_agent,
                "agent_request": {"input": input},
                "sgr_config": self.sgr_config
            },
            llm_client=self.llm_client,
            cache_manager=self.cache_manager,
            telemetry=self.telemetry
        )
        
        return result


class SGRAnalysisChain(Runnable):
    """Dedicated chain for structured analysis tasks."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.sgr = SGRRunnable(schema_type="analysis", budget="full")
    
    def invoke(self, input: str, **kwargs) -> Dict[str, Any]:
        """Analyze input and return structured results."""
        result = self.sgr.invoke(input, **kwargs)
        
        if self.verbose:
            print(f"Confidence: {result['confidence']}")
            print(f"Key insights: {result['reasoning']}")
        
        return result


class SGRPlanningChain(Runnable):
    """Chain for structured planning tasks."""
    
    def __init__(self):
        self.sgr = SGRRunnable(schema_type="planning", budget="full")
    
    def invoke(self, input: str, **kwargs) -> Dict[str, Any]:
        """Create structured plan from input."""
        return self.sgr.invoke(input, **kwargs)


class SGRDecisionChain(Runnable):
    """Chain for structured decision making."""
    
    def __init__(self):
        self.sgr = SGRRunnable(schema_type="decision", budget="full")
    
    def invoke(self, input: str, **kwargs) -> Dict[str, Any]:
        """Make structured decision with reasoning."""
        return self.sgr.invoke(input, **kwargs)


# Convenience functions
def create_sgr_chain(
    schema_type: str = "auto",
    budget: str = "lite"
) -> SGRRunnable:
    """Create a basic SGR chain."""
    return SGRRunnable(schema_type=schema_type, budget=budget)


def wrap_chain_with_sgr(
    chain: Runnable,
    pre_analysis: bool = True,
    post_analysis: bool = True
) -> SGRChainWrapper:
    """Wrap existing chain with SGR analysis."""
    return SGRChainWrapper(
        chain,
        sgr_config={
            "pre_analysis": pre_analysis,
            "post_analysis": post_analysis
        }
    )


# Example usage
if __name__ == "__main__":
    # Example 1: Direct SGR analysis
    analysis_chain = SGRAnalysisChain(verbose=True)
    result = analysis_chain.invoke("How can I optimize my Python web API for better performance?")
    
    # Example 2: Wrap existing chain
    from langchain.chains import LLMChain
    from langchain.llms import OpenAI
    
    # Assuming you have a chain
    # llm_chain = LLMChain(llm=OpenAI(), prompt=...)
    # wrapped_chain = wrap_chain_with_sgr(llm_chain)
    # result = wrapped_chain.arun("Your input here")