"""CrewAI integration for MCP-SGR.

Enhances CrewAI agents and crews with structured reasoning capabilities.
"""

import asyncio
from typing import Any, Dict, List, Optional, Union

# Note: Install crewai with: pip install crewai
try:
    from crewai import Agent, Task, Crew, Process
except ImportError:
    # Stub classes when crewai not installed
    class Agent:
        def __init__(self, *args, **kwargs):
            pass
    class Task:
        def __init__(self, *args, **kwargs):
            pass
    class Crew:
        def __init__(self, *args, **kwargs):
            pass
    class Process:
        sequential = "sequential"
        hierarchical = "hierarchical"

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.tools import apply_sgr_tool, wrap_agent_call_tool, enhance_prompt_tool
from src.utils.llm_client import LLMClient
from src.utils.cache import CacheManager
from src.utils.telemetry import TelemetryManager


class SGRAgent(Agent):
    """CrewAI Agent enhanced with SGR reasoning."""
    
    def __init__(
        self,
        role: str,
        goal: str,
        backstory: str,
        schema_type: str = "auto",
        budget: str = "lite",
        sgr_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Initialize SGR-enhanced CrewAI agent.
        
        Args:
            role: Agent's role
            goal: Agent's goal
            backstory: Agent's backstory
            schema_type: SGR schema type
            budget: SGR budget
            sgr_config: Additional SGR configuration
            **kwargs: Other Agent parameters
        """
        super().__init__(
            role=role,
            goal=goal,
            backstory=backstory,
            **kwargs
        )
        
        self.schema_type = schema_type
        self.budget = budget
        self.sgr_config = sgr_config or {}
        
        # Initialize SGR components
        self.llm_client = LLMClient()
        self.cache_manager = CacheManager()
        self.telemetry = TelemetryManager()
        self._sgr_initialized = False
        
        # Override execute method
        self._original_execute = self.execute
        self.execute = self._sgr_execute
    
    async def _ensure_sgr_initialized(self):
        """Ensure SGR components are initialized."""
        if not self._sgr_initialized:
            await self.cache_manager.initialize()
            await self.telemetry.initialize()
            self._sgr_initialized = True
    
    def _sgr_execute(self, task: Union[str, Task]) -> str:
        """Execute task with SGR enhancement.
        
        Args:
            task: Task to execute
            
        Returns:
            Task result with SGR reasoning
        """
        # Run async SGR enhancement in sync context
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self._execute_with_sgr(task))
        finally:
            loop.close()
    
    async def _execute_with_sgr(self, task: Union[str, Task]) -> str:
        """Execute task with SGR pre/post analysis."""
        await self._ensure_sgr_initialized()
        
        # Extract task description
        task_description = task if isinstance(task, str) else task.description
        
        # Pre-analysis with SGR
        pre_analysis = await apply_sgr_tool(
            arguments={
                "task": f"{self.role} needs to: {task_description}",
                "schema_type": self.schema_type,
                "budget": self.budget,
                "context": {
                    "agent_role": self.role,
                    "agent_goal": self.goal,
                    **self.sgr_config
                }
            },
            llm_client=self.llm_client,
            cache_manager=self.cache_manager,
            telemetry=self.telemetry
        )
        
        # Execute original task with enhanced context
        enhanced_task = self._enhance_task_with_insights(task, pre_analysis)
        result = self._original_execute(enhanced_task)
        
        # Post-analysis
        post_analysis = await self._post_analyze_result(task_description, result)
        
        # Return enhanced result
        return self._format_enhanced_result(result, pre_analysis, post_analysis)
    
    def _enhance_task_with_insights(
        self,
        task: Union[str, Task],
        analysis: Dict[str, Any]
    ) -> Union[str, Task]:
        """Enhance task with SGR insights.
        
        Args:
            task: Original task
            analysis: SGR analysis
            
        Returns:
            Enhanced task
        """
        insights = analysis.get("suggested_actions", [])
        
        if isinstance(task, str):
            return f"{task}\n\nKey considerations:\n" + "\n".join(f"- {i}" for i in insights[:3])
        else:
            # For Task objects, update description
            task.description += "\n\nKey considerations:\n" + "\n".join(f"- {i}" for i in insights[:3])
            return task
    
    async def _post_analyze_result(
        self,
        task: str,
        result: str
    ) -> Dict[str, Any]:
        """Analyze task execution result.
        
        Args:
            task: Original task
            result: Execution result
            
        Returns:
            Post-analysis
        """
        analysis_task = f"""Analyze this task completion:
        
Task: {task}
Result: {result}

Evaluate completeness, quality, and suggest improvements.
"""
        
        return await apply_sgr_tool(
            arguments={
                "task": analysis_task,
                "schema_type": "analysis",
                "budget": "lite"
            },
            llm_client=self.llm_client,
            cache_manager=self.cache_manager,
            telemetry=self.telemetry
        )
    
    def _format_enhanced_result(
        self,
        result: str,
        pre_analysis: Dict[str, Any],
        post_analysis: Dict[str, Any]
    ) -> str:
        """Format result with SGR enhancements.
        
        Args:
            result: Original result
            pre_analysis: Pre-execution analysis
            post_analysis: Post-execution analysis
            
        Returns:
            Enhanced result
        """
        confidence = post_analysis.get("confidence", 0)
        
        enhanced = result
        
        # Add confidence indicator if configured
        if self.sgr_config.get("show_confidence", True) and confidence < 0.7:
            enhanced = f"[Confidence: {confidence:.2f}]\n{enhanced}"
        
        # Add improvement suggestions if any
        suggestions = post_analysis.get("suggested_actions", [])
        if suggestions and self.sgr_config.get("show_suggestions", False):
            enhanced += "\n\nPotential improvements:\n" + "\n".join(f"- {s}" for s in suggestions[:2])
        
        return enhanced


class SGRCrew:
    """CrewAI Crew with SGR orchestration and analysis."""
    
    def __init__(
        self,
        agents: List[Agent],
        tasks: List[Task],
        process: Process = Process.sequential,
        sgr_orchestration: bool = True,
        **kwargs
    ):
        """Initialize SGR-enhanced Crew.
        
        Args:
            agents: List of agents
            tasks: List of tasks
            process: Execution process
            sgr_orchestration: Enable SGR orchestration
            **kwargs: Other Crew parameters
        """
        self.base_crew = Crew(
            agents=agents,
            tasks=tasks,
            process=process,
            **kwargs
        )
        
        self.sgr_orchestration = sgr_orchestration
        self.llm_client = LLMClient()
        self.cache_manager = CacheManager()
        self.telemetry = TelemetryManager()
    
    def kickoff(self) -> str:
        """Execute crew with SGR enhancements.
        
        Returns:
            Crew execution result
        """
        if not self.sgr_orchestration:
            return self.base_crew.kickoff()
        
        # Run with SGR orchestration
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self._kickoff_with_sgr())
        finally:
            loop.close()
    
    async def _kickoff_with_sgr(self) -> str:
        """Execute crew with SGR orchestration."""
        await self.cache_manager.initialize()
        await self.telemetry.initialize()
        
        # Pre-execution planning
        execution_plan = await self._create_execution_plan()
        
        # Execute with monitoring
        result = self.base_crew.kickoff()
        
        # Post-execution analysis
        analysis = await self._analyze_execution(result)
        
        # Return enhanced result
        return self._format_crew_result(result, execution_plan, analysis)
    
    async def _create_execution_plan(self) -> Dict[str, Any]:
        """Create execution plan using SGR.
        
        Returns:
            Execution plan
        """
        crew_description = f"""Crew composition:
Agents: {[a.role for a in self.base_crew.agents]}
Tasks: {[t.description[:50] + "..." for t in self.base_crew.tasks]}
Process: {self.base_crew.process}

Create an optimal execution plan.
"""
        
        return await apply_sgr_tool(
            arguments={
                "task": crew_description,
                "schema_type": "planning",
                "budget": "full"
            },
            llm_client=self.llm_client,
            cache_manager=self.cache_manager,
            telemetry=self.telemetry
        )
    
    async def _analyze_execution(self, result: str) -> Dict[str, Any]:
        """Analyze crew execution result.
        
        Args:
            result: Execution result
            
        Returns:
            Analysis
        """
        analysis_task = f"""Analyze this crew execution:
Result: {result[:500]}...

Evaluate team performance, task completion, and areas for improvement.
"""
        
        return await apply_sgr_tool(
            arguments={
                "task": analysis_task,
                "schema_type": "analysis",
                "budget": "lite"
            },
            llm_client=self.llm_client,
            cache_manager=self.cache_manager,
            telemetry=self.telemetry
        )
    
    def _format_crew_result(
        self,
        result: str,
        plan: Dict[str, Any],
        analysis: Dict[str, Any]
    ) -> str:
        """Format crew result with SGR insights.
        
        Args:
            result: Original result
            plan: Execution plan
            analysis: Execution analysis
            
        Returns:
            Enhanced result
        """
        return f"""{result}

---
SGR Crew Analysis:
- Execution confidence: {analysis.get('confidence', 0):.2f}
- Key insights: {', '.join(analysis.get('suggested_actions', [])[:3])}
"""


# Convenience functions
def create_sgr_agent(
    role: str,
    goal: str,
    backstory: str,
    schema_type: str = "auto",
    **kwargs
) -> SGRAgent:
    """Create SGR-enhanced CrewAI agent.
    
    Args:
        role: Agent role
        goal: Agent goal
        backstory: Agent backstory
        schema_type: SGR schema type
        **kwargs: Additional parameters
        
    Returns:
        SGR-enhanced agent
    """
    return SGRAgent(
        role=role,
        goal=goal,
        backstory=backstory,
        schema_type=schema_type,
        **kwargs
    )


def enhance_crew_with_sgr(crew: Crew) -> SGRCrew:
    """Enhance existing crew with SGR capabilities.
    
    Args:
        crew: CrewAI crew
        
    Returns:
        SGR-enhanced crew
    """
    return SGRCrew(
        agents=crew.agents,
        tasks=crew.tasks,
        process=crew.process,
        sgr_orchestration=True
    )


# Task enhancement utilities
async def enhance_task_description(task_description: str) -> str:
    """Enhance task description using SGR.
    
    Args:
        task_description: Original task description
        
    Returns:
        Enhanced description
    """
    llm_client = LLMClient()
    cache_manager = CacheManager()
    telemetry = TelemetryManager()
    
    await cache_manager.initialize()
    await telemetry.initialize()
    
    result = await enhance_prompt_tool(
        arguments={
            "original_prompt": task_description,
            "target_model": "gpt-4"
        },
        llm_client=llm_client,
        cache_manager=cache_manager,
        telemetry=telemetry
    )
    
    return result.get("enhanced_prompt", task_description)