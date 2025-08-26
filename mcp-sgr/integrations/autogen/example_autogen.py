"""Example of using MCP-SGR with AutoGen.

This example shows how to enhance AutoGen agents with structured reasoning.
"""

import asyncio

try:
	from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
	from autogen_agentchat.teams import RoundRobinGroupChat as GroupChat
except Exception as _e:
	raise SystemExit("AutoGen packages not installed. Install pyautogen>=0.10.0.")

# Import SGR integration
from sgr_autogen import SGRAgent, create_sgr_assistant, wrap_autogen_agent, SGRGroupChatManager


def example_1_basic_sgr_agent():
    """Example 1: Create SGR-enhanced agent."""
    print("=== Example 1: Basic SGR Agent ===\n")
    
    # Create SGR-enhanced assistant
    sgr_assistant = create_sgr_assistant(
        name="SGR_Assistant",
        system_message="""You are a helpful AI assistant enhanced with structured reasoning.
        Always analyze problems systematically before providing solutions.""",
        schema_type="analysis",  # Use analysis schema
        llm_config={"model": "gpt-4"}
    )
    
    # Create user proxy
    user_proxy = UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0
    )
    
    # Start conversation
    user_proxy.initiate_chat(
        sgr_assistant,
        message="How can I optimize the performance of my Python web application?"
    )


def example_2_wrap_existing_agent():
    """Example 2: Wrap existing AutoGen agent with SGR."""
    print("\n=== Example 2: Wrap Existing Agent ===\n")
    
    # Create standard AutoGen assistant
    standard_assistant = AssistantAgent(
        name="Standard_Assistant",
        system_message="You are a coding assistant.",
        llm_config={"model": "gpt-3.5-turbo"}
    )
    
    # Wrap with SGR capabilities
    enhanced_assistant = wrap_autogen_agent(
        agent=standard_assistant,
        schema_type="code_generation",
        budget="full"  # Use full analysis for code generation
    )
    
    # Create user proxy
    user_proxy = UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        code_execution_config={"use_docker": False}
    )
    
    # Start conversation
    user_proxy.initiate_chat(
        enhanced_assistant,
        message="Write a Python function to calculate Fibonacci numbers efficiently"
    )


def example_3_group_chat_with_sgr():
    """Example 3: Group chat with SGR analysis."""
    print("\n=== Example 3: Group Chat with SGR ===\n")
    
    # Create multiple agents with different SGR schemas
    analyst = create_sgr_assistant(
        name="Analyst",
        system_message="You analyze problems and identify key issues.",
        schema_type="analysis"
    )
    
    planner = create_sgr_assistant(
        name="Planner",
        system_message="You create detailed plans based on analysis.",
        schema_type="planning"
    )
    
    decision_maker = create_sgr_assistant(
        name="DecisionMaker",
        system_message="You make final decisions based on analysis and plans.",
        schema_type="decision"
    )
    
    user_proxy = UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0
    )
    
    # Create group chat
    agents = [user_proxy, analyst, planner, decision_maker]
    groupchat = GroupChat(
        agents=agents,
        messages=[],
        max_round=6
    )
    
    # Create SGR-enhanced group chat manager
    sgr_manager = SGRGroupChatManager(
        agents=agents,
        sgr_config={
            "analyze_interactions": True,
            "suggest_next_speaker": True
        }
    )
    
    # Standard manager for execution
    manager = groupchat  # Use the GroupChat instance directly for this example
    
    # Start group chat
    user_proxy.initiate_chat(
        manager,
        message="We need to design a scalable microservices architecture for an e-commerce platform"
    )
    
    # Analyze the conversation with SGR
    async def analyze():
        analysis = await sgr_manager.analyze_conversation(groupchat.messages)
        print("\n=== SGR Conversation Analysis ===")
        print(f"Confidence: {analysis.get('confidence', 0):.2f}")
        print(f"Key Insights: {analysis.get('reasoning', {})}")
        print(f"Suggested Actions: {analysis.get('suggested_actions', [])}")
    
    # Run analysis
    asyncio.run(analyze())


def example_4_custom_sgr_config():
    """Example 4: Custom SGR configuration."""
    print("\n=== Example 4: Custom SGR Configuration ===\n")
    
    # Create agent with custom SGR config
    custom_agent = SGRAgent(
        name="Custom_SGR_Agent",
        system_message="You are an AI assistant with custom reasoning configuration.",
        schema_type="auto",  # Auto-detect best schema
        budget="full",  # Always use full analysis
        sgr_config={
            "include_actions": True,
            "min_confidence": 0.8,
            "cache_results": True,
            "context": {
                "domain": "software_engineering",
                "expertise_level": "senior"
            }
        },
        llm_config={"model": "gpt-4"}
    )
    
    user_proxy = UserProxyAgent(
        name="User",
        human_input_mode="NEVER"
    )
    
    # Test with complex task
    user_proxy.initiate_chat(
        custom_agent,
        message="""Design a fault-tolerant distributed system for processing 
        real-time financial transactions with strict consistency requirements"""
    )


def example_5_async_sgr_operations():
    """Example 5: Async SGR operations for parallel processing."""
    print("\n=== Example 5: Async SGR Operations ===\n")
    
    async def parallel_sgr_analysis():
        # Create multiple SGR agents
        agents = [
            create_sgr_assistant(f"Agent_{i}", f"You are agent {i}", "analysis")
            for i in range(3)
        ]
        
        # Tasks to analyze
        tasks = [
            "Analyze database performance bottlenecks",
            "Review security vulnerabilities in the API",
            "Evaluate scalability of the current architecture"
        ]
        
        # Run analyses in parallel
        async def analyze_task(agent, task):
            # Simulate agent processing
            await agent._ensure_sgr_initialized()
            result = await apply_sgr_tool(
                arguments={
                    "task": task,
                    "schema_type": "analysis",
                    "budget": "full"
                },
                llm_client=agent.llm_client,
                cache_manager=agent.cache_manager,
                telemetry=agent.telemetry
            )
            return result
        
        # Execute all analyses concurrently
        results = await asyncio.gather(*[
            analyze_task(agent, task) 
            for agent, task in zip(agents, tasks)
        ])
        
        # Print results
        for i, (task, result) in enumerate(zip(tasks, results)):
            print(f"\nTask {i+1}: {task}")
            print(f"Confidence: {result.get('confidence', 0):.2f}")
            print(f"Key findings: {len(result.get('suggested_actions', []))} actions suggested")
    
    # Run async example
    asyncio.run(parallel_sgr_analysis())


if __name__ == "__main__":
    print("MCP-SGR AutoGen Integration Examples")
    print("=" * 50)
    
    # Run examples
    try:
        example_1_basic_sgr_agent()
    except Exception as e:
        print(f"Example 1 error: {e}")
    
    try:
        example_2_wrap_existing_agent()
    except Exception as e:
        print(f"Example 2 error: {e}")
    
    try:
        example_3_group_chat_with_sgr()
    except Exception as e:
        print(f"Example 3 error: {e}")
    
    try:
        example_4_custom_sgr_config()
    except Exception as e:
        print(f"Example 4 error: {e}")
    
    try:
        example_5_async_sgr_operations()
    except Exception as e:
        print(f"Example 5 error: {e}")
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nNote: Install autogen with: pip install pyautogen")