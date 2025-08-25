"""Example of using MCP-SGR with CrewAI.

This example shows how to enhance CrewAI agents and crews with structured reasoning.
"""

import asyncio
from crewai import Agent, Task, Crew, Process
from sgr_crewai import SGRAgent, SGRCrew, create_sgr_agent, enhance_task_description


def example_1_basic_sgr_crew():
    """Example 1: Basic crew with SGR agents."""
    print("=== Example 1: Basic SGR Crew ===\n")
    
    # Create SGR-enhanced agents
    researcher = create_sgr_agent(
        role="Research Analyst",
        goal="Conduct thorough research and analysis on technical topics",
        backstory="You are an expert researcher with deep knowledge in technology",
        schema_type="analysis",  # Use analysis schema for research
        verbose=True
    )
    
    writer = create_sgr_agent(
        role="Technical Writer", 
        goal="Create clear and comprehensive technical documentation",
        backstory="You are a skilled technical writer who explains complex topics simply",
        schema_type="summarization",  # Use summarization for writing
        verbose=True
    )
    
    reviewer = create_sgr_agent(
        role="Quality Reviewer",
        goal="Review and ensure high quality of deliverables",
        backstory="You are a meticulous reviewer who ensures accuracy and completeness",
        schema_type="decision",  # Use decision schema for reviews
        verbose=True
    )
    
    # Create tasks
    research_task = Task(
        description="Research best practices for microservices architecture",
        agent=researcher,
        expected_output="Comprehensive analysis of microservices best practices"
    )
    
    write_task = Task(
        description="Write a technical guide based on the research",
        agent=writer,
        expected_output="Clear technical guide with examples"
    )
    
    review_task = Task(
        description="Review the guide for accuracy and completeness",
        agent=reviewer,
        expected_output="Reviewed and approved guide with feedback"
    )
    
    # Create SGR crew
    crew = SGRCrew(
        agents=[researcher, writer, reviewer],
        tasks=[research_task, write_task, review_task],
        process=Process.sequential,
        sgr_orchestration=True  # Enable SGR orchestration
    )
    
    # Execute crew
    result = crew.kickoff()
    print(f"\nCrew Result:\n{result}")


def example_2_enhanced_existing_crew():
    """Example 2: Enhance existing CrewAI crew with SGR."""
    print("\n=== Example 2: Enhance Existing Crew ===\n")
    
    # Create standard CrewAI agents
    standard_developer = Agent(
        role="Software Developer",
        goal="Develop high-quality software solutions",
        backstory="Experienced developer with expertise in multiple languages"
    )
    
    standard_tester = Agent(
        role="QA Engineer",
        goal="Ensure software quality through comprehensive testing",
        backstory="Detail-oriented QA professional"
    )
    
    # Create standard crew
    standard_crew = Crew(
        agents=[standard_developer, standard_tester],
        tasks=[
            Task(
                description="Implement a REST API for user management",
                agent=standard_developer
            ),
            Task(
                description="Create comprehensive test cases for the API",
                agent=standard_tester
            )
        ],
        process=Process.sequential
    )
    
    # Enhance with SGR
    from sgr_crewai import enhance_crew_with_sgr
    sgr_crew = enhance_crew_with_sgr(standard_crew)
    
    # Execute enhanced crew
    result = sgr_crew.kickoff()
    print(f"\nEnhanced Crew Result:\n{result}")


def example_3_custom_sgr_configuration():
    """Example 3: Agents with custom SGR configuration."""
    print("\n=== Example 3: Custom SGR Configuration ===\n")
    
    # Create agent with custom SGR config
    architect = SGRAgent(
        role="Software Architect",
        goal="Design scalable and maintainable software architectures",
        backstory="Senior architect with 15+ years experience",
        schema_type="planning",  # Architecture needs planning
        budget="full",  # Use full analysis for architecture decisions
        sgr_config={
            "show_confidence": True,
            "show_suggestions": True,
            "context": {
                "domain": "enterprise_software",
                "constraints": ["scalability", "security", "maintainability"]
            }
        },
        verbose=True
    )
    
    # Create complex task
    architecture_task = Task(
        description="""Design a microservices architecture for an e-commerce platform that needs to:
        - Handle 1M+ daily active users
        - Process payments securely
        - Scale horizontally
        - Maintain 99.9% uptime""",
        agent=architect,
        expected_output="Detailed architecture design with diagrams and rationale"
    )
    
    # Single agent crew
    crew = Crew(
        agents=[architect],
        tasks=[architecture_task]
    )
    
    result = crew.kickoff()
    print(f"\nArchitecture Result:\n{result}")


def example_4_parallel_sgr_analysis():
    """Example 4: Parallel task execution with SGR analysis."""
    print("\n=== Example 4: Parallel SGR Analysis ===\n")
    
    # Create multiple analysts for parallel work
    security_analyst = create_sgr_agent(
        role="Security Analyst",
        goal="Identify and mitigate security vulnerabilities",
        backstory="Cybersecurity expert with pentesting experience",
        schema_type="analysis"
    )
    
    performance_analyst = create_sgr_agent(
        role="Performance Analyst",
        goal="Optimize system performance and identify bottlenecks",
        backstory="Performance engineering specialist",
        schema_type="analysis"
    )
    
    ux_analyst = create_sgr_agent(
        role="UX Analyst",
        goal="Evaluate and improve user experience",
        backstory="User experience researcher and designer",
        schema_type="analysis"
    )
    
    # Create parallel tasks
    tasks = [
        Task(
            description="Analyze security vulnerabilities in the current system",
            agent=security_analyst
        ),
        Task(
            description="Identify performance bottlenecks and optimization opportunities",
            agent=performance_analyst
        ),
        Task(
            description="Evaluate user experience and suggest improvements",
            agent=ux_analyst
        )
    ]
    
    # Create crew with hierarchical process for parallel execution
    crew = SGRCrew(
        agents=[security_analyst, performance_analyst, ux_analyst],
        tasks=tasks,
        process=Process.hierarchical,  # Allows parallel execution
        sgr_orchestration=True
    )
    
    result = crew.kickoff()
    print(f"\nParallel Analysis Result:\n{result}")


def example_5_task_enhancement():
    """Example 5: Enhance task descriptions with SGR."""
    print("\n=== Example 5: Task Enhancement ===\n")
    
    async def enhance_tasks():
        # Original simple task descriptions
        simple_tasks = [
            "Build a login system",
            "Create API documentation",
            "Setup CI/CD pipeline"
        ]
        
        print("Original tasks:")
        for task in simple_tasks:
            print(f"- {task}")
        
        print("\nEnhanced tasks:")
        # Enhance each task description
        for task in simple_tasks:
            enhanced = await enhance_task_description(task)
            print(f"\nOriginal: {task}")
            print(f"Enhanced: {enhanced[:200]}...")
    
    # Run async enhancement
    asyncio.run(enhance_tasks())


if __name__ == "__main__":
    print("MCP-SGR CrewAI Integration Examples")
    print("=" * 50)
    
    # Note about dependencies
    print("\nNote: Install crewai with: pip install crewai")
    print("Make sure MCP-SGR server is running for full functionality\n")
    
    # Run examples
    try:
        example_1_basic_sgr_crew()
    except Exception as e:
        print(f"Example 1 error: {e}")
    
    try:
        example_2_enhanced_existing_crew()
    except Exception as e:
        print(f"Example 2 error: {e}")
    
    try:
        example_3_custom_sgr_configuration()
    except Exception as e:
        print(f"Example 3 error: {e}")
    
    try:
        example_4_parallel_sgr_analysis()
    except Exception as e:
        print(f"Example 4 error: {e}")
    
    try:
        example_5_task_enhancement()
    except Exception as e:
        print(f"Example 5 error: {e}")
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nFor production use:")
    print("1. Configure LLM backends in .env")
    print("2. Enable caching for better performance")
    print("3. Use appropriate budget levels for your use case")