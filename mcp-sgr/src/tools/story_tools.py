"""Story generation tools using structured outputs."""

import json
from typing import Any, Dict, List, Optional
from openai import OpenAI

from ..schemas.story_generation import (
    StoryResponse, 
    get_story_generation_prompt,
    StoryGenerationSchema
)
from ..utils.llm_client import LLMClient
from ..utils.telemetry import TelemetryManager


def generate_story_outline(
    prompt: str,
    constraints: Optional[Dict[str, Any]] = None,
    client: Optional[OpenAI] = None,
    model_id: str = "gpt-4o-mini",
    temperature: float = 0.7,
    telemetry: Optional[TelemetryManager] = None
) -> StoryResponse:
    """Generate a complete story outline using structured outputs.
    
    Args:
        prompt: Initial story concept or prompt
        constraints: Optional constraints (chapters, length, rating, etc.)
        client: OpenAI client instance (or compatible)
        model_id: Model to use for generation
        temperature: Sampling temperature (0.7 default for creativity)
        telemetry: Optional telemetry manager
        
    Returns:
        StoryResponse with complete story outline
    """
    if telemetry:
        telemetry.track_tool_use("generate_story_outline", {
            "model": model_id,
            "temperature": temperature,
            "has_constraints": constraints is not None
        })
    
    # Use provided client or create default
    if client is None:
        # For now, require client to be provided
        # TODO: Integrate with LLMClient for multi-backend support
        raise ValueError("OpenAI client must be provided")
    
    # Prepare the user message
    user_message = f"Story prompt: {prompt}"
    if constraints:
        user_message += f"\n\nConstraints: {json.dumps(constraints, indent=2)}"
    
    try:
        # Call LLM with structured output
        completion = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": get_story_generation_prompt(include_instructions=True)},
                {"role": "user", "content": user_message}
            ],
            response_format=StoryResponse,
            temperature=temperature,
            max_tokens=4000,
        )
        
        if not completion.choices or not completion.choices[0].message:
            raise ValueError("Empty LLM response")
        
        story_outline: StoryResponse = getattr(completion.choices[0].message, "parsed", None)
        if story_outline is None:
            raise ValueError("Parsed StoryResponse is missing")
        
        if telemetry:
            telemetry.track_tool_use("generate_story_outline_success", {
                "chapters": len(story_outline.chapter_outlines),
                "characters": len(story_outline.characters),
                "plot_points": len(story_outline.plot_points)
            })
        
        return story_outline
        
    except Exception as e:
        if telemetry:
            telemetry.track_error("generate_story_outline_error", str(e))
        raise


async def generate_story_with_sgr(
    prompt: str,
    constraints: Optional[Dict[str, Any]] = None,
    provider: str = "openai",
    model_id: Optional[str] = None,
    temperature: float = 0.7,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """Generate story using SGR with flexible provider support.
    
    Supports multiple LLM providers (OpenAI, local, custom endpoints).
    
    Args:
        prompt: Story prompt
        constraints: Optional generation constraints
        provider: LLM provider ("openai", "local", "custom")
        model_id: Model identifier
        temperature: Sampling temperature
        api_base: API endpoint (for local/custom providers)
        api_key: API key
        
    Returns:
        Dict with story outline and metadata
    """
    # Configure client based on provider
    if provider == "local" or api_base:
        client = OpenAI(
            base_url=api_base or "http://localhost:1234/v1",
            api_key=api_key or "dummy"
        )
        default_model = model_id or "local-model"
    else:
        client = OpenAI(api_key=api_key)
        default_model = model_id or "gpt-4o-mini"
    
    # Generate story outline
    story_outline = generate_story_outline(
        prompt=prompt,
        constraints=constraints,
        client=client,
        model_id=default_model,
        temperature=temperature
    )
    
    # Convert to dict format
    result = {
        "outline": story_outline.model_dump(),
        "metadata": {
            "provider": provider,
            "model": default_model,
            "temperature": temperature,
            "sgr_version": "1.0"
        }
    }
    
    return result


def format_story_outline_markdown(story: StoryResponse) -> str:
    """Format story outline as readable Markdown."""
    
    # Characters section with table
    characters_table = "| Name | Role | Description | Arc |\n"
    characters_table += "|------|------|-------------|-----|\n"
    for char in story.characters:
        characters_table += f"| {char.name} | {char.role} | {char.description} | {char.arc} |\n"
    
    # Plot points as timeline
    plot_timeline = ""
    for i, point in enumerate(story.plot_points, 1):
        plot_timeline += f"{i}. **Chapter {point.chapter}**: {point.event}\n"
        plot_timeline += f"   - *Impact*: {point.impact}\n\n"
    
    # Themes
    themes_section = ""
    for theme in story.themes:
        themes_section += f"- **{theme.theme}**: {theme.manifestation}\n"
    
    # Chapter outlines
    chapters_section = ""
    for chapter in story.chapter_outlines:
        chapters_section += f"### Chapter {chapter.number}: {chapter.title}\n\n"
        chapters_section += f"{chapter.summary}\n\n"
        chapters_section += "**Key Scenes:**\n"
        for scene in chapter.key_scenes:
            chapters_section += f"- {scene}\n"
        chapters_section += "\n"
    
    # Build complete document
    markdown = f"""# {story.title}

**Genre**: {story.genre}  
**Setting**: {story.setting}  
**Tone**: {story.tone}  
**Target Audience**: {story.target_audience}

## Characters

{characters_table}

## Plot Timeline

{plot_timeline}

## Themes

{themes_section}

## Chapter Outlines

{chapters_section}"""
    
    # Add optional sections
    if story.subplots:
        markdown += "\n## Subplots\n\n"
        for subplot in story.subplots:
            markdown += f"- {subplot}\n"
    
    if story.writing_style_notes:
        markdown += f"\n## Writing Style Notes\n\n{story.writing_style_notes}\n"
    
    return markdown