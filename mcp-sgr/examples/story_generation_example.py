#!/usr/bin/env python3
"""Example of using SGR for story generation with structured outputs."""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

# Configure for local or cloud provider
USE_LOCAL = True  # Set to False for OpenAI

if USE_LOCAL:
    os.environ["OPENAI_API_BASE"] = "http://localhost:1234/v1"
    os.environ["OPENAI_API_KEY"] = "dummy"
    DEFAULT_MODEL = "local-model"  # Adjust based on your local model
else:
    # Make sure to set OPENAI_API_KEY environment variable
    DEFAULT_MODEL = "gpt-4o-mini"

from dotenv import load_dotenv
load_dotenv()

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.story_tools import (
    generate_story_outline,
    format_story_outline_markdown,
    generate_story_with_sgr
)
from src.utils.telemetry import TelemetryManager


async def example_basic_story():
    """Generate a basic story outline."""
    print("\n=== Example: Basic Story Generation ===\n")
    
    prompt = """
    A young archaeologist discovers an ancient map that leads to a hidden city 
    in the Amazon rainforest. The city holds secrets about a lost civilization 
    and their advanced technology, but a rival expedition is also searching for it.
    """
    
    telemetry = TelemetryManager(enabled=False)  # Disable telemetry for example
    
    print(f"Generating story from prompt: {prompt[:100]}...")
    start_time = datetime.now()
    
    try:
        story = await generate_story_outline(
            prompt=prompt,
            model_id=DEFAULT_MODEL,
            temperature=0.7,
            telemetry=telemetry
        )
        
        generation_time = (datetime.now() - start_time).total_seconds()
        print(f"\n✅ Story generated in {generation_time:.2f} seconds\n")
        
        # Display results
        print(f"Title: {story.title}")
        print(f"Genre: {story.genre}")
        print(f"Setting: {story.setting}")
        print(f"Tone: {story.tone}")
        print(f"\nCharacters: {len(story.characters)}")
        print(f"Plot Points: {len(story.plot_points)}")
        print(f"Chapters: {len(story.chapter_outlines)}")
        print(f"Themes: {len(story.themes)}")
        
        # Save as markdown
        markdown = format_story_outline_markdown(story)
        output_file = f"story_outline_{story.title.replace(' ', '_')}_{int(generation_time)}s.md"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(markdown)
        print(f"\n📄 Full outline saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


async def example_constrained_story():
    """Generate a story with specific constraints."""
    print("\n=== Example: Constrained Story Generation ===\n")
    
    prompt = "A cozy mystery set in a small bookshop where rare books keep disappearing"
    
    constraints = {
        "chapters": 8,
        "target_length": "novella",
        "content_rating": "PG",
        "include_romance_subplot": True,
        "narrator": "first_person"
    }
    
    print(f"Prompt: {prompt}")
    print(f"Constraints: {json.dumps(constraints, indent=2)}")
    
    start_time = datetime.now()
    
    try:
        result = await generate_story_with_sgr(
            prompt=prompt,
            constraints=constraints,
            provider="local" if USE_LOCAL else "openai",
            model_id=DEFAULT_MODEL,
            temperature=0.8
        )
        
        generation_time = (datetime.now() - start_time).total_seconds()
        print(f"\n✅ Story generated in {generation_time:.2f} seconds\n")
        
        story = result["outline"]
        print(f"Title: {story['title']}")
        print(f"Chapters: {len(story['chapter_outlines'])}")
        
        # Show first chapter outline
        if story['chapter_outlines']:
            first_chapter = story['chapter_outlines'][0]
            print(f"\nChapter 1: {first_chapter['title']}")
            print(f"Summary: {first_chapter['summary']}")
            print("Key Scenes:")
            for scene in first_chapter['key_scenes']:
                print(f"  - {scene}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


async def example_multi_genre_story():
    """Generate a complex multi-genre story."""
    print("\n=== Example: Multi-Genre Story Generation ===\n")
    
    prompt = """
    Create a story that blends science fiction and fantasy elements. 
    The protagonist is a quantum physicist who discovers that magic is real 
    and operates on quantum principles. They must navigate both the scientific 
    community and a hidden magical society while preventing a catastrophe that 
    threatens both worlds.
    """
    
    constraints = {
        "chapters": 15,
        "genres": ["science_fiction", "fantasy", "thriller"],
        "dual_magic_system": True,
        "multiple_povs": ["protagonist", "magical_mentor", "scientific_rival"],
        "target_audience": "young_adult"
    }
    
    print("Generating complex multi-genre story...")
    start_time = datetime.now()
    
    try:
        story = await generate_story_outline(
            prompt=prompt,
            constraints=constraints,
            model_id=DEFAULT_MODEL,
            temperature=0.85  # Higher for more creativity
        )
        
        generation_time = (datetime.now() - start_time).total_seconds()
        print(f"\n✅ Story generated in {generation_time:.2f} seconds\n")
        
        # Analyze themes
        print("Themes explored:")
        for theme in story.themes:
            print(f"- {theme.theme}: {theme.manifestation}")
        
        # Show character arcs
        print("\nCharacter Arcs:")
        for char in story.characters:
            print(f"- {char.name} ({char.role}): {char.arc}")
        
        # Save the outline
        markdown = format_story_outline_markdown(story)
        output_file = f"multi_genre_story_{int(generation_time)}s.md"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(markdown)
        print(f"\n📄 Full outline saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


async def main():
    """Run all examples."""
    print("🎭 SGR Story Generation Examples")
    print("=" * 50)
    
    # Check if using local or cloud
    provider_info = "Local LLM" if USE_LOCAL else "OpenAI API"
    print(f"Using: {provider_info} with model {DEFAULT_MODEL}")
    
    # Run examples
    await example_basic_story()
    await example_constrained_story()
    await example_multi_genre_story()
    
    print("\n✨ All examples completed!")


if __name__ == "__main__":
    asyncio.run(main())