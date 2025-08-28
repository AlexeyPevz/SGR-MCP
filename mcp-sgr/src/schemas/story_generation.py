"""Story generation schema using SGR patterns from meeting_summary example."""

from typing import List, Optional
from pydantic import BaseModel, Field
from .base import BaseSchema, SchemaField, ValidationResult


class CharacterItem(BaseModel):
    """Individual character using SGR Cycle pattern."""
    name: str = Field(description="Character name")
    role: str = Field(description="Character's role in the story (protagonist, antagonist, supporting)")
    description: str = Field(description="Brief character description including key traits")
    arc: str = Field(description="Character's development throughout the story")


class PlotPoint(BaseModel):
    """Individual plot point using SGR Cycle pattern."""
    event: str = Field(description="What happens in this plot point")
    impact: str = Field(description="How this event affects the story and characters")
    chapter: int = Field(description="Chapter number where this occurs", ge=1)


class ThemeElement(BaseModel):
    """Individual theme element using SGR Cycle pattern."""
    theme: str = Field(description="The thematic element")
    manifestation: str = Field(description="How this theme manifests in the story")


class ChapterOutline(BaseModel):
    """Individual chapter outline using SGR Cycle pattern."""
    number: int = Field(description="Chapter number", ge=1)
    title: str = Field(description="Chapter title")
    summary: str = Field(description="Brief summary of chapter events")
    key_scenes: List[str] = Field(
        description="List of key scenes in this chapter",
        min_items=2,
        max_items=5
    )


class StoryResponse(BaseModel):
    """Structured output for story generation with SGR Cycle patterns.
    
    Fields are ordered to guide the model's reasoning (SGR Cascade + Cycle):
    1) Generate character list (SGR Cycle)
    2) Generate plot points (SGR Cycle)  
    3) Extract themes (SGR Cycle)
    4) Create chapter outlines (SGR Cycle)
    5) Generate story metadata
    """
    
    # Core story elements using SGR Cycle
    characters: List[CharacterItem] = Field(
        description="List of main characters in the story",
        min_items=2,
        max_items=6
    )
    
    plot_points: List[PlotPoint] = Field(
        description="Major plot points that drive the narrative",
        min_items=5,
        max_items=12
    )
    
    themes: List[ThemeElement] = Field(
        description="Central themes explored in the story",
        min_items=1,
        max_items=4
    )
    
    chapter_outlines: List[ChapterOutline] = Field(
        description="Outline for each chapter",
        min_items=3,
        max_items=20
    )
    
    # Story metadata (generated after structured elements)
    title: str = Field(description="Story title (max 60 chars)", max_length=60)
    genre: str = Field(description="Primary genre of the story")
    setting: str = Field(description="Time and place where the story occurs")
    tone: str = Field(description="Overall tone/mood of the story")
    target_audience: str = Field(description="Intended audience for the story")
    
    # Optional elements
    subplots: Optional[List[str]] = Field(
        default=None,
        description="Secondary storylines that complement the main plot"
    )
    
    writing_style_notes: Optional[str] = Field(
        default=None,
        description="Specific style guidelines for writing this story"
    )


class StoryGenerationSchema(BaseSchema):
    """Schema for generating structured story outlines using SGR patterns."""
    
    def get_fields(self) -> List[SchemaField]:
        """Get schema fields for story generation."""
        return [
            SchemaField(
                name="prompt",
                type="string",
                required=True,
                description="Initial story prompt or concept"
            ),
            SchemaField(
                name="constraints",
                type="object",
                required=False,
                description="Optional constraints (word count, content rating, etc.)"
            )
        ]
    
    def get_description(self) -> str:
        """Get schema description."""
        return (
            "Generate a complete story outline using SGR Cycle patterns. "
            "This schema creates structured lists of characters, plot points, "
            "themes, and chapter outlines in a single LLM call."
        )
    
    def get_examples(self) -> List[dict]:
        """Get example story generation requests."""
        return [
            {
                "prompt": "A mystery story about a detective investigating disappearances in a small coastal town",
                "constraints": {
                    "chapters": 8,
                    "target_length": "novella",
                    "content_rating": "PG-13"
                }
            },
            {
                "prompt": "A sci-fi adventure about first contact with an alien civilization on Mars",
                "constraints": {
                    "chapters": 12,
                    "hard_sci_fi": True,
                    "multiple_povs": True
                }
            }
        ]
    
    def validate(self, data: dict) -> ValidationResult:
        """Validate story generation data."""
        errors = []
        warnings = []
        
        # Validate required fields
        if "prompt" not in data:
            errors.append("Story prompt is required")
        elif len(data["prompt"]) < 10:
            errors.append("Story prompt too short (min 10 characters)")
        
        # Validate constraints if provided
        if "constraints" in data:
            constraints = data["constraints"]
            if "chapters" in constraints:
                if not isinstance(constraints["chapters"], int) or constraints["chapters"] < 1:
                    errors.append("Chapter count must be a positive integer")
                elif constraints["chapters"] > 50:
                    warnings.append("Very long stories (>50 chapters) may be challenging to maintain consistency")
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            confidence=0.9 if len(errors) == 0 else 0.3
        )


# System message for story generation (following meeting_summary pattern)
STORY_GENERATION_SYSTEM_MESSAGE = """
You are a professional story writer and narrative designer.
Generate a complete story outline based on the provided prompt.
Do not add plot elements or characters that weren't implied by the prompt.
Ensure all story elements are coherent and contribute to the overall narrative.
"""

def get_story_generation_prompt(include_instructions: bool = True) -> str:
    """Get the full prompt for story generation."""
    base_prompt = STORY_GENERATION_SYSTEM_MESSAGE
    
    if include_instructions:
        base_prompt += """

Fill all fields in the StoryResponse schema in the specified order:

characters: Main characters driving the story. Include their role, key traits, and character arc.
plot_points: Major events that move the story forward. Include what happens and its impact.
themes: Central themes explored throughout the narrative and how they manifest.
chapter_outlines: Brief outline for each chapter with key scenes.
title: Compelling story title (max 60 characters).
genre: Primary genre classification.
setting: When and where the story takes place.
tone: Overall mood and atmosphere.
target_audience: Intended readership.
subplots: Secondary storylines if applicable.
writing_style_notes: Specific style guidelines if needed.
"""
    
    return base_prompt