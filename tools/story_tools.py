"""
Tools for converting stories into meme prompts.
"""

import os
from typing import List, Dict, Any
from pydantic import BaseModel, Field

from langgraph.prebuilt import ToolNode


class StoryToMemePromptInput(BaseModel):
    """Input for converting a story to meme prompts."""
    story: str = Field(..., description="The story to convert into meme prompts")
    num_memes: int = Field(9, description="Number of meme prompts to generate")
    meme_style: str = Field("funny and insightful", description="Style of the memes to generate")
    word_limit: int = Field(20, description="Maximum number of words per meme text")


def create_meme_prompts(input_data: StoryToMemePromptInput) -> Dict[str, Any]:
    """
    Convert a story into a series of meme prompts for image generation.
    
    This function analyzes a story and breaks it down into a series of visual prompts
    that can tell the story sequentially through meme-style images.
    
    Args:
        input_data: Parameters including story, number of memes to generate, and style.
        
    Returns:
        Dictionary containing the meme prompts and analysis information.
    """
    # The actual implementation will be handled by the LLM in the agent
    # This is just a placeholder structure
    return {
        "success": True,
        "prompts": [f"Meme {i+1} prompt would be generated here" for i in range(input_data.num_memes)],
        "story_analysis": "Analysis of the story flow and key moments",
        "message": f"Generated {input_data.num_memes} meme prompts in {input_data.meme_style} style"
    }


# Create tool definitions for langgraph
story_tools = [
    {
        "type": "function",
        "function": {
            "name": "create_meme_prompts",
            "description": "Convert a story into a series of meme prompts for image generation",
            "parameters": StoryToMemePromptInput.model_json_schema()
        }
    }
]

# Create a ToolNode with the story tools
story_tool_node = ToolNode(tools=[create_meme_prompts])
