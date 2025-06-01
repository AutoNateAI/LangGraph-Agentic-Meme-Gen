"""
Tools for interacting with the OpenAI Image Generation API through a langgraph agent.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, TypedDict, Annotated

from langgraph.graph import END
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from services.openai_image_service import OpenAIImageService


# Initialize the image service once
image_service = OpenAIImageService()


# Input models for the tools
class GenerateImageInput(BaseModel):
    """Input for single image generation tool."""
    prompt: str = Field(..., description="Text description of the image to generate")
    output_path: Optional[str] = Field(None, description="Optional path to save the image")
    model: str = Field("gpt-image-1", description="OpenAI model to use for generation")

class EditImageInput(BaseModel):
    """Input for image editing tool."""
    prompt: str = Field(..., description="Text description of how to edit the image")
    image_paths: List[str] = Field(..., description="List of paths to source images")
    output_path: Optional[str] = Field(None, description="Optional path to save the edited image")
    model: str = Field("gpt-image-1", description="OpenAI model to use for editing")
    
class BulkGenerateImagesInput(BaseModel):
    """Input for bulk image generation tool."""
    prompts: List[str] = Field(..., description="List of text descriptions for images to generate")
    output_dir: Optional[str] = Field(None, description="Optional directory to save generated images")
    model: str = Field("gpt-image-1", description="OpenAI model to use for generation")
    
class BulkEditImagesInput(BaseModel):
    """Input for bulk image editing tool."""
    prompts: List[str] = Field(..., description="List of text descriptions for image edits")
    image_paths_list: List[List[str]] = Field(..., description="List of lists of paths to source images")
    output_dir: Optional[str] = Field(None, description="Optional directory to save edited images")
    model: str = Field("gpt-image-1", description="OpenAI model to use for editing")

class StoryToMemePromptInput(BaseModel):
    """Input for converting a story to meme prompts."""
    story: str = Field(..., description="The story to convert into meme prompts")
    num_memes: int = Field(9, description="Number of meme prompts to generate")
    meme_style: str = Field("funny and insightful", description="Style of the memes to generate")


# Tool functions
def generate_image(input_data: GenerateImageInput) -> Dict[str, Any]:
    """
    Generate a single image from a text prompt.
    
    Args:
        input_data: Generation parameters including prompt, model and output path.
        
    Returns:
        Dictionary containing the operation result including output path.
    """
    output_path = input_data.output_path
    
    # If no output path is provided, create one
    if not output_path:
        os.makedirs("generated_images", exist_ok=True)
        output_path = f"generated_images/image_{int(time.time())}.png"
    
    # Generate the image
    result = image_service.generate_image(
        prompt=input_data.prompt,
        model=input_data.model,
        output_path=output_path
    )
    
    return {
        "success": True,
        "output_path": output_path,
        "message": f"Image generated and saved to {output_path}"
    }

def edit_image(input_data: EditImageInput) -> Dict[str, Any]:
    """
    Edit or combine multiple images based on a text prompt.
    
    Args:
        input_data: Editing parameters including prompt, source images, model, and output path.
        
    Returns:
        Dictionary containing the operation result including output path.
    """
    output_path = input_data.output_path
    
    # If no output path is provided, create one
    if not output_path:
        os.makedirs("generated_images", exist_ok=True)
        output_path = f"generated_images/edited_{int(time.time())}.png"
    
    # Check if all source images exist
    for path in input_data.image_paths:
        if not os.path.exists(path):
            return {
                "success": False,
                "error": f"Source image not found: {path}"
            }
    
    # Edit the images
    result = image_service.edit_image(
        prompt=input_data.prompt,
        image_paths=input_data.image_paths,
        model=input_data.model,
        output_path=output_path
    )
    
    return {
        "success": True,
        "output_path": output_path,
        "message": f"Image edited and saved to {output_path}"
    }

def bulk_generate_images(input_data: BulkGenerateImagesInput) -> Dict[str, Any]:
    """
    Generate multiple images in parallel from a list of prompts.
    
    Args:
        input_data: Parameters for bulk generation including prompts, model, and output directory.
        
    Returns:
        Dictionary containing the operation results for all prompts.
    """
    output_dir = input_data.output_dir
    
    # Convert string path to Path object if provided
    output_dir_path = Path(output_dir) if output_dir else None
    
    # Generate images in bulk
    results = image_service.bulk_generate_images(
        prompts=input_data.prompts,
        model=input_data.model,
        output_dir=output_dir_path
    )
    
    # Extract successful paths
    successful_paths = [result["output_path"] for result in results if result["success"]]
    
    return {
        "success": all(result["success"] for result in results),
        "results": results,
        "output_paths": successful_paths,
        "session_dir": str(output_dir_path) if output_dir_path else str(results[0]["output_path"]).rsplit("/", 1)[0] if results and results[0]["success"] else None,
        "message": f"Generated {len(successful_paths)} images out of {len(input_data.prompts)} requested"
    }

def bulk_edit_images(input_data: BulkEditImagesInput) -> Dict[str, Any]:
    """
    Edit multiple sets of images in parallel based on prompts.
    
    Args:
        input_data: Parameters for bulk editing including prompts, source images, model, and output directory.
        
    Returns:
        Dictionary containing the operation results for all edits.
    """
    output_dir = input_data.output_dir
    
    # Convert string path to Path object if provided
    output_dir_path = Path(output_dir) if output_dir else None
    
    # Validate that image paths exist
    for i, paths in enumerate(input_data.image_paths_list):
        for path in paths:
            if not os.path.exists(path):
                return {
                    "success": False,
                    "error": f"Source image not found at prompt index {i}: {path}"
                }
    
    # Edit images in bulk
    results = image_service.bulk_edit_images(
        prompts=input_data.prompts,
        image_paths_list=input_data.image_paths_list,
        model=input_data.model,
        output_dir=output_dir_path
    )
    
    # Extract successful paths
    successful_paths = [result["output_path"] for result in results if result["success"]]
    
    return {
        "success": all(result["success"] for result in results),
        "results": results,
        "output_paths": successful_paths,
        "session_dir": str(output_dir_path) if output_dir_path else str(results[0]["output_path"]).rsplit("/", 1)[0] if results and results[0]["success"] else None,
        "message": f"Edited {len(successful_paths)} images out of {len(input_data.prompts)} requested"
    }

# Create tool definitions for langgraph
tools = [
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "Generate a single image from a text prompt",
            "parameters": GenerateImageInput.model_json_schema()
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_image",
            "description": "Edit or combine multiple images based on a text prompt",
            "parameters": EditImageInput.model_json_schema()
        }
    },
    {
        "type": "function",
        "function": {
            "name": "bulk_generate_images",
            "description": "Generate multiple images in parallel from a list of prompts",
            "parameters": BulkGenerateImagesInput.model_json_schema()
        }
    },
    {
        "type": "function",
        "function": {
            "name": "bulk_edit_images",
            "description": "Edit multiple sets of images in parallel based on prompts",
            "parameters": BulkEditImagesInput.model_json_schema()
        }
    }
]

# Create a ToolNode with the image tools
image_tool_node = ToolNode(tools=[
    generate_image,
    edit_image,
    bulk_generate_images,
    bulk_edit_images
])
