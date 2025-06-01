"""
Meme Generator Agent using langgraph.

This agent takes a story input and generates a sequence of meme images
that tell the story in a visual and engaging way.
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Annotated, Any, Dict, List, Tuple, TypedDict, Literal

from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode, create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

import openai
from openai import OpenAI
from langchain_openai import ChatOpenAI

from tools.image_tools import image_tool_node, bulk_generate_images
from tools.story_tools import story_tool_node


# Define the agent state
class AgentState(TypedDict):
    # The original story to be converted into memes
    story: str
    # The current messages in the conversation
    messages: List[Dict[str, Any]]
    # The meme prompts extracted from the story
    meme_prompts: List[str]
    # Paths to the generated meme images
    image_paths: List[str]
    # Additional metadata about the generation process
    metadata: Dict[str, Any]
    # Status of the current task
    status: Literal["in_progress", "complete", "error"]
    # Optional error message if something went wrong
    error: str | None


# Define the initial state creator
def create_initial_state(story: str) -> AgentState:
    """
    Create the initial state for the meme generator agent.
    
    Args:
        story: The story to be converted into memes
    
    Returns:
        Initial agent state
    """
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return {
        "story": story,
        "messages": [{
            "role": "system",
            "content": """
                You are a creative meme generator assistant. Your task is to analyze stories
                and convert them into engaging visual memes. Each meme should include a short
                caption (15-20 words maximum) that helps tell the story in a funny and
                insightful way. The sequence of memes should capture the overall narrative
                arc of the story.
                
                Guidelines for creating good meme prompts:
                1. Each prompt should be specific and detailed about the visual scene
                2. Include style specification (e.g., 'pixar style', 'photorealistic')
                3. Mention any text that should appear on the meme
                4. Make sure the sequence flows well and tells a coherent story
            """
        }],
        "meme_prompts": [],
        "image_paths": [],
        "metadata": {
            "start_time": current_time,
            "story_length": len(story.split()),
        },
        "status": "in_progress",
        "error": None
    }


# Define the nodes for our agent graph
def analyze_story(state: AgentState) -> AgentState:
    """
    Analyze the story and break it down into key narrative points.
    
    Args:
        state: Current agent state
    
    Returns:
        Updated agent state with analysis
    """
    model = ChatOpenAI(model="gpt-4o")
    
    # Add messages to instruct the model to analyze the story
    messages = state["messages"] + [{
        "role": "user",
        "content": f"""
            Please analyze the following story and break it down into 9 key narrative points
            that would make good meme images. For each point:
            1. Identify the key moment, character interaction, or plot development
            2. Suggest a visual scene that captures this moment
            3. Create a short, funny caption (15-20 words maximum) that is moving and insightful
            4. Include style specification (e.g., 'pixar style', 'photorealistic')
            5. Mention any text that should appear on the meme
            6. Make sure the sequence flows well and tells a coherent story
            
            Story:
            {state['story']}
            
            Respond with a JSON structure that contains an array of exactly 9 meme prompts.
            Each prompt should have a detailed visual description and the text to appear on the meme.
            Format each prompt to work well with the OpenAI image generation API.
        """
    }]
    
    # Get the model response
    response = model.invoke(messages)
    
    # Extract the meme prompts from the response
    response_content = response.content
    
    # Try to parse the JSON from the response
    try:
        # Look for JSON-like structure in the response
        start_idx = response_content.find("[{\"")
        if start_idx == -1:
            start_idx = response_content.find("[{\n")
        if start_idx == -1:
            start_idx = response_content.find("[")
            
        end_idx = response_content.rfind("]")
        if start_idx >= 0 and end_idx > start_idx:
            json_content = response_content[start_idx:end_idx + 1]
            meme_data = json.loads(json_content)
            meme_prompts = []
            
            for meme in meme_data:
                # Extract visual description and caption
                visual = meme.get("visual", "")
                caption = meme.get("caption", "")
                
                # Format the prompt for image generation
                prompt = f"""
                Create a meme image with the following scene: {visual}
                
                The image should include the following text caption:
                \"{caption}\"
                
                Style: Cartoon meme style, vibrant colors, modern, humorous
                """
                meme_prompts.append(prompt)
        else:
            # Fallback: Try to extract 9 sections from the text
            parts = response_content.split("\n\n")
            meme_prompts = []
            for part in parts:
                if len(meme_prompts) >= 9:
                    break
                if len(part.strip()) > 20:  # Ensure it's substantive
                    meme_prompts.append(part)
            
            # If we don't have 9 prompts, fill with placeholders
            while len(meme_prompts) < 9:
                meme_prompts.append(f"Generic meme scene {len(meme_prompts) + 1}")
    
    except Exception as e:
        # If JSON parsing fails, update state with error
        return {
            **state,
            "status": "error",
            "error": f"Failed to parse meme prompts from model response: {str(e)}"
        }
    
    # Update state with the extracted meme prompts
    updated_state = {
        **state,
        "messages": state["messages"] + [{
            "role": "user",
            "content": "Please analyze this story and create 9 meme prompts."
        }, {
            "role": "assistant",
            "content": "I've analyzed the story and created 9 meme prompts that capture the narrative arc."
        }],
        "meme_prompts": meme_prompts,
        "metadata": {
            **state["metadata"],
            "analysis_complete": True,
            "num_prompts": len(meme_prompts)
        }
    }
    
    return updated_state


def generate_meme_images(state: AgentState) -> Dict[str, Any]:
    """
    Generate meme images based on the analyzed prompts.
    This node connects to the image_tool_node.
    
    Args:
        state: Current agent state with meme prompts
    
    Returns:
        Updated state with messages formatted for the ToolNode
    """
    # Check if we have meme prompts
    if not state["meme_prompts"]:
        return {
            "__end__": {
                **state,
                "status": "error",
                "error": "No meme prompts were generated from the story analysis."
            }
        }
    
    # Import necessary modules for AIMessage with tool calls
    from langchain_core.messages import AIMessage, ToolCall
    
    # Create a tool call for bulk_generate_images
    tool_call = ToolCall(
        id=f"call_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        name="bulk_generate_images",
        args={
            "input_data": {
                "prompts": state["meme_prompts"],
                "model": "gpt-image-1",
                "output_dir": None  # Use the default timestamped directory
            }
        }
    )
    
    # Create an AIMessage with the tool call
    ai_message = AIMessage(
        content="I'll generate meme images based on these prompts.",
        tool_calls=[tool_call]
    )
    
    # Update the state with messages formatted for ToolNode
    return {
        **state,
        "messages": state["messages"] + [ai_message]
    }


def process_image_results(state: AgentState, tool_messages: List[Any]) -> AgentState:
    """
    Process the results from the image generation.
    
    Args:
        state: Current agent state
        tool_messages: Tool messages returned from the ToolNode
    
    Returns:
        Updated agent state with image paths
    """
    # Extract result from tool messages
    if not tool_messages or len(tool_messages) == 0:
        return {
            **state,
            "status": "error",
            "error": "No response received from image generation tool"
        }
    
    # Parse the content of the tool message
    # The ToolNode returns a list of ToolMessages, but we only have one tool call
    tool_message = tool_messages[0]
    try:
        # Convert the message content to a dictionary
        import json
        result = json.loads(tool_message.content)
        
        # Check if image generation was successful
        if not result.get("success"):
            return {
                **state,
                "status": "error",
                "error": f"Failed to generate images: {result.get('error', 'Unknown error')}"
            }
        
        # Extract the successful image paths
        image_paths = result.get("output_paths", [])
        session_dir = result.get("session_dir")
    except (json.JSONDecodeError, AttributeError) as e:
        return {
            **state,
            "status": "error",
            "error": f"Failed to parse tool response: {str(e)}"
        }
    
    # Update state with the generated image paths
    updated_state = {
        **state,
        "image_paths": image_paths,
        "metadata": {
            **state["metadata"],
            "session_dir": session_dir,
            "images_generated": len(image_paths),
            "completion_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "status": "complete"
    }
    
    return updated_state


def handle_error(state: AgentState) -> AgentState:
    """
    Handle any errors in the agent workflow.
    
    Args:
        state: Current agent state with error
    
    Returns:
        Final agent state with error information
    """
    return {
        **state,
        "status": "error",
        "metadata": {
            **state["metadata"],
            "error_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    }


# Create the agent workflow
def create_meme_generator_agent():
    """
    Create a langgraph agent for meme generation.
    
    Returns:
        Compiled state graph for the meme generator agent
    """
    # Create an in-memory saver for checkpoints
    checkpointer = InMemorySaver()
    
    # Create the state graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("analyze_story", analyze_story)
    workflow.add_node("generate_meme_images", generate_meme_images)
    
    # Add process_image_results with proper mapping for the tool_messages parameter
    def process_image_results_mapping(state, messages=None):
        # This function maps the state and tool messages to the process_image_results function
        # Check if messages is None (happens when ToolNode returns messages in the state)
        if messages is None and "messages" in state:
            # Find tool messages in the state - they're usually the last messages
            # that were added by the ToolNode
            tool_messages = [
                msg for msg in state["messages"] 
                if hasattr(msg, "name") and msg.name == "bulk_generate_images"
            ]
            return process_image_results(state, tool_messages)
        elif messages is None:
            # If no messages found, pass an empty list
            return process_image_results(state, [])
        else:
            # If messages were passed directly, use them
            return process_image_results(state, messages)
    workflow.add_node("process_image_results", process_image_results_mapping)
    
    workflow.add_node("handle_error", handle_error)
    
    # Add the tool node for image generation
    workflow.add_node("image_tools", image_tool_node)
    
    # Define edges
    workflow.add_edge(START, "analyze_story")
    workflow.add_edge("analyze_story", "generate_meme_images")
    workflow.add_edge("generate_meme_images", "image_tools")
    workflow.add_edge("image_tools", "process_image_results")
    workflow.add_edge("process_image_results", END)
    
    # Add error handling
    workflow.add_conditional_edges(
        "analyze_story",
        lambda state: "handle_error" if state.get("error") else "generate_meme_images"
    )
    
    # Add conditional edge for potential errors in process_image_results
    workflow.add_conditional_edges(
        "process_image_results",
        lambda state: "handle_error" if state.get("error") else END
    )
    
    workflow.add_edge("handle_error", END)
    
    # Compile the graph with the checkpointer
    return workflow.compile(checkpointer=checkpointer)


# Main function to run the agent
def generate_memes_from_story(story: str) -> Dict[str, Any]:
    """
    Generate a sequence of memes from a story.
    
    Args:
        story: The story to convert into memes
    
    Returns:
        Dictionary with the result of meme generation
    """
    # Create the agent
    agent = create_meme_generator_agent()
    
    # Create the initial state
    initial_state = create_initial_state(story)
    
    # Create a unique thread ID for this story generation session
    thread_id = f"meme_gen_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # Run the agent with checkpoint configuration
    result = agent.invoke(
        {
            "story": story,
            "messages": initial_state["messages"],
            "meme_prompts": [],
            "image_paths": [],
            "metadata": initial_state["metadata"],
            "status": "in_progress",
            "error": None
        },
        {"configurable": {"thread_id": thread_id}}
    )
    
    return {
        "success": result["status"] == "complete",
        "error": result.get("error"),
        "image_paths": result.get("image_paths", []),
        "session_dir": result.get("metadata", {}).get("session_dir"),
        "analysis": {
            "meme_prompts": result.get("meme_prompts", []),
            "story_length": result.get("metadata", {}).get("story_length"),
            "num_images": len(result.get("image_paths", []))
        }
    }


if __name__ == "__main__":
    # Example usage
    story = """
    Once upon a time, a software developer named Alex was struggling with a particularly difficult bug. 
    After three days of non-stop work, Alex finally discovered the issue was just a missing semicolon. 
    Frustrated but relieved, Alex went to grab coffee, only to spill it on the laptop. 
    In a panic, Alex tried to save the code before the computer died, frantically hitting Ctrl+S repeatedly.
    By some miracle, not only did the computer survive, but the coffee somehow fixed another bug 
    that had been plaguing the project for weeks. When Alex's boss asked how they solved both problems so quickly, 
    Alex just smiled and said, "It's all about having the right development environment."
    The team laughed, but from then on, a cup of coffee always sat ready by Alex's desk—just in case another bug needed fixing.
    """
    
    result = generate_memes_from_story(story)
    
    if result["success"]:
        print(f"Successfully generated {len(result['image_paths'])} meme images")
        print(f"Images saved in: {result['session_dir']}")
    else:
        print(f"Failed to generate memes: {result['error']}")
