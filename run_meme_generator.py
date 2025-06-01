#!/usr/bin/env python
"""
Command-line interface for the meme generator agent.
Allows users to generate meme sequences from stories provided via command line or text file.
"""

import os
import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Make sure required packages are available
try:
    import langchain_openai
    import langgraph
except ImportError:
    print("Error: Required packages are missing. Please run:")
    print("    pip install langchain-openai langgraph")
    sys.exit(1)

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from agent import generate_memes_from_story


def main():
    """
    Parse command line arguments and run the meme generator agent.
    """
    parser = argparse.ArgumentParser(description="Generate meme sequences from stories using OpenAI Image API")
    
    # Input options - mutually exclusive
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--story", "-s", type=str, help="Story text to convert into memes")
    input_group.add_argument("--file", "-f", type=str, help="Path to a text file containing the story")
    
    # Optional arguments
    parser.add_argument("--output", "-o", type=str, help="Custom output directory for generated memes")
    parser.add_argument("--display", "-d", action="store_true", help="Display the generated memes (requires PIL)")
    
    args = parser.parse_args()
    
    # Get story text
    story_text = ""
    if args.story:
        story_text = args.story
    elif args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                story_text = f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            sys.exit(1)
    
    # Check for OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please set your OpenAI API key in the .env file or as an environment variable.")
        sys.exit(1)
    
    print("🎬 Generating meme sequence from your story...")
    print("📝 Analyzing narrative structure...")
    
    # Generate the memes
    result = generate_memes_from_story(story_text)
    
    if result["success"]:
        num_images = len(result["image_paths"])
        print(f"✅ Successfully generated {num_images} meme images!")
        print(f"📁 Images saved in: {result['session_dir']}")
        
        # List the generated images
        print("\n📊 Generated memes:")
        for i, path in enumerate(result["image_paths"]):
            print(f"  {i+1}. {Path(path).name}")
        
        # Display images if requested
        if args.display:
            try:
                from PIL import Image
                import matplotlib.pyplot as plt
                
                print("\n🖼️ Displaying images...")
                fig, axes = plt.subplots(3, 3, figsize=(15, 15))
                axes = axes.flatten()
                
                for i, path in enumerate(result["image_paths"]):
                    if i < 9:  # Limit to 9 images for the grid
                        img = Image.open(path)
                        axes[i].imshow(img)
                        axes[i].set_title(f"Meme {i+1}")
                        axes[i].axis("off")
                
                plt.tight_layout()
                plt.show()
            except ImportError:
                print("Could not display images. Please install PIL and matplotlib:")
                print("pip install pillow matplotlib")
    else:
        print(f"❌ Failed to generate memes: {result['error']}")


if __name__ == "__main__":
    main()
