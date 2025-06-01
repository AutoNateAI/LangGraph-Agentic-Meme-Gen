from pathlib import Path
import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the project root to the Python path to import the service
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.openai_image_service import OpenAIImageService


def main():
    # Make sure to set the OPENAI_API_KEY environment variable or pass it directly
    # os.environ["OPENAI_API_KEY"] = "your-api-key-here"
    
    # Initialize the service
    image_service = OpenAIImageService()
    
    # Example 1: Generate a new image from a prompt
    output_dir = Path("generated_images")
    output_dir.mkdir(exist_ok=True)
    
    '''
    prompt = """
    A children's book drawing of a veterinarian using a stethoscope to 
    listen to the heartbeat of a baby otter. pixar animated.
    """
    
    output_path = output_dir / "otter.png"
    image_service.generate_image(
        prompt=prompt,
        output_path=str(output_path)
    )
    print(f"Generated image saved to: {output_path}")
    '''
    
    # Example 2: Edit/combine multiple images
    # This requires existing images to work with
    # Example 2: Edit/combine multiple images (commented out)
    # Uncomment and use once you have the source images
    
    # Define prompt for image editing
    prompt = "Generate a disney pixar animated image of a gift basket on a white background " \
            "labeled 'Relax & Unwind' with a ribbon and handwriting-like font, " \
            "containing all the items in the reference pictures."
    
    # Source image paths
    source_images = [
        "generated_images/otter.png", 
    ]
    
    # Check if all source images exist
    if all(Path(img).exists() for img in source_images):
        output_path = output_dir / "gift-basket.png"
        image_service.edit_image(
            prompt=prompt,
            image_paths=source_images,
            output_path=str(output_path)
        )
        print(f"Edited image saved to: {output_path}")
    else:
        print("One or more source images not found. Skipping image editing example.")
    


if __name__ == "__main__":
    main()
