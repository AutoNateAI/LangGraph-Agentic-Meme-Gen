# Meme Generator with OpenAI Image Generation and LangGraph

This project provides a service for interacting with OpenAI's image generation and editing APIs, which can be used for creating memes and other image content. The project includes both standalone service components and a complete langgraph-based agent that can transform stories into sequences of meme images.

## OpenAI Image Service

The `OpenAIImageService` allows you to:

1. Generate new images from text prompts
2. Edit existing images or combine multiple images based on a text prompt

## Installation and Setup

### 1. Install dependencies

```bash
# Create a virtual environment (optional but recommended)
python -m venv venv
.\venv\Scripts\activate  # Windows
# OR
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Set up environment variables

Create a `.env` file in the project root directory with the following content:

```
# OpenAI API Key - Required for image generation
OPENAI_API_KEY=your-api-key-here

# Optional: OpenAI Organization ID
# OPENAI_ORG_ID=your-org-id

# Optional: Model customization
# OPENAI_IMAGE_MODEL=gpt-image-1
```

Alternatively, you can set these environment variables directly:

```bash
# Windows
set OPENAI_API_KEY=your-api-key-here

# Linux/Mac
export OPENAI_API_KEY=your-api-key-here
```

## Usage Examples

### Generate a New Image

```python
from services.openai_image_service import OpenAIImageService

# Initialize the service
image_service = OpenAIImageService()

# Generate an image from a prompt
prompt = "A children's book drawing of a veterinarian using a stethoscope to listen to the heartbeat of a baby otter."

# Save the generated image
image_service.generate_image(
    prompt=prompt,
    output_path="generated_images/otter.png"
)
```

### Edit or Combine Multiple Images

```python
from services.openai_image_service import OpenAIImageService

# Initialize the service
image_service = OpenAIImageService()

# Define a prompt for editing/combining images
prompt = "Generate a photorealistic image of a gift basket on a white background labeled 'Relax & Unwind' with a ribbon and handwriting-like font, containing all the items in the reference pictures."

# List of source image paths
source_images = [
    "source_images/body-lotion.png", 
    "source_images/bath-bomb.png",
    "source_images/incense-kit.png",
    "source_images/soap.png"
]

# Edit/combine the images based on the prompt
image_service.edit_image(
    prompt=prompt,
    image_paths=source_images,
    output_path="generated_images/gift-basket.png"
)
```

## Running the Example

The project includes an example script that demonstrates how to use the OpenAI image service:

```bash
python examples/image_generation_example.py
```

## LangGraph Meme Generator Agent

The project includes a langgraph-based agent that can:

1. Take a story input and analyze its narrative structure
2. Break the story down into 9 key moments
3. Generate meme prompts for each key moment
4. Use bulk image generation to create a sequence of meme images in parallel
5. Save all images in a timestamped session directory

### Agent Architecture

The agent uses a directed graph workflow with the following components:

- `analyze_story`: Processes the input story using LLM to extract key narrative points
- `generate_meme_images`: Prepares the image generation requests
- `image_tools`: Handles the parallel execution of image generation
- `process_image_results`: Collects and organizes the generated images
- Error handling nodes to manage any failures gracefully

### Running the Agent

You can run the meme generator agent using the provided command-line interface:

```bash
# From a story in a text file
python run_meme_generator.py --file path/to/story.txt

# From a story provided directly
python run_meme_generator.py --story "Your story text here..."

# Display the generated memes (requires matplotlib)
python run_meme_generator.py --file path/to/story.txt --display
```

## Example Story to Meme Sequence

Here's an example of how the agent processes a story:

1. **Input**: A short story about a software developer's experience with a bug
2. **Processing**: The agent identifies 9 key moments in the narrative
3. **Output**: 9 meme images with captions that tell the story visually

### Sample Command

```bash
python run_meme_generator.py --file examples/developer_story.txt --display
```

## Note on API Usage

Using the OpenAI image generation API requires an API key and will incur costs based on OpenAI's pricing model. Make sure to review the [OpenAI pricing page](https://openai.com/pricing) for the latest information on costs associated with image generation and editing.