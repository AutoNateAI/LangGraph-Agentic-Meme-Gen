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
python example.py
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

## Customizing the Agent for Different Genres/Niches

The meme generator agent can be customized to create content for different audiences, brands, or aesthetic preferences. The agent is currently configured for Tea Time Moments, an organization focused on women's empowerment through authentic connection and spiritual renewal.

To customize the agent for a different genre or niche, you'll need to modify two key prompt sections in the `agent.py` file:

### 1. System Prompt (in `create_initial_state` function)

This defines the overall purpose and style guidelines for the meme generation. Locate this section around line 52-63:

```python
"content": """
    You are a sophisticated meme creator for Tea Time Moments, an organization with 15+ years of empowering women through authentic connection and spiritual renewal. Your task is to analyze personal stories and convert them into elegant, meaningful visual content that resonates with affluent women seeking empowerment, spiritual connection, and community.

    Each meme should include a thoughtful caption (15-20 words maximum) that conveys depth, wisdom, and gentle inspiration rather than humor. The imagery should feel refined, warm, and inviting - like being welcomed to an intimate gathering of supportive women.
    
    Guidelines for creating Tea Time Moments meme prompts:
    1. Focus on elegant, sophisticated visuals that feature diverse women in moments of connection, reflection, or empowerment
    2. Use warm color palettes with gold, burgundy, and soft pastels that evoke tea time aesthetics
    3. Include meaningful, inspirational text that speaks to personal transformation
    4. Ensure the sequence feels like a journey of restoration, empowerment, and community
    5. Avoid overly casual or juvenile styling - aim for a polished, premium aesthetic
"""
```

### 2. Image Style Formatting (in `analyze_story` function)

This defines the specific style instructions for the image generation API. Locate this section around line 145-153:

```python
prompt = f"""
Create an elegant, inspirational illustrated image with the following scene: {visual}

The image should include the following thoughtful caption:
\"{caption}\"

Style: Sophisticated animation with a high-end illustrated feel - similar to premium motion graphics or upscale animated short films. Use warm color palette (burgundy, gold, earth tones) with elegant line work and tasteful design elements reminiscent of fine tea settings. The characters should be diverse women rendered in a refined, artistic illustration style (not cartoon or comic). The overall aesthetic should feel premium and aspirational while maintaining the warmth and approachability of illustration.
"""
```

### Customization Examples

#### For Tech Startup Content:

```python
# System Prompt
"content": """
    You are a tech-savvy meme creator for a cutting-edge software startup. Your task is to analyze stories and convert them into witty, intelligent visual content that resonates with developers, engineers, and tech professionals.

    Each meme should include a clever caption (15-20 words maximum) that demonstrates tech humor and insider knowledge. The imagery should feel modern, sleek, and slightly futuristic.
    
    Guidelines for creating tech meme prompts:
    1. Focus on visual scenarios that reference coding, debugging, deployment, or other tech processes
    2. Use color palettes associated with popular IDEs and tech brands (dark mode, blue accent colors)
    3. Include text that contains witty tech puns or references
    4. Ensure the sequence feels like it's telling a coherent tech story
    5. Appeal to a tech-savvy audience that appreciates intelligent humor
"""

# Image Style Formatting
prompt = f"""
Create a modern tech-themed meme with the following scene: {visual}

The image should include the following witty caption:
\"{caption}\"

Style: Modern digital illustration with a tech aesthetic - clean lines, subtle gradients, and minimalist design. Use a color palette inspired by dark mode interfaces with vibrant accent colors. The overall look should be sleek, slightly futuristic, and instantly recognizable to a tech audience.
"""
```

#### For Fitness Brand Content:

```python
# System Prompt
"content": """
    You are an energetic meme creator for a premium fitness brand. Your task is to analyze stories and convert them into motivational, high-energy visual content that resonates with fitness enthusiasts and people on their wellness journey.

    Each meme should include a powerful, motivational caption (15-20 words maximum) that inspires action and reinforces a growth mindset. The imagery should feel dynamic, bold, and empowering.
    
    Guidelines for creating fitness meme prompts:
    1. Focus on visual scenarios that demonstrate transformation, strength, and perseverance
    2. Use vibrant, high-contrast color palettes that convey energy and vitality
    3. Include text that motivates and challenges the viewer
    4. Ensure the sequence feels like a fitness journey from challenge to triumph
    5. Appeal to fitness enthusiasts across all levels while maintaining aspirational quality
"""

# Image Style Formatting
prompt = f"""
Create a dynamic fitness-themed image with the following scene: {visual}

The image should include the following motivational caption:
\"{caption}\"

Style: Bold, high-energy photorealistic style with dramatic lighting and vibrant colors. The visuals should have a premium fitness brand aesthetic - clean, powerful, and inspirational. Characters should appear strong, confident, and diverse, captured in dynamic moments of action or achievement.
"""
```