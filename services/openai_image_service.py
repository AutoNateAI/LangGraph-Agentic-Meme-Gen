import base64
import os
import concurrent.futures
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union, Dict, Tuple, Any

from openai import OpenAI


class OpenAIImageService:
    """
    Service for interacting with OpenAI's image generation and editing APIs.
    """
    
    def __init__(self, api_key: Optional[str] = None, max_workers: int = 10):
        """
        Initialize the OpenAI client with an API key.
        
        Args:
            api_key: OpenAI API key. If not provided, it will be read from the OPENAI_API_KEY environment variable.
            max_workers: Maximum number of parallel workers for bulk operations, default is 10.
        """
        self.client = OpenAI(api_key=api_key)
        self.max_workers = min(max_workers, 10)  # Cap at 10 workers
    
    def generate_image(self, 
                      prompt: str, 
                      model: str = "gpt-image-1", 
                      output_path: Optional[str] = None) -> Union[str, bytes]:
        """
        Generate an image from a text prompt.
        
        Args:
            prompt: The text description of the image to generate.
            model: The OpenAI model to use for image generation.
            output_path: Optional path to save the generated image. If provided, the image
                         will be saved to this path and the path will be returned.
                         If not provided, the image bytes will be returned.
        
        Returns:
            If output_path is provided, returns the path where the image was saved.
            Otherwise, returns the raw image bytes.
        """
        result = self.client.images.generate(
            model=model,
            prompt=prompt
        )
        
        image_base64 = result.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)
        
        if output_path:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save the image to the specified path
            with open(output_path, "wb") as f:
                f.write(image_bytes)
            return output_path
        
        return image_bytes
    
    def edit_image(self, 
                  prompt: str, 
                  image_paths: List[str], 
                  model: str = "gpt-image-1",
                  output_path: Optional[str] = None) -> Union[str, bytes]:
        """
        Edit or combine multiple images based on a text prompt.
        
        Args:
            prompt: The text description of how to edit/combine the images.
            image_paths: List of paths to the images to use as base for editing.
            model: The OpenAI model to use for image editing.
            output_path: Optional path to save the edited image. If provided, the image
                         will be saved to this path and the path will be returned.
                         If not provided, the image bytes will be returned.
        
        Returns:
            If output_path is provided, returns the path where the image was saved.
            Otherwise, returns the raw image bytes.
        """
        # Open all image files
        images = [open(path, "rb") for path in image_paths]
        
        try:
            result = self.client.images.edit(
                model=model,
                image=images,
                prompt=prompt
            )
            
            image_base64 = result.data[0].b64_json
            image_bytes = base64.b64decode(image_base64)
            
            if output_path:
                # Ensure the directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Save the image to the specified path
                with open(output_path, "wb") as f:
                    f.write(image_bytes)
                return output_path
            
            return image_bytes
        
        finally:
            # Close all opened files
            for img in images:
                img.close()
                
    def _generate_session_directory(self) -> Path:
        """
        Generate a timestamped directory for storing a batch of images.
        
        Returns:
            Path object for the session directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = Path(f"generated_images/session_{timestamp}")
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir
    
    def _worker_generate_image(self, args: Tuple[str, str, int, str]) -> Dict[str, Any]:
        """
        Worker function to generate a single image in parallel processing.
        
        Args:
            args: Tuple containing (prompt, model, index, output_path)
            
        Returns:
            Dictionary with operation results
        """
        prompt, model, index, output_path = args
        try:
            result = self.generate_image(prompt=prompt, model=model, output_path=output_path)
            return {
                "index": index,
                "success": True,
                "output_path": output_path,
                "error": None
            }
        except Exception as e:
            return {
                "index": index,
                "success": False,
                "output_path": None,
                "error": str(e)
            }
    
    def _worker_edit_image(self, args: Tuple[str, List[str], int, str]) -> Dict[str, Any]:
        """
        Worker function to edit a single image in parallel processing.
        
        Args:
            args: Tuple containing (prompt, image_paths, index, output_path)
            
        Returns:
            Dictionary with operation results
        """
        prompt, image_paths, index, output_path = args
        try:
            result = self.edit_image(prompt=prompt, image_paths=image_paths, output_path=output_path)
            return {
                "index": index,
                "success": True,
                "output_path": output_path,
                "error": None
            }
        except Exception as e:
            return {
                "index": index,
                "success": False,
                "output_path": None,
                "error": str(e)
            }
    
    def bulk_generate_images(self, 
                           prompts: List[str], 
                           model: str = "gpt-image-1",
                           output_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
        """
        Generate multiple images in parallel from a list of prompts.
        
        Args:
            prompts: List of text descriptions for the images to generate.
            model: The OpenAI model to use for image generation.
            output_dir: Optional custom directory to save images. If not provided,
                        a timestamped session directory will be created.
                        
        Returns:
            List of dictionaries containing operation results for each prompt.
        """
        # Create a session directory if not provided
        session_dir = output_dir if output_dir else self._generate_session_directory()
        session_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        # Prepare tasks for parallel execution
        tasks = []
        for i, prompt in enumerate(prompts):
            # Create a filename with index for proper ordering
            output_path = str(session_dir / f"image_{i:03d}.png")
            tasks.append((prompt, model, i, output_path))
        
        # Execute tasks in parallel with a maximum number of workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks and collect futures
            future_to_task = {executor.submit(self._worker_generate_image, task): task for task in tasks}
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_task):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    task = future_to_task[future]
                    index = task[2]  # Index is at position 2 in the task tuple
                    results.append({
                        "index": index,
                        "success": False,
                        "output_path": None,
                        "error": str(exc)
                    })
        
        # Sort results by index for consistent ordering
        results.sort(key=lambda x: x["index"])
        return results
    
    def bulk_edit_images(self,
                        prompts: List[str],
                        image_paths_list: List[List[str]],
                        model: str = "gpt-image-1",
                        output_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
        """
        Edit multiple sets of images in parallel based on prompts.
        
        Args:
            prompts: List of text descriptions for the image edits.
            image_paths_list: List of lists, where each inner list contains paths to images to edit/combine.
            model: The OpenAI model to use for image editing.
            output_dir: Optional custom directory to save images. If not provided,
                        a timestamped session directory will be created.
                        
        Returns:
            List of dictionaries containing operation results for each set of images.
        """
        if len(prompts) != len(image_paths_list):
            raise ValueError("Number of prompts must match number of image path lists")
        
        # Create a session directory if not provided
        session_dir = output_dir if output_dir else self._generate_session_directory()
        session_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        # Prepare tasks for parallel execution
        tasks = []
        for i, (prompt, img_paths) in enumerate(zip(prompts, image_paths_list)):
            # Create a filename with index for proper ordering
            output_path = str(session_dir / f"edited_image_{i:03d}.png")
            tasks.append((prompt, img_paths, i, output_path))
        
        # Execute tasks in parallel with a maximum number of workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks and collect futures
            future_to_task = {executor.submit(self._worker_edit_image, task): task for task in tasks}
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_task):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    task = future_to_task[future]
                    index = task[2]  # Index is at position 2 in the task tuple
                    results.append({
                        "index": index,
                        "success": False,
                        "output_path": None,
                        "error": str(exc)
                    })
        
        # Sort results by index for consistent ordering
        results.sort(key=lambda x: x["index"])
        return results
