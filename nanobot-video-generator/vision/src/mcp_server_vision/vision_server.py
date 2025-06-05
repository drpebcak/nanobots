#!/usr/bin/env python3
"""
Vision MCP Server
A Model Context Protocol server that uploads images to OpenAI and returns descriptions.
"""

import os
import base64
import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from io import BytesIO

import httpx
from PIL import Image
from openai import AsyncOpenAI
from dotenv import load_dotenv

from fastmcp import FastMCP

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("Vision Server")

# Initialize OpenAI client
openai_client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

def encode_image_to_base64(image_path: str) -> str:
    """
    Encode an image file to base64 string.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string of the image
    """
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded_string
    except Exception as e:
        logger.error(f"Error encoding image: {e}")
        raise

def get_image_mime_type(image_path: str) -> str:
    """
    Get MIME type of an image file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        MIME type string
    """
    try:
        with Image.open(image_path) as img:
            format_to_mime = {
                'JPEG': 'image/jpeg',
                'PNG': 'image/png',
                'GIF': 'image/gif',
                'WEBP': 'image/webp',
                'BMP': 'image/bmp'
            }
            return format_to_mime.get(img.format, 'image/jpeg')
    except Exception as e:
        logger.error(f"Error getting image MIME type: {e}")
        return 'image/jpeg'  # Default fallback

@mcp.tool()
async def describe_image(
    image_path: str,
    prompt: Optional[str] = None,
    max_tokens: int = 300,
    detail: str = "auto"
) -> Dict[str, Any]:
    """
    Upload an image to OpenAI and get a description.
    
    Args:
        image_path: Path to the image file to analyze
        prompt: Optional custom prompt for the description (defaults to "Describe this image in detail")
        max_tokens: Maximum number of tokens in the response (default: 300)
        detail: Level of detail for image analysis ("low", "high", or "auto")
        
    Returns:
        Dictionary containing the image description and metadata
    """
    try:
        # Check if file exists
        if not os.path.exists(image_path):
            return {
                "error": f"Image file not found: {image_path}",
                "success": False
            }
        
        # Validate file is an image
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                format_name = img.format
        except Exception as e:
            return {
                "error": f"Invalid image file: {e}",
                "success": False
            }
        
        # Encode image to base64
        base64_image = encode_image_to_base64(image_path)
        mime_type = get_image_mime_type(image_path)
        
        # Default prompt if none provided
        if not prompt:
            prompt = "Describe this image in detail, including any text, objects, people, scenes, colors, and overall composition."
        
        # Create the message for OpenAI
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}",
                            "detail": detail
                        }
                    }
                ]
            }
        ]
        
        # Make request to OpenAI
        logger.info(f"Sending image to OpenAI for description: {image_path}")
        response = await openai_client.chat.completions.create(
            model="gpt-4o",  # Using GPT-4 with vision capabilities
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7
        )
        
        # Extract the description
        description = response.choices[0].message.content
        
        return {
            "success": True,
            "description": description,
            "image_path": image_path,
            "image_info": {
                "width": width,
                "height": height,
                "format": format_name,
                "mime_type": mime_type
            },
            "prompt_used": prompt,
            "tokens_used": response.usage.total_tokens if response.usage else None
        }
        
    except Exception as e:
        logger.error(f"Error describing image: {e}")
        return {
            "error": f"Failed to describe image: {str(e)}",
            "success": False
        }

@mcp.tool()
async def batch_describe_images(
    image_paths: List[str],
    prompt: Optional[str] = None,
    max_tokens: int = 300,
    detail: str = "auto"
) -> Dict[str, Any]:
    """
    Describe multiple images in batch.
    
    Args:
        image_paths: List of paths to image files
        prompt: Optional custom prompt for descriptions
        max_tokens: Maximum number of tokens per response
        detail: Level of detail for image analysis
        
    Returns:
        Dictionary containing results for all images
    """
    results = []
    
    for image_path in image_paths:
        result = await describe_image(image_path, prompt, max_tokens, detail)
        results.append(result)
    
    return {
        "success": True,
        "results": results,
        "total_images": len(image_paths)
    }

@mcp.tool()
async def compare_images(
    image_path1: str,
    image_path2: str,
    comparison_prompt: Optional[str] = None,
    max_tokens: int = 400
) -> Dict[str, Any]:
    """
    Compare two images and describe their differences and similarities.
    
    Args:
        image_path1: Path to the first image
        image_path2: Path to the second image
        comparison_prompt: Optional custom prompt for comparison
        max_tokens: Maximum number of tokens in the response
        
    Returns:
        Dictionary containing the comparison results
    """
    try:
        # Check if both files exist
        for path in [image_path1, image_path2]:
            if not os.path.exists(path):
                return {
                    "error": f"Image file not found: {path}",
                    "success": False
                }
        
        # Encode both images
        base64_image1 = encode_image_to_base64(image_path1)
        base64_image2 = encode_image_to_base64(image_path2)
        
        mime_type1 = get_image_mime_type(image_path1)
        mime_type2 = get_image_mime_type(image_path2)
        
        # Default comparison prompt
        if not comparison_prompt:
            comparison_prompt = "Compare these two images. Describe their similarities and differences in detail, including objects, composition, colors, style, and any other notable aspects."
        
        # Create message with both images
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": comparison_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type1};base64,{base64_image1}",
                            "detail": "auto"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type2};base64,{base64_image2}",
                            "detail": "auto"
                        }
                    }
                ]
            }
        ]
        
        # Make request to OpenAI
        logger.info(f"Comparing images: {image_path1} vs {image_path2}")
        response = await openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7
        )
        
        comparison = response.choices[0].message.content
        
        return {
            "success": True,
            "comparison": comparison,
            "image1_path": image_path1,
            "image2_path": image_path2,
            "prompt_used": comparison_prompt,
            "tokens_used": response.usage.total_tokens if response.usage else None
        }
        
    except Exception as e:
        logger.error(f"Error comparing images: {e}")
        return {
            "error": f"Failed to compare images: {str(e)}",
            "success": False
        }

def serve():
    """Main entry point for the Vision MCP Server."""
    # Check for OpenAI API key
    # if not os.getenv("OPENAI_API_KEY"):
    #     logger.error("OPENAI_API_KEY environment variable is not set!")
    #     logger.info("Please set your OpenAI API key in the environment or .env file")
    #     exit(1)
    
    # logger.info("Starting Vision MCP Server...")
    # logger.info("Available tools:")
    # logger.info("  - describe_image: Get detailed descriptions of images")
    # logger.info("  - batch_describe_images: Describe multiple images at once")
    # logger.info("  - compare_images: Compare two images and describe differences")

    
    # Run the MCP server
    mcp.run()

if __name__ == "__main__":
    main() 