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
from typing import Annotated
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

def encode_image_to_base64(image_path: Annotated[str, "Path to the image file"]) -> str:
    """Encode an image file to base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded_string
    except Exception as e:
        logger.error(f"Error encoding image: {e}")
        raise

def get_image_mime_type(image_path: Annotated[str, "Path to the image file"]) -> str:
    """Get MIME type of an image file."""
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
    image_path: Annotated[str, "Path to the image file to analyze"],
    prompt: Annotated[str, "Optional custom prompt for the description"] = None,
    max_tokens: Annotated[int, "Maximum number of tokens in the response"] = 300,
    detail: Annotated[str, "Level of detail for image analysis"] = "auto"
) -> Dict[str, Any]:
    """Upload an image to OpenAI and get a description."""
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
            model="gpt-4.1",
            messages=messages,
            max_tokens=max_tokens,
            temperature=1
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
    image_paths: Annotated[List[str], "List of paths to image files"],
    prompt: Annotated[str, "Optional custom prompt for descriptions"] = None,
    max_tokens: Annotated[int, "Maximum number of tokens per response"] = 300,
    detail: Annotated[str, "Level of detail for image analysis"] = "auto"
) -> Dict[str, Any]:
    """Describe multiple images in batch."""
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
    image_path1: Annotated[str, "Path to the first image"],
    image_path2: Annotated[str, "Path to the second image"],
    comparison_prompt: Annotated[str, "Prompt for the comparison"] = None,
    max_tokens: Annotated[int, "Maximum number of tokens in the response"] = 400
) -> Dict[str, Any]:
    """Compare two images and describe their differences and similarities."""
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
            model="gpt-4.1",
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
    mcp.run()

if __name__ == "__main__":
    serve()