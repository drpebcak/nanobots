#!/usr/bin/env python3
"""
videogenerator MCP Server
A Model Context Protocol server that generates videos.
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
from dotenv import load_dotenv

from fastmcp import FastMCP
from google import genai
import time


# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("videogenerator Server")

from google import genai
from google.genai import types

client = genai.Client(
    vertexai=True, project=os.getenv("GOOGLE_PROJECT_ID"), location=os.getenv("GOOGLE_LOCATION")
)


@mcp.tool()
async def generate_video(
    storyboard: Annotated[str, "Storyboard of the video"],
    duration: Annotated[int, "Video duration in seconds. Default is 5 seconds. Max is 8 seconds."] = 5
) -> Dict[str, Any]:
    """Generate a video based on a storyboard."""

    if duration > 8:
        duration = 8

    operation = client.models.generate_videos(
        # model='veo-3.0-generate-preview',
        model='veo-2.0-generate-001',
        prompt=storyboard,
        config=types.GenerateVideosConfig(
            aspect_ratio='16:9',
            person_generation="allow_adult",
            # generate_audio=True,
            number_of_videos=1,
            duration_seconds=duration,
            enhance_prompt=True,
            output_gcs_uri="gs://test-public-vertex"
        ),
    )
    while not operation.done:
        time.sleep(20)
        operation = client.operations.get(operation)

    video_uri = operation.result.generated_videos[0].video.uri
    http_url = video_uri.replace("gs://", "https://storage.googleapis.com/")
    # video.show()
    return {"video_link": http_url}

def serve():
    """Main entry point for the videogenerator MCP Server."""
    mcp.run()

if __name__ == "__main__":
    serve()