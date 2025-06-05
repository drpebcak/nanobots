#!/usr/bin/env python3
"""
Coordinator MCP Server
A Model Context Protocol server that coordinates between agents for generating videos.
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
# from openai import AsyncOpenAI
from dotenv import load_dotenv

from fastmcp import FastMCP, Context

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("Coordinator Server")

@mcp.tool()
async def storyboard(
    description: Annotated[str, "A description of the scene video"],
    ctx: Context,
    duration: Annotated[int, "The duration of the video in seconds. Default is 5 seconds. Max is 8 seconds."] = 5,
) -> dict:
    """Generate a storyboard for a video based on a description of a scene."""
    await ctx.info("Generating storyboard...")
    storyboard = await ctx.sample(f"Generate a storyboard for a short video (duration should be roughly {duration} seconds) based on the following description: {description}", model_preferences="storyWriter")
    return {
        "storyboard": storyboard
        }

@mcp.tool()
async def videoGeneration(
    storyboard: Annotated[str, "The storyboard of the scene video"],
    ctx: Context,
    duration: Annotated[int, "The duration of the video in seconds. Default is 5 seconds. Max is 8 seconds."] = 5,
) -> dict:
    """Generate a video based on a storyboard."""
    await ctx.info("Generating video...")
    video = await ctx.sample(f"Generate a short video (${duration} seconds long) based on the following storyboard: {storyboard}", model_preferences="videoMaker")
    return {
        "video": video
        }


def serve():
    """Main entry point for the Coordinator MCP Server."""
    mcp.run()

if __name__ == "__main__":
    serve()