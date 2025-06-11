#!/usr/bin/env python3
"""
videogenerator MCP Server
A Model Context Protocol server that generates videos.
"""

import os
import json
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
from google.oauth2 import service_account
from google.genai import types
import time
import tempfile
import atexit


# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("videogenerator Server")

bucket_name = os.getenv("GCS_BUCKET_NAME", "test-public-vertex")

def load_service_account_credentials(service_account_path: str = None, service_account_json: str = None):
    """
    Load Google Cloud credentials from a service account JSON file or JSON string.
    
    Args:
        service_account_path: Path to the service account JSON key file
        service_account_json: JSON string containing service account credentials
        
    Returns:
        google.oauth2.service_account.Credentials object
    """
    try:
        if service_account_json:
            # Load from JSON string (environment variable)
            credentials_info = json.loads(service_account_json)
            credentials = service_account.Credentials.from_service_account_info(
                credentials_info,
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            logger.info("Successfully loaded service account credentials from JSON string")
            return credentials
        elif service_account_path:
            # Load from file
            credentials = service_account.Credentials.from_service_account_file(
                service_account_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            logger.info(f"Successfully loaded service account credentials from {service_account_path}")
            return credentials
        else:
            raise ValueError("Either service_account_path or service_account_json must be provided")
    except FileNotFoundError:
        logger.error(f"Service account file not found: {service_account_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in service account credentials: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading service account credentials: {e}")
        raise

def get_project_id_from_service_account(service_account_path: str) -> str:
    """
    Extract the project_id from the service account JSON file.
    
    Args:
        service_account_path: Path to the service account JSON key file
        
    Returns:
        Project ID string
    """
    try:
        with open(service_account_path, 'r') as f:
            data = json.load(f)
            project_id = data.get("project_id")
            if not project_id:
                raise ValueError("Key 'project_id' not found in service account file.")
            return project_id
    except Exception as e:
        logger.error(f"Error reading project_id from service account file: {e}")
        raise

def initialize_genai_client():
    """
    Initialize the GenAI client with proper authentication.
    
    Returns:
        Configured GenAI client
    """
    # Check for service account credentials
    service_account_file = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")
    service_account_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    
    if service_account_json:
        # Use service account JSON from environment variable
        logger.info("Using service account authentication from JSON environment variable")
        credentials = load_service_account_credentials(service_account_json=service_account_json)
        
        # Extract project ID from JSON or use environment variable
        project_id = os.getenv("GOOGLE_PROJECT_ID")
        if not project_id:
            credentials_info = json.loads(service_account_json)
            project_id = credentials_info.get("project_id")
        
        # Initialize client with credentials
        client = genai.Client(
            vertexai=True,
            project=project_id,
            location=os.getenv("GOOGLE_LOCATION", "us-central1"),
            credentials=credentials
        )
        logger.info(f"GenAI client initialized with service account JSON for project: {project_id}")
        
    elif service_account_file and os.path.exists(service_account_file):
        # Use service account file
        logger.info("Using service account authentication from file")
        credentials = load_service_account_credentials(service_account_path=service_account_file)
        
        # Get project ID from service account file or environment
        project_id = os.getenv("GOOGLE_PROJECT_ID")
        if not project_id:
            project_id = get_project_id_from_service_account(service_account_file)
        
        # Initialize client with credentials
        client = genai.Client(
            vertexai=True,
            project=project_id,
            location=os.getenv("GOOGLE_LOCATION", "us-central1"),
            credentials=credentials
        )
        logger.info(f"GenAI client initialized with service account file for project: {project_id}")
        
    else:
        # Use Application Default Credentials (ADC)
        logger.info("Using Application Default Credentials")
        client = genai.Client(
            vertexai=True,
            project=os.getenv("GOOGLE_PROJECT_ID"),
            location=os.getenv("GOOGLE_LOCATION", "us-central1")
        )
        logger.info("GenAI client initialized with ADC")
    
    return client

# Initialize the client
client = initialize_genai_client()

@mcp.tool()
async def generate_video(
    storyboard: Annotated[str, "Storyboard of the video"],
    duration: Annotated[
        int, "Video duration in seconds. Default is 5 seconds. Max is 8 seconds."
    ] = 5,
) -> Dict[str, Any]:
    """Generate a video based on a storyboard."""

    if duration > 8:
        duration = 8

    operation = client.models.generate_videos(
        # model='veo-3.0-generate-preview',
        model="veo-2.0-generate-001",
        prompt=storyboard,
        config=types.GenerateVideosConfig(
            aspect_ratio="16:9",
            person_generation="allow_adult",
            # generate_audio=True,
            number_of_videos=1,
            duration_seconds=duration,
            enhance_prompt=True,
            output_gcs_uri=f"gs://{bucket_name}",
        ),
    )
    while not operation.done:
        time.sleep(20)
        operation = client.operations.get(operation)

    if (
        hasattr(operation.result, "generated_videos")
        and operation.result.generated_videos
    ):
        video = operation.result.generated_videos[0].video
        http_url = video.uri.replace("gs://", "https://storage.googleapis.com/")

        return {"video_link": http_url}

    else:
        return {"error": "No video was generated", "operation_result": operation.result}


def serve():
    """Main entry point for the videogenerator MCP Server."""
    mcp.run()


if __name__ == "__main__":
    serve()
