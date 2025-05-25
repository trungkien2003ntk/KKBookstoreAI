"""
Script to pre-download models during container build time.
This ensures the models are already available when the application starts.
"""

import os
import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Model configuration
TEXT_MODEL = str(os.getenv("TEXT_MODEL", "hiieu/halong_embedding"))
MODEL_REPOSITORY = str(os.getenv("REPO_OR_DIR", "facebookresearch/dinov2"))
MODEL_NAME = str(os.getenv("DINO_MODEL", "dinov2_vitl14"))

# Set cache directories
os.environ["TORCH_HOME"] = os.getenv("TORCH_HOME", "/app/models/torch")
os.environ["TRANSFORMERS_CACHE"] = os.getenv("TRANSFORMERS_CACHE", "/app/models/transformers")
os.environ["HF_HOME"] = os.getenv("HF_HOME", "/app/models/huggingface")

# Create directories if they don't exist
for path in [os.environ["TORCH_HOME"], os.environ["TRANSFORMERS_CACHE"], os.environ["HF_HOME"]]:
    os.makedirs(path, exist_ok=True)


def download_text_model():
    """Download and cache the text embedding model."""
    try:
        logger.info(f"Downloading text model: {TEXT_MODEL}")
        model = SentenceTransformer(TEXT_MODEL)
        logger.info(f"Successfully downloaded text model: {TEXT_MODEL}")
        return True
    except Exception as e:
        logger.error(f"Error downloading text model: {e}")
        return False


def download_image_model():
    """Download and cache the image embedding model."""
    try:
        logger.info(f"Downloading image model: {MODEL_REPOSITORY}/{MODEL_NAME}")
        model = torch.hub.load(
            repo_or_dir=MODEL_REPOSITORY,
            model=MODEL_NAME
        )
        logger.info(f"Successfully downloaded image model: {MODEL_NAME}")
        return True
    except Exception as e:
        logger.error(f"Error downloading image model: {e}")
        return False


if __name__ == "__main__":
    logger.info("Starting model downloads...")
    
    text_success = download_text_model()
    image_success = download_image_model()
    
    if text_success and image_success:
        logger.info("All models downloaded successfully!")
        exit(0)
    else:
        logger.error("Failed to download one or more models.")
        exit(1)
