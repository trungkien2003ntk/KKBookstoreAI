"""
Script to pre-download models during container build time.
This ensures the models are already available when the application starts.
"""

import os
import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import logging
import requests
from pathlib import Path

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

# Use the standard pretrained weights without register tokens to match the model
MODEL_WEIGHTS = f"{MODEL_NAME}_pretrain"  # Use standard weights without reg tokens

# Set cache directories
os.environ["TORCH_HOME"] = os.getenv("TORCH_HOME", "/app/models/torch")
os.environ["TRANSFORMERS_CACHE"] = os.getenv("TRANSFORMERS_CACHE", "/app/models/transformers")
os.environ["HF_HOME"] = os.getenv("HF_HOME", "/app/models/huggingface")

# Create directories if they don't exist
for path in [os.environ["TORCH_HOME"], 
             f"{os.environ['TORCH_HOME']}/hub/checkpoints", 
             os.environ["TRANSFORMERS_CACHE"], 
             os.environ["HF_HOME"]]:
    os.makedirs(path, exist_ok=True)


def download_text_model():
    """Download and cache the text embedding model."""
    try:
        logger.info(f"Downloading text model: {TEXT_MODEL}")
        # This will download and cache the model in TRANSFORMERS_CACHE/HF_HOME
        model = SentenceTransformer(TEXT_MODEL)
        
        # Test the model to ensure it works
        dummy_text = "This is a test sentence to ensure the model is working properly."
        _ = model.encode([dummy_text])
        
        logger.info(f"Successfully downloaded and verified text model: {TEXT_MODEL}")
        return True
    except Exception as e:
        logger.error(f"Error downloading text model: {e}")
        return False


def download_image_model():
    """Download and cache the image embedding model with weights."""
    try:
        logger.info(f"Downloading image model: {MODEL_REPOSITORY}/{MODEL_NAME}")
        
        # Download the model repository WITH weights - this will cache everything properly
        model = torch.hub.load(
            repo_or_dir=MODEL_REPOSITORY,
            model=MODEL_NAME,
            pretrained=True,  # Download with pretrained weights
            trust_repo=True
        )
          # Test the model to ensure it works
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            _ = model(dummy_input)
        
        logger.info(f"Successfully downloaded and verified image model")
        return True
    except Exception as e:
        logger.error(f"Error downloading image model: {e}")
        return False


def verify_model_loading():
    """Verify that models can be loaded correctly."""
    try:
        logger.info("Verifying text model loading...")
        text_model = SentenceTransformer(TEXT_MODEL)
        dummy_text = "Test sentence for verification."
        _ = text_model.encode([dummy_text])
        logger.info("Text model verification successful!")
        
        logger.info("Verifying image model loading...")
        # Test loading with our exact approach used in application
        model = torch.hub.load(
            repo_or_dir=MODEL_REPOSITORY,
            model=MODEL_NAME,
            pretrained=True,
            force_reload=False
        )
            
        # Test forward pass
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            _ = model(dummy_input)
            
        logger.info("Image model verification successful!")
        return True
    except Exception as e:
        logger.error(f"Model verification failed: {e}")
        return False


if __name__ == "__main__":
    logger.info("Starting model downloads...")
    
    # Download text model
    text_success = download_text_model()
    
    # Download image model with weights
    image_success = download_image_model()
    
    # Verify that models can be loaded properly
    verification_success = verify_model_loading()
    
    if text_success and image_success and verification_success:
        logger.info("All models downloaded and verified successfully!")
        exit(0)
    else:
        logger.error("Failed to download or verify one or more models.")
        exit(1)
