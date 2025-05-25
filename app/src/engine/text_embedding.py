"""
This module provides functionality for generating text embeddings 
using a pre-trained SentenceTransformer model.
"""

import os
import logging
from typing import List
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import torch

# Load environment variables
load_dotenv()

# Constants for model loading
TEXT_MODEL = str(os.getenv("TEXT_MODEL", "hiieu/halong_embedding"))

# Configure paths to use models that were pre-downloaded during container build
# These environment variables are checked first and fallback to default locations
os.environ["TORCH_HOME"] = os.getenv("TORCH_HOME", os.environ.get("TORCH_HOME", None))
os.environ["TRANSFORMERS_CACHE"] = os.getenv("TRANSFORMERS_CACHE", os.environ.get("TRANSFORMERS_CACHE", None))
os.environ["HF_HOME"] = os.getenv("HF_HOME", os.environ.get("HF_HOME", None))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextEmbeddingGenerator:
    """
    A class to generate text embeddings using a pre-trained SentenceTransformer model.
    """    def __init__(self, device: torch.device = None):
        """
        Initializes the TextEmbeddingGenerator class.

        Args:
            device (torch.device): The device to run the model on (e.g., 'cuda' or 'cpu').
        """
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        try:
            logger.info(f"Loading pre-downloaded text model: {TEXT_MODEL}")
            self.model = SentenceTransformer(TEXT_MODEL).to(self.device)
            logger.info("Text model loaded successfully")
        except Exception as error:
            logger.error(f"Failed to load the SentenceTransformer model: {error}")
            raise RuntimeError(
                f"Failed to load the SentenceTransformer model '{TEXT_MODEL}': {error}"
            ) from error

    async def generate_text_embedding(self, input_text: str) -> List[float]:
        """
        Encodes the input text into an embedding.

        Args:
            input_text (str): The input text string.

        Returns:
            List[float]: The embedding of the input text as a list of floats.
        """
        embedding = self.model.encode([input_text], convert_to_tensor=True)
        return embedding[0].cpu().detach().numpy().tolist()
