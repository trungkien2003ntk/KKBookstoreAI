"""
This module provides functionality for generating text embeddings 
using a pre-trained SentenceTransformer model.
"""

import os
from typing import List
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import torch

# Load environment variables
load_dotenv()

# Constants for model loading
TEXT_MODEL = str(os.getenv("TEXT_MODEL", "hiieu/halong_embedding"))


class TextEmbeddingGenerator:
    """
    A class to generate text embeddings using a pre-trained SentenceTransformer model.
    """

    def __init__(self, device: torch.device = None):
        """
        Initializes the TextEmbeddingGenerator class.

        Args:
            device (torch.device): The device to run the model on (e.g., 'cuda' or 'cpu').
        """
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        try:
            self.model = SentenceTransformer(TEXT_MODEL).to(self.device)
        except Exception as error:
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
