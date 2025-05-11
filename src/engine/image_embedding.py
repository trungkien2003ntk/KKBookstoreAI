"""
This module provides functionality for generating image embeddings using a pre-trained model.
"""

import os
import base64
from typing import List
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
import torch
import torchvision.transforms as T

# Load environment variables
load_dotenv()

# Constants for model loading
MODEL_REPOSITORY = str(os.getenv("REPO_OR_DIR", "facebookresearch/dinov2"))
MODEL_NAME = str(os.getenv("DINO_MODEL", "dinov2_vitl14"))


class ImageEmbeddingGenerator:
    """
    A class to generate image embeddings using a pre-trained model.
    """

    def __init__(self, device: torch.device = None):
        """
        Initializes the ImageEmbeddingGenerator class.

        Args:
            device (torch.device): The device to run the model on (e.g., 'cuda' or 'cpu').
        """
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        try:
            self.model = torch.hub.load(
                repo_or_dir=MODEL_REPOSITORY,
                model=MODEL_NAME
            ).to(self.device)
        except Exception as error:
            raise RuntimeError(
                f"Failed to load the model: '{MODEL_REPOSITORY}' with name '{MODEL_NAME}': {error}"
            ) from error

        # Define image transformation pipeline
        self.image_transform_pipeline = T.Compose(
            [
                T.ToTensor(),
                T.Resize(244),
                T.CenterCrop(224),
                T.Normalize([0.5], [0.5])
            ]
        )

    async def decode_and_transform_image(self, encoded_image: str) -> torch.Tensor:
        """
        Decodes a base64-encoded image and applies transformations.

        Args:
            encoded_image (str): The base64-encoded image string.

        Returns:
            torch.Tensor: The transformed image tensor.
        """
        decoded_image_data = base64.b64decode(encoded_image)
        image = Image.open(BytesIO(decoded_image_data)).convert("RGB")

        transformed_image = self.image_transform_pipeline(image)[
            :3].unsqueeze(0)

        return transformed_image

    async def generate_image_embedding(self, encoded_image: str) -> List[float]:
        """
        Generates an embedding for the given base64-encoded image.

        Args:
            encoded_image (str): The base64-encoded image string.

        Returns:
            List[float]: The embedding as a list of floats.
        """
        transformed_image = await self.decode_and_transform_image(encoded_image)
        transformed_image = transformed_image.to(self.device)

        embedding_tensor = self.model(transformed_image)

        return embedding_tensor[0].cpu().detach().numpy().tolist()
