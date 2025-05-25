"""
This module provides functionality for searching relevant products based on embeddings.
"""

import os
import logging
from typing import List
from dotenv import load_dotenv
from src.database_helper.index_storage import ChromaDBManager
from src.engine.text_embedding import TextEmbeddingGenerator
from src.engine.image_embedding import ImageEmbeddingGenerator


# Load environment variables
load_dotenv()

# Constants for search results
DEFAULT_TEXT_RESULTS = int(os.getenv("TEXT_N_RESULTS", "100"))
DEFAULT_IMAGE_RESULTS = int(os.getenv("IMAGE_N_RESULTS", "100"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingSearchService:
    """
    A service for searching relevant products using embeddings and a ChromaDB collection.
    """

    def __init__(
        self,
        text_embedding_engine: TextEmbeddingGenerator,
        image_embedding_engine: ImageEmbeddingGenerator,
        index_storage: ChromaDBManager,
    ):
        """
        Initializes the EmbeddingSearchService.

        Args:
            text_embedding_engine (TextEmbeddingGenerator): The text embedding engine.
            image_embedding_engine (ImageEmbeddingGenerator): The image embedding engine.
            index_storage (ChromaDBManager): The ChromaDB index storage instance.
        """
        self.text_embedding_engine = text_embedding_engine
        self.image_embedding_engine = image_embedding_engine
        self.index_storage = index_storage
        self.image_collection = self.index_storage.get_image_collection
        self.text_collection = self.index_storage.get_text_collection

    async def search_by_id(
        self,
        product_id: str,
        n_results: int = DEFAULT_TEXT_RESULTS
    ) -> List[str]:
        """
        Searches for relevant products based on a given product ID.

        Args:
            product_id (str): The ID of the product to search for.
            n_results (int): The number of results to return. Defaults to 100.

        Returns:
            List[str]: A list of IDs for the most relevant products,
            or an empty list if no results are found.
        """
        retrieved_data = self.text_collection.get(ids=[product_id])
        documents = retrieved_data.get("documents")
        if not documents:
            logger.warning(
                "Failed to retrieve by ID. Returning an empty list.")
            return []

        description = documents[0]
        embedding = await self.text_embedding_engine.generate_text_embedding(
            input_text=description)

        results = self.text_collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
        )

        ids = results.get("ids", [])
        flat_ids = [item for sublist in ids for item in sublist] if isinstance(
            ids[0], list) else ids

        return flat_ids

    async def search_by_image_embedding(
        self,
        base64_image: str,
        n_results: int = DEFAULT_IMAGE_RESULTS
    ) -> List[str]:
        """
        Searches for relevant products based on a given base64 image.

        Args:
            base64_image (str): The base64-encoded image to search for.
            n_results (int): The number of results to return. Defaults to 100.

        Returns:
            List[str]: A list of IDs for the most relevant products,
            or an empty list if no results are found.
        """
        embedding = await self.image_embedding_engine.generate_image_embedding(
            encoded_image=base64_image)

        results = self.image_collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
        )

        metadatas = results.get("metadatas", [])
        if not metadatas:
            logger.warning(
                "Failed to retrieve by image. Returning an empty list.")
            return []

        seen = set()
        ordered_product_ids = []
        for item in metadatas[0]:
            pid = item.get("product_id")
            if pid and pid not in seen:
                seen.add(pid)
                ordered_product_ids.append(pid)

        return ordered_product_ids
