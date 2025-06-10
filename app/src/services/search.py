"""
This module provides functionality for searching relevant products based on embeddings.
"""

import os
import logging
from typing import List
from collections import defaultdict
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
        print(f"Image collection count: {self.image_collection.count()}")
        print(f"Text collection count: {self.text_collection.count()}")

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
        n_results: int = DEFAULT_IMAGE_RESULTS,
        deduplication_strategy: str = "first"  # New parameter for deduplication strategy
    ) -> List[str]:
        """
        Searches for relevant products based on a given base64 image.

        Args:
            base64_image (str): The base64-encoded image to search for.
            n_results (int): The number of results to return. Defaults to 100.
            deduplication_strategy (str): The strategy for deduplication.
            Options: "first" - keep first occurrence, "last" - keep last occurrence, "none" - no deduplication.

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
        ids = results.get("ids", [])
        distances = results.get("distances", [])
        
        if not metadatas or not ids:
            logger.warning(
                "Failed to retrieve by image. Returning an empty list.")
            return []

        # Log the top few results for debugging
        logger.info(f"Top 5 image search results:")
        for i in range(min(5, len(metadatas[0]))):
            logger.info(f"  {i+1}. ID: {ids[0][i]}, Product: {metadatas[0][i].get('product_id')}, Distance: {distances[0][i] if distances else 'N/A'}")

        # Deduplication logic based on the selected strategy
        if deduplication_strategy == "first":
            seen = set()
            ordered_product_ids = []
            for i, item in enumerate(metadatas[0]):
                pid = item.get("product_id")
                if pid and pid not in seen:
                    seen.add(pid)
                    ordered_product_ids.append(pid)
                    logger.debug(f"Selected product {pid} from image {ids[0][i]} at position {i}")
            return ordered_product_ids

        elif deduplication_strategy == "last":
            seen = set()
            ordered_product_ids = []
            for i in range(len(metadatas[0]) - 1, -1, -1):
                item = metadatas[0][i]
                pid = item.get("product_id")
                if pid and pid not in seen:
                    seen.add(pid)
                    ordered_product_ids.append(pid)
                    logger.debug(f"Selected product {pid} from image {ids[0][i]} at position {i}")
            return list(reversed(ordered_product_ids))

        # Option 3: No deduplication, return all results
        return [metadata.get("product_id") for metadata in metadatas[0] if metadata.get("product_id")]

    async def search_by_image_embedding_enhanced(
        self,
        base64_image: str,
        n_results: int = DEFAULT_IMAGE_RESULTS,
        deduplication_strategy: str = "first_occurrence"
    ) -> List[str]:
        """
        Enhanced image search with configurable deduplication strategies.
        
        Args:
            base64_image (str): The base64-encoded image to search for.
            n_results (int): The number of results to return from ChromaDB.
            deduplication_strategy (str): Strategy for handling duplicate products:
                - "first_occurrence": Current behavior - take first occurrence
                - "best_match": Take best (lowest distance) match per product  
                - "no_deduplication": Return all results preserving ChromaDB order
        
        Returns:
            List[str]: A list of product IDs ordered by relevance.
        """
        embedding = await self.image_embedding_engine.generate_image_embedding(
            encoded_image=base64_image)

        results = self.image_collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["metadatas", "distances"]
        )

        metadatas = results.get("metadatas", [])
        ids = results.get("ids", [])
        distances = results.get("distances", [])
        
        if not metadatas or not ids:
            logger.warning("Failed to retrieve by image. Returning an empty list.")
            return []

        # Log top results for debugging
        logger.info(f"Enhanced search - Top 5 results (strategy: {deduplication_strategy}):")
        for i in range(min(5, len(metadatas[0]))):
            metadata = metadatas[0][i]
            distance = distances[0][i] if distances else float('inf')
            logger.info(f"  {i+1}. Product: {metadata.get('product_id')}, "
                       f"Image: {metadata.get('image_id')}, Distance: {distance:.6f}")

        # Apply deduplication strategy
        if deduplication_strategy == "no_deduplication":
            return [metadata.get("product_id") for metadata in metadatas[0] 
                    if metadata.get("product_id")]
        elif deduplication_strategy == "best_match":
            return self._best_match_deduplication(metadatas[0], distances[0] if distances else None)
        else:  # "first_occurrence" (default/current behavior)
            return self._first_occurrence_deduplication_enhanced(metadatas[0], ids[0] if ids else None)

    def _first_occurrence_deduplication_enhanced(self, metadatas: List[dict], ids: List[str] = None) -> List[str]:
        """Enhanced first occurrence deduplication with logging."""
        seen = set()
        ordered_product_ids = []
        
        for i, metadata in enumerate(metadatas):
            product_id = metadata.get("product_id")
            if product_id and product_id not in seen:
                seen.add(product_id)
                ordered_product_ids.append(product_id)
                if ids:
                    logger.debug(f"Selected product {product_id} from image {ids[i]} at position {i+1}")
        
        return ordered_product_ids

    def _best_match_deduplication(self, metadatas: List[dict], distances: List[float] = None) -> List[str]:
        """Take the best (lowest distance) match per product."""
        if not distances:
            logger.warning("No distances available, falling back to first occurrence")
            return self._first_occurrence_deduplication_enhanced(metadatas)
        
        # Group by product_id and keep the best match
        product_best_match = {}
        
        for i, (metadata, distance) in enumerate(zip(metadatas, distances)):
            product_id = metadata.get("product_id")
            if not product_id:
                continue
                
            if product_id not in product_best_match or distance < product_best_match[product_id]["distance"]:
                product_best_match[product_id] = {
                    "distance": distance,
                    "rank": i,
                    "image_id": metadata.get("image_id")
                }
        
        # Sort by best distance per product, maintaining relative order for ties
        sorted_products = sorted(
            product_best_match.items(),
            key=lambda x: (x[1]["distance"], x[1]["rank"])
        )
        
        result = [product_id for product_id, info in sorted_products]
        
        logger.info(f"Best match deduplication selected {len(result)} unique products")
        for product_id, info in sorted_products[:3]:
            logger.debug(f"  Product {product_id}: distance {info['distance']:.6f}, "
                        f"image {info['image_id']}")
        
        return result
