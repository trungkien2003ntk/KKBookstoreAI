"""
Alternative implementation of search_by_image_embedding with improved deduplication strategies.
You can use these implementations to replace or enhance your current search method.
"""

from typing import List, Dict, Tuple
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

async def search_by_image_embedding_v2(
    self,
    base64_image: str,
    n_results: int = 200,
    deduplication_strategy: str = "first_occurrence"
) -> List[str]:
    """
    Enhanced image search with multiple deduplication strategies.
    
    Args:
        base64_image (str): The base64-encoded image to search for.
        n_results (int): The number of results to return from ChromaDB.
        deduplication_strategy (str): One of:
            - "first_occurrence": Take first occurrence of each product (current behavior)
            - "best_match": Take best (lowest distance) match per product
            - "no_deduplication": Return all results preserving ChromaDB order
            - "weighted_best": Use distance-weighted selection
    
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
    logger.info(f"Top 5 image search results (strategy: {deduplication_strategy}):")
    for i in range(min(5, len(metadatas[0]))):
        metadata = metadatas[0][i]
        distance = distances[0][i] if distances else float('inf')
        logger.info(f"  {i+1}. Product: {metadata.get('product_id')}, "
                   f"Image: {metadata.get('image_id')}, Distance: {distance:.6f}")

    # Apply deduplication strategy
    if deduplication_strategy == "no_deduplication":
        return _no_deduplication(metadatas[0])
    elif deduplication_strategy == "best_match":
        return _best_match_deduplication(metadatas[0], distances[0] if distances else None)
    elif deduplication_strategy == "weighted_best":
        return _weighted_best_deduplication(metadatas[0], distances[0] if distances else None)
    else:  # "first_occurrence" (default/current behavior)
        return _first_occurrence_deduplication(metadatas[0], ids[0] if ids else None)

def _no_deduplication(metadatas: List[Dict]) -> List[str]:
    """Return all product IDs preserving ChromaDB order."""
    return [metadata.get("product_id") for metadata in metadatas 
            if metadata.get("product_id")]

def _first_occurrence_deduplication(metadatas: List[Dict], ids: List[str] = None) -> List[str]:
    """Current behavior: take first occurrence of each product."""
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

def _best_match_deduplication(metadatas: List[Dict], distances: List[float] = None) -> List[str]:
    """Take the best (lowest distance) match per product."""
    if not distances:
        logger.warning("No distances available, falling back to first occurrence")
        return _first_occurrence_deduplication(metadatas)
    
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
    
    logger.debug(f"Best match deduplication selected {len(result)} unique products")
    for product_id, info in sorted_products[:5]:
        logger.debug(f"  Product {product_id}: distance {info['distance']:.6f}, "
                    f"image {info['image_id']}")
    
    return result

def _weighted_best_deduplication(metadatas: List[Dict], distances: List[float] = None) -> List[str]:
    """
    Use weighted selection: heavily favor original ChromaDB order, but adjust for distance.
    This is a compromise between preserving order and ensuring best matches.
    """
    if not distances:
        return _first_occurrence_deduplication(metadatas)
    
    # Group by product and calculate weighted scores
    product_scores = defaultdict(list)
    
    for i, (metadata, distance) in enumerate(zip(metadatas, distances)):
        product_id = metadata.get("product_id")
        if not product_id:
            continue
        
        # Weighted score: lower is better
        # Favor early positions (low i) but also consider distance
        position_weight = i * 0.1  # Small penalty for later positions
        distance_weight = distance * 10  # Larger penalty for higher distances
        weighted_score = position_weight + distance_weight
        
        product_scores[product_id].append({
            "score": weighted_score,
            "rank": i,
            "distance": distance,
            "image_id": metadata.get("image_id")
        })
    
    # Select best score per product
    selected_products = {}
    for product_id, candidates in product_scores.items():
        best_candidate = min(candidates, key=lambda x: x["score"])
        selected_products[product_id] = best_candidate
    
    # Sort by best score (which incorporates both position and distance)
    sorted_products = sorted(
        selected_products.items(),
        key=lambda x: x[1]["score"]
    )
    
    result = [product_id for product_id, _ in sorted_products]
    
    logger.debug(f"Weighted deduplication selected {len(result)} unique products")
    
    return result

# Usage examples:
"""
# In your search service, you can now call:

# For exact ChromaDB ordering (no deduplication)
results = await search_by_image_embedding_v2(
    base64_image, 
    deduplication_strategy="no_deduplication"
)

# For best distance-based matching
results = await search_by_image_embedding_v2(
    base64_image, 
    deduplication_strategy="best_match"
)

# For weighted compromise approach
results = await search_by_image_embedding_v2(
    base64_image, 
    deduplication_strategy="weighted_best"
)
"""
