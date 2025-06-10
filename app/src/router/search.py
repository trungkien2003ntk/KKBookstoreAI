"""
This module provides API endpoints for searching relevant products based on product embeddings.
"""

from typing import List
from fastapi import (
    status,
    Depends,
    APIRouter,
    HTTPException
)
from src.services.service import ServiceManager
from src.dependencies.service_dependency import get_service
from pydantic import BaseModel

class ImageSearchRequest(BaseModel):
    base64_image: str

class EnhancedImageSearchRequest(BaseModel):
    base64_image: str
    deduplication_strategy: str = "first_occurrence"  # Default to current behavior
    n_results: int = 200

# Define the router
product_router = APIRouter(
    tags=["Product Retrieval"],
    prefix="/product",
)


@product_router.post(
    '/{product_id}/related',
    status_code=status.HTTP_200_OK,
    response_model=List[str],
    description="Search for relevant products based on product ID",
)
async def search_by_id(
    product_id: str,
    service: ServiceManager = Depends(get_service)
) -> List[str]:
    """
    Searches for relevant products based on a given product ID.

    Args:
        product_id (str): The ID of the product to search for.
        service (ServiceManager): The service instance for handling product search.

    Returns:
        List[str]: A list of IDs for the most relevant products.

    Raises:
        HTTPException: If the product ID is invalid, not found, or an internal error occurs.
    """
    if not product_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": "Product ID is required."}
        )

    # Check if the product ID exists in the collection
    retrieved_data = service.text_collection.get(ids=[product_id])
    if not retrieved_data.get("documents"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"message": "Product ID not found."}
        )

    try:
        # Perform the search
        search_results = await service.search_service.search_by_id(
            product_id=product_id,
        )

        return search_results

    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": "An error occurred while processing the request."}
        ) from error


@product_router.post(
    '/related-by-image',
    status_code=status.HTTP_200_OK,
    response_model=List[str],
    description="Search for relevant products based on a base64-encoded image",
)
async def search_by_image_embedding(
    request_data: ImageSearchRequest,  # Now expecting a request body
    service: ServiceManager = Depends(get_service)
) -> List[str]:
    """
    Searches for relevant products based on a given base64-encoded image.

    Args:
        base64_image (str): The base64-encoded image to search for.
        service (ServiceManager): The service instance for handling product search.

    Returns:
        List[str]: A list of IDs for the most relevant products.

    Raises:
        HTTPException: If the base64 image is invalid or an internal error occurs.
    """
    base64_image = request_data.base64_image
    if not base64_image:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": "Base64 image is required."}
        )

    try:
        # Perform the search
        search_results = await service.search_service.search_by_image_embedding(
            base64_image=base64_image,
        )

        return search_results

    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": "An error occurred while processing the request."}
        ) from error


@product_router.post(
    '/enhanced-related-by-image',
    status_code=status.HTTP_200_OK,
    response_model=List[str],
    description="Enhanced search for relevant products based on a base64-encoded image",
)
async def enhanced_search_by_image_embedding(
    request_data: EnhancedImageSearchRequest,  # Now expecting a request body
    service: ServiceManager = Depends(get_service)
) -> List[str]:
    """
    Enhanced search for relevant products based on a given base64-encoded image.

    Args:
        base64_image (str): The base64-encoded image to search for.
        deduplication_strategy (str): The strategy for deduplicating results.
        n_results (int): The number of results to return.
        service (ServiceManager): The service instance for handling product search.

    Returns:
        List[str]: A list of IDs for the most relevant products.

    Raises:
        HTTPException: If the base64 image is invalid or an internal error occurs.
    """
    base64_image = request_data.base64_image
    if not base64_image:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": "Base64 image is required."}
        )

    try:
        # Perform the enhanced search
        search_results = await service.search_service.enhanced_search_by_image_embedding(
            base64_image=base64_image,
            deduplication_strategy=request_data.deduplication_strategy,
            n_results=request_data.n_results
        )

        return search_results

    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": "An error occurred while processing the request."}
        ) from error


@product_router.post(
    '/related-by-image-enhanced',
    status_code=status.HTTP_200_OK,
    response_model=List[str],
    description="Enhanced image search with configurable deduplication strategies",
)
async def search_by_image_embedding_enhanced(
    request_data: EnhancedImageSearchRequest,
    service: ServiceManager = Depends(get_service)
) -> List[str]:
    """
    Enhanced image search with configurable deduplication strategies.
    
    Deduplication strategies:
    - "first_occurrence": Take first occurrence of each product (current behavior)
    - "best_match": Take best (lowest distance) match per product
    - "no_deduplication": Return all results preserving ChromaDB order
    
    Args:
        request_data: Enhanced request containing base64 image and strategy options
        service: The service instance for handling product search.

    Returns:
        List[str]: A list of product IDs ordered by relevance.

    Raises:
        HTTPException: If the request is invalid or an internal error occurs.
    """
    if not request_data.base64_image:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": "Base64 image is required."}
        )
    
    valid_strategies = ["first_occurrence", "best_match", "no_deduplication"]
    if request_data.deduplication_strategy not in valid_strategies:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": f"Invalid deduplication strategy. Must be one of: {valid_strategies}"}
        )

    try:
        # Use the enhanced search method
        search_results = await service.search_service.search_by_image_embedding_enhanced(
            base64_image=request_data.base64_image,
            n_results=request_data.n_results,
            deduplication_strategy=request_data.deduplication_strategy
        )

        return search_results

    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": "An error occurred while processing the request."}
        ) from error
