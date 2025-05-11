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
    # response_model=List[str],
    description="Search for relevant products based on a base64-encoded image",
)
async def search_by_image_embedding(
    base64_image: str,
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
