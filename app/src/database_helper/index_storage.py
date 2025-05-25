"""
This module provides functionality for managing ChromaDB collections.
"""

import os
from dotenv import load_dotenv
import chromadb

# Load environment variables
load_dotenv()

# Constants
IMAGE_CHROMA_COLLECTION = str(
    os.getenv("IMAGE_CHROMADB_NAME", "image_collection"))
TEXT_CHROMA_COLLECTION = str(
    os.getenv("TEXT_CHROMADB_NAME", "text_collection"))
CHROMADB_STORAGE_PATH = str(os.getenv("CHROMADB_PATH", "./chroma_db"))


class ChromaDBManager:
    """
    A manager class for handling ChromaDB collections,
    which are used for storing and retrieving embeddings.
    """

    def __init__(self) -> None:
        """
        Initializes the ChromaDB client and retrieves or creates collections.

        Raises:
            RuntimeError: If the ChromaDB client or collection initialization fails.
        """
        try:
            self._chroma_client = chromadb.PersistentClient(
                path=CHROMADB_STORAGE_PATH)
        except Exception as error:
            raise RuntimeError(
                f"Failed to initialize ChromaDB client: {error}"
            ) from error

    @property
    def get_image_collection(self) -> chromadb.api.Collection:
        """
        Provides access to the ChromaDB image collection.

        Returns:
            chromadb.api.Collection: The ChromaDB image collection instance.
        """
        try:
            return self._chroma_client.get_or_create_collection(
                name=IMAGE_CHROMA_COLLECTION,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as error:
            raise RuntimeError(
                f"Failed to access or create image collection '{IMAGE_CHROMA_COLLECTION}': {error}"
            ) from error

    @property
    def get_text_collection(self) -> chromadb.api.Collection:
        """
        Provides access to the ChromaDB text collection.

        Returns:
            chromadb.api.Collection: The ChromaDB text collection instance.
        """
        try:
            return self._chroma_client.get_or_create_collection(
                name=TEXT_CHROMA_COLLECTION,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as error:
            raise RuntimeError(
                f"Failed to access or create text collection '{TEXT_CHROMA_COLLECTION}': {error}"
            ) from error
