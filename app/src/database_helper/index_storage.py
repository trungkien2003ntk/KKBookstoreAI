"""
This module provides functionality for managing ChromaDB collections.
"""

import os
import shutil
import logging
from pathlib import Path
from dotenv import load_dotenv
import chromadb

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Default read-only index location (in app)
READ_ONLY_CHROMA_PATH = "./chromadb/chromadb"
# Writable runtime path
CHROMADB_STORAGE_PATH = "/tmp/chromadb"

IMAGE_CHROMA_COLLECTION = str(os.getenv("IMAGE_CHROMADB_NAME", "image_collection"))
TEXT_CHROMA_COLLECTION = str(os.getenv("TEXT_CHROMADB_NAME", "text_collection"))


def setup_chromadb():
    """
    Set up ChromaDB by copying from read-only location to writable location.
    
    Returns:
        bool: True if setup successful, False otherwise
    """
    try:
        # Check if source exists
        if not os.path.exists(READ_ONLY_CHROMA_PATH):
            logger.warning(f"Source ChromaDB path does not exist: {READ_ONLY_CHROMA_PATH}")
            logger.info("Will create new empty database")
            return True
            
        # Check if source has data
        source_path = Path(READ_ONLY_CHROMA_PATH)
        if not any(source_path.iterdir()):
            logger.warning(f"Source ChromaDB directory is empty: {READ_ONLY_CHROMA_PATH}")
            return True
            
        # Copy only if destination doesn't exist or is empty
        if not os.path.exists(CHROMADB_STORAGE_PATH):
            logger.info(f"Copying ChromaDB from {READ_ONLY_CHROMA_PATH} to {CHROMADB_STORAGE_PATH}")
            shutil.copytree(READ_ONLY_CHROMA_PATH, CHROMADB_STORAGE_PATH)
            logger.info("ChromaDB copy completed successfully")
        else:
            dest_path = Path(CHROMADB_STORAGE_PATH)
            if not any(dest_path.iterdir()):
                logger.info("Destination exists but is empty, removing and copying fresh")
                shutil.rmtree(CHROMADB_STORAGE_PATH)
                shutil.copytree(READ_ONLY_CHROMA_PATH, CHROMADB_STORAGE_PATH)
            else:
                logger.info("Destination already exists with data, skipping copy")
                
        return True
        
    except Exception as e:
        logger.error(f"Failed to setup ChromaDB: {e}")
        return False


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
        # Setup ChromaDB first
        if not setup_chromadb():
            logger.warning("ChromaDB setup had issues, proceeding anyway")
            
        try:
            # Ensure directory exists
            os.makedirs(CHROMADB_STORAGE_PATH, exist_ok=True)
            
            logger.info(f"Initializing ChromaDB client at: {CHROMADB_STORAGE_PATH}")
            self._chroma_client = chromadb.PersistentClient(path=CHROMADB_STORAGE_PATH)
            
            # Log existing collections
            try:
                collections = self._chroma_client.list_collections()
                logger.info(f"Found {len(collections)} existing collections: {[c.name for c in collections]}")
            except Exception as e:
                logger.warning(f"Could not list collections: {e}")
                
        except Exception as error:
            logger.error(f"Failed to initialize ChromaDB client: {error}")
            raise RuntimeError(f"Failed to initialize ChromaDB client: {error}") from error

    def _get_collection_info(self, collection_name: str) -> dict:
        """
        Get information about a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            dict: Collection information including count
        """
        try:
            collection = self._chroma_client.get_collection(name=collection_name)
            count = collection.count()
            return {
                "name": collection_name,
                "count": count,
                "exists": True
            }
        except Exception:
            return {
                "name": collection_name,
                "count": 0,
                "exists": False
            }

    @property
    def get_image_collection(self) -> chromadb.api.Collection:
        """
        Provides access to the ChromaDB image collection.

        Returns:
            chromadb.api.Collection: The ChromaDB image collection instance.
        """
        try:
            collection = self._chroma_client.get_or_create_collection(
                name=IMAGE_CHROMA_COLLECTION,
                metadata={"hnsw:space": "cosine"}
            )
            
            # Log collection info
            info = self._get_collection_info(IMAGE_CHROMA_COLLECTION)
            logger.info(f"Image collection - Count: {info['count']}, Exists: {info['exists']}")
            
            return collection
            
        except Exception as error:
            logger.error(f"Failed to access image collection: {error}")
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
            collection = self._chroma_client.get_or_create_collection(
                name=TEXT_CHROMA_COLLECTION,
                metadata={"hnsw:space": "cosine"}
            )
            
            # Log collection info
            info = self._get_collection_info(TEXT_CHROMA_COLLECTION)
            logger.info(f"Text collection - Count: {info['count']}, Exists: {info['exists']}")
            
            return collection
            
        except Exception as error:
            logger.error(f"Failed to access text collection: {error}")
            raise RuntimeError(
                f"Failed to access or create text collection '{TEXT_CHROMA_COLLECTION}': {error}"
            ) from error
    
    def debug_info(self) -> dict:
        """
        Get debug information about the ChromaDB setup.
        
        Returns:
            dict: Debug information
        """
        info = {
            "storage_path": CHROMADB_STORAGE_PATH,
            "storage_exists": os.path.exists(CHROMADB_STORAGE_PATH),
            "read_only_path": READ_ONLY_CHROMA_PATH,
            "read_only_exists": os.path.exists(READ_ONLY_CHROMA_PATH),
            "collections": {}
        }
        
        try:
            collections = self._chroma_client.list_collections()
            for collection in collections:
                info["collections"][collection.name] = {
                    "count": collection.count()
                }
        except Exception as e:
            info["error"] = str(e)
            
        return info