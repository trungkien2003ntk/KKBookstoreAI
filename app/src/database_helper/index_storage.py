import os
import shutil
from dotenv import load_dotenv
import chromadb

load_dotenv()

# Default read-only index location (in app)
READ_ONLY_CHROMA_PATH = "./chromadb/chromadb"
# Writable runtime path
CHROMADB_STORAGE_PATH = "/tmp/chromadb"

# Copy prebuilt DB from app to /tmp
if not os.path.exists(CHROMADB_STORAGE_PATH):
    try:
        shutil.copytree(READ_ONLY_CHROMA_PATH, CHROMADB_STORAGE_PATH)
    except Exception as e:
        raise RuntimeError(f"Failed to copy ChromaDB index to /tmp: {e}")

IMAGE_CHROMA_COLLECTION = str(os.getenv("IMAGE_CHROMADB_NAME", "image_collection"))
TEXT_CHROMA_COLLECTION = str(os.getenv("TEXT_CHROMADB_NAME", "text_collection"))

class ChromaDBManager:
    def __init__(self) -> None:
        try:
            self._chroma_client = chromadb.PersistentClient(path=CHROMADB_STORAGE_PATH)
        except Exception as error:
            raise RuntimeError(f"Failed to initialize ChromaDB client: {error}") from error

    @property
    def get_image_collection(self) -> chromadb.api.Collection:
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
        try:
            return self._chroma_client.get_or_create_collection(
                name=TEXT_CHROMA_COLLECTION,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as error:
            raise RuntimeError(
                f"Failed to access or create text collection '{TEXT_CHROMA_COLLECTION}': {error}"
            ) from error
