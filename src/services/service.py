"""
This module provides a central service class for managing and accessing various services.
"""

from src.database_helper.index_storage import ChromaDBManager
from src.engine.text_embedding import TextEmbeddingGenerator
from src.engine.image_embedding import ImageEmbeddingGenerator
from src.services.search import EmbeddingSearchService


class ServiceManager:
    """
    A central service class that initializes and provides access to various services.
    """

    def __init__(
        self,
        db_manager: ChromaDBManager = None,
        text_embedding_generator: TextEmbeddingGenerator = None,
        image_embedding_generator: ImageEmbeddingGenerator = None
    ) -> None:
        """
        Initializes the ServiceManager class and its dependencies.

        Args:
            db_manager (ChromaDBManager, optional): 
                The ChromaDBManager instance for managing collections.
            text_embedding_generator (TextEmbeddingGenerator, optional):
                The text embedding generator instance.
        """
        self._db_manager = db_manager or ChromaDBManager()
        self._text_embedding_generator = text_embedding_generator or TextEmbeddingGenerator()
        self._image_embedding_generator = image_embedding_generator or ImageEmbeddingGenerator()

        self.search_service = EmbeddingSearchService(
            text_embedding_engine=self._text_embedding_generator,
            image_embedding_engine=self._image_embedding_generator,
            index_storage=self._db_manager
        )

        self.text_collection = self._db_manager.get_text_collection
