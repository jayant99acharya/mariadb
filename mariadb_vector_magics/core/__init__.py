"""
Core functionality for MariaDB Vector Magics.

This package contains the core business logic separated from the IPython magic interface.
"""

from .database import DatabaseManager
from .embeddings import EmbeddingManager
from .vector_store import VectorStoreManager
from .search import SearchManager
from .rag import RAGManager
from .exceptions import MariaDBVectorError
from .utils import HTMLRenderer

__all__ = [
    'DatabaseManager',
    'EmbeddingManager', 
    'VectorStoreManager',
    'SearchManager',
    'RAGManager',
    'MariaDBVectorError',
    'HTMLRenderer'
]