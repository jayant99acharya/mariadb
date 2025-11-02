"""
Embedding management for MariaDB Vector Magics.

This module handles text embedding generation using sentence transformers.
"""

import logging
from typing import List, Optional, Dict, Any
import time

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
except ImportError:
    raise ImportError("sentence-transformers package is required. Install with: pip install sentence-transformers")

from .exceptions import EmbeddingError
from .utils import TextProcessor, ProcessingStats

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages text embedding generation and model operations."""
    
    def __init__(self):
        self.model: Optional[SentenceTransformer] = None
        self.model_name: Optional[str] = None
        self.model_info: Dict[str, Any] = {}
    
    def load_model(self, model_name: str = "all-MiniLM-L6-v2", 
                   cache_folder: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Load a sentence transformer model.
        
        Args:
            model_name: Name of the model to load
            cache_folder: Custom cache folder for model storage
            **kwargs: Additional model parameters
            
        Returns:
            Dict with model information
            
        Raises:
            EmbeddingError: If model loading fails
        """
        try:
            logger.info(f"Loading embedding model: {model_name}")
            start_time = time.time()
            
            # Load the model
            self.model = SentenceTransformer(
                model_name, 
                cache_folder=cache_folder,
                **kwargs
            )
            
            # Configure model settings
            self.model.max_seq_length = kwargs.get('max_seq_length', 256)
            
            # Store model information
            self.model_name = model_name
            self.model_info = {
                'name': model_name,
                'max_seq_length': self.model.max_seq_length,
                'embedding_dimension': self.model.get_sentence_embedding_dimension(),
                'load_time': time.time() - start_time,
                'device': str(self.model.device) if hasattr(self.model, 'device') else 'unknown'
            }
            
            logger.info(f"Model loaded successfully in {self.model_info['load_time']:.2f}s")
            logger.info(f"Model info: {self.model_info}")
            
            return self.model_info
            
        except Exception as e:
            error_msg = f"Failed to load model '{model_name}': {str(e)}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg, model_name) from e
    
    def is_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self.model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the currently loaded model."""
        if not self.is_loaded():
            raise EmbeddingError("No model is currently loaded")
        return self.model_info.copy()
    
    def encode_single(self, text: str, **kwargs) -> List[float]:
        """
        Encode a single text into embeddings.
        
        Args:
            text: Text to encode
            **kwargs: Additional encoding parameters
            
        Returns:
            List of embedding values
            
        Raises:
            EmbeddingError: If encoding fails
        """
        if not self.is_loaded():
            raise EmbeddingError("No model is currently loaded")
        
        try:
            # Sanitize text
            text = TextProcessor.sanitize_text(text)
            
            if not text.strip():
                raise EmbeddingError("Cannot encode empty text")
            
            # Generate embedding
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False,
                **kwargs
            )
            
            return embedding.tolist()
            
        except Exception as e:
            error_msg = f"Failed to encode text: {str(e)}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg, self.model_name) from e
    
    def encode_batch(self, texts: List[str], batch_size: int = 32, 
                     show_progress: bool = True, **kwargs) -> List[List[float]]:
        """
        Encode multiple texts into embeddings.
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            **kwargs: Additional encoding parameters
            
        Returns:
            List of embedding lists
            
        Raises:
            EmbeddingError: If encoding fails
        """
        if not self.is_loaded():
            raise EmbeddingError("No model is currently loaded")
        
        if not texts:
            raise EmbeddingError("Cannot encode empty text list")
        
        try:
            logger.info(f"Encoding {len(texts)} texts with batch size {batch_size}")
            start_time = time.time()
            
            # Sanitize texts
            sanitized_texts = []
            for i, text in enumerate(texts):
                sanitized = TextProcessor.sanitize_text(text)
                if not sanitized.strip():
                    logger.warning(f"Empty text at index {i}, using placeholder")
                    sanitized = "[EMPTY]"
                sanitized_texts.append(sanitized)
            
            # Generate embeddings
            embeddings = self.model.encode(
                sanitized_texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                **kwargs
            )
            
            processing_time = time.time() - start_time
            logger.info(f"Encoded {len(texts)} texts in {processing_time:.2f}s")
            
            return [emb.tolist() for emb in embeddings]
            
        except Exception as e:
            error_msg = f"Failed to encode batch: {str(e)}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg, self.model_name) from e
    
    def process_documents(self, documents: List[str], chunk_size: int = 500, 
                         overlap: int = 50, batch_size: int = 32) -> ProcessingStats:
        """
        Process documents with chunking and embedding generation.
        
        Args:
            documents: List of documents to process
            chunk_size: Maximum characters per chunk (0 to disable)
            overlap: Character overlap between chunks
            batch_size: Batch size for embedding generation
            
        Returns:
            ProcessingStats with processing information
            
        Raises:
            EmbeddingError: If processing fails
        """
        if not self.is_loaded():
            raise EmbeddingError("No model is currently loaded")
        
        # Validate documents
        is_valid, error_msg = TextProcessor.validate_documents(documents)
        if not is_valid:
            raise EmbeddingError(f"Invalid documents: {error_msg}")
        
        try:
            start_time = time.time()
            stats = ProcessingStats()
            
            logger.info(f"Processing {len(documents)} documents")
            
            # Process documents into chunks
            all_chunks = []
            chunk_metadata = []
            
            for doc_idx, document in enumerate(documents):
                if chunk_size > 0 and len(document) > chunk_size:
                    # Split into chunks
                    chunks = TextProcessor.chunk_text(document, chunk_size, overlap)
                    for chunk_idx, chunk in enumerate(chunks):
                        all_chunks.append(chunk)
                        chunk_metadata.append({
                            'original_doc_index': doc_idx,
                            'chunk_index': len(all_chunks) - 1,
                            'is_chunk': True,
                            'chunk_start': chunk_idx * (chunk_size - overlap),
                            'chunk_end': chunk_idx * (chunk_size - overlap) + len(chunk)
                        })
                else:
                    # Use document as-is
                    all_chunks.append(document)
                    chunk_metadata.append({
                        'original_doc_index': doc_idx,
                        'chunk_index': len(all_chunks) - 1,
                        'is_chunk': False
                    })
            
            stats.documents_processed = len(documents)
            stats.chunks_created = len(all_chunks)
            
            logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
            
            # Generate embeddings
            embeddings = self.encode_batch(all_chunks, batch_size=batch_size)
            stats.embeddings_generated = len(embeddings)
            
            stats.processing_time = time.time() - start_time
            
            logger.info(f"Document processing completed in {stats.processing_time:.2f}s")
            
            return stats, all_chunks, embeddings, chunk_metadata
            
        except Exception as e:
            error_msg = f"Failed to process documents: {str(e)}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg, self.model_name) from e
    
    def similarity_search(self, query_text: str, embeddings: List[List[float]], 
                         texts: List[str], top_k: int = 5, 
                         distance_metric: str = 'cosine') -> List[tuple]:
        """
        Perform similarity search using embeddings.
        
        Args:
            query_text: Query text
            embeddings: List of document embeddings
            texts: List of document texts
            top_k: Number of results to return
            distance_metric: Distance metric ('cosine' or 'euclidean')
            
        Returns:
            List of (index, text, distance) tuples
            
        Raises:
            EmbeddingError: If search fails
        """
        if not self.is_loaded():
            raise EmbeddingError("No model is currently loaded")
        
        try:
            # Generate query embedding
            query_embedding = self.encode_single(query_text)
            
            # Calculate distances
            distances = []
            query_array = np.array(query_embedding)
            
            for i, doc_embedding in enumerate(embeddings):
                doc_array = np.array(doc_embedding)
                
                if distance_metric == 'cosine':
                    # Cosine distance = 1 - cosine similarity
                    similarity = np.dot(query_array, doc_array) / (
                        np.linalg.norm(query_array) * np.linalg.norm(doc_array)
                    )
                    distance = 1 - similarity
                elif distance_metric == 'euclidean':
                    distance = np.linalg.norm(query_array - doc_array)
                else:
                    raise EmbeddingError(f"Unsupported distance metric: {distance_metric}")
                
                distances.append((i, texts[i], distance))
            
            # Sort by distance and return top_k
            distances.sort(key=lambda x: x[2])
            return distances[:top_k]
            
        except Exception as e:
            error_msg = f"Failed to perform similarity search: {str(e)}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg, self.model_name) from e
    
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension of the current model."""
        if not self.is_loaded():
            raise EmbeddingError("No model is currently loaded")
        return self.model.get_sentence_embedding_dimension()
    
    def unload_model(self) -> None:
        """Unload the current model to free memory."""
        if self.model is not None:
            logger.info(f"Unloading model: {self.model_name}")
            del self.model
            self.model = None
            self.model_name = None
            self.model_info = {}
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.unload_model()