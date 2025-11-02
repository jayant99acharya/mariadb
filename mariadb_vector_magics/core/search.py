"""
Search management for MariaDB Vector operations.

This module handles semantic search and similarity operations.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import json

from .database import DatabaseManager
from .embeddings import EmbeddingManager
from .exceptions import SearchError, DatabaseOperationError, EmbeddingError

logger = logging.getLogger(__name__)


class SearchManager:
    """Manages semantic search operations on vector stores."""
    
    def __init__(self, db_manager: DatabaseManager, embedding_manager: EmbeddingManager):
        """
        Initialize search manager.
        
        Args:
            db_manager: Database manager instance
            embedding_manager: Embedding manager instance
        """
        self.db_manager = db_manager
        self.embedding_manager = embedding_manager
        self.table_distance_types: Dict[str, str] = {}
    
    def semantic_search(self, query: str, table: str, top_k: int = 5,
                       distance_threshold: Optional[float] = None,
                       include_distance: bool = False,
                       distance_metric: Optional[str] = None) -> List[Tuple]:
        """
        Perform semantic search on a vector table.
        
        Args:
            query: Search query text
            table: Table to search
            top_k: Number of results to return
            distance_threshold: Optional distance threshold filter
            include_distance: Whether to include distance scores
            distance_metric: Override distance metric for this search
            
        Returns:
            List of result tuples
            
        Raises:
            SearchError: If search fails
        """
        try:
            # Validate inputs
            self._validate_search_params(query, table, top_k)
            
            logger.info(f"Performing semantic search: '{query}' in table '{table}'")
            
            # Generate query embedding
            query_embedding = self.embedding_manager.encode_single(query)
            
            # Determine distance function
            distance_type = distance_metric or self.table_distance_types.get(table, 'cosine')
            distance_func = self._get_distance_function(distance_type)
            
            # Build and execute search query
            search_sql, params = self._build_search_query(
                table, distance_func, top_k, include_distance
            )
            
            embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
            if include_distance:
                params = (embedding_str, embedding_str, top_k)
            else:
                params = (embedding_str, top_k)
            
            results = self.db_manager.execute_query(search_sql, params)
            
            # Apply distance threshold if specified
            if distance_threshold is not None and include_distance:
                results = [r for r in results if r[-1] <= distance_threshold]
            
            logger.info(f"Search returned {len(results)} results")
            return results
            
        except (DatabaseOperationError, EmbeddingError) as e:
            error_msg = f"Search failed for query '{query}' in table '{table}': {e}"
            logger.error(error_msg)
            raise SearchError(error_msg, query, table) from e
        except Exception as e:
            error_msg = f"Unexpected error during search: {e}"
            logger.error(error_msg)
            raise SearchError(error_msg, query, table) from e
    
    def _validate_search_params(self, query: str, table: str, top_k: int) -> None:
        """Validate search parameters."""
        if not query or not query.strip():
            raise SearchError("Query cannot be empty")
        
        if not table or not table.strip():
            raise SearchError("Table name cannot be empty")
        
        if top_k <= 0:
            raise SearchError("top_k must be positive")
        
        if top_k > 1000:
            raise SearchError("top_k cannot exceed 1000")
        
        # Check if table exists
        if not self.db_manager.table_exists(table):
            raise SearchError(f"Table '{table}' does not exist", query, table)
    
    def _get_distance_function(self, distance_type: str) -> str:
        """Get MariaDB distance function name."""
        if distance_type.lower() == 'cosine':
            return 'VEC_DISTANCE_COSINE'
        elif distance_type.lower() == 'euclidean':
            return 'VEC_DISTANCE_EUCLIDEAN'
        else:
            raise SearchError(f"Unsupported distance type: {distance_type}")
    
    def _build_search_query(self, table: str, distance_func: str, 
                           top_k: int, include_distance: bool) -> Tuple[str, Tuple]:
        """Build SQL query for semantic search."""
        if include_distance:
            query_sql = f"""
            SELECT id, document_text, metadata, 
                {distance_func}(embedding, VEC_FromText(?)) as distance
            FROM {table}
            ORDER BY {distance_func}(embedding, VEC_FromText(?))
            LIMIT ?
            """
        else:
            query_sql = f"""
            SELECT id, document_text, metadata
            FROM {table}
            ORDER BY {distance_func}(embedding, VEC_FromText(?))
            LIMIT ?
            """
        
        return query_sql, ()
    
    def hybrid_search(self, query: str, table: str, top_k: int = 5,
                     text_weight: float = 0.3, vector_weight: float = 0.7,
                     include_distance: bool = False) -> List[Tuple]:
        """
        Perform hybrid search combining text and vector search.
        
        Args:
            query: Search query
            table: Table to search
            top_k: Number of results
            text_weight: Weight for text search (0-1)
            vector_weight: Weight for vector search (0-1)
            include_distance: Whether to include combined scores
            
        Returns:
            List of result tuples
            
        Raises:
            SearchError: If search fails
        """
        try:
            # Validate weights
            if abs(text_weight + vector_weight - 1.0) > 0.001:
                raise SearchError("Text weight and vector weight must sum to 1.0")
            
            logger.info(f"Performing hybrid search: '{query}' in table '{table}'")
            
            # Generate query embedding
            query_embedding = self.embedding_manager.encode_single(query)
            embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            # Get distance function
            distance_type = self.table_distance_types.get(table, 'cosine')
            distance_func = self._get_distance_function(distance_type)
            
            # Build hybrid search query
            if include_distance:
                hybrid_sql = f"""
                SELECT id, document_text, metadata,
                    ({text_weight} * (1 - MATCH(document_text) AGAINST(? IN NATURAL LANGUAGE MODE)) +
                     {vector_weight} * {distance_func}(embedding, VEC_FromText(?))) as combined_score
                FROM {table}
                WHERE MATCH(document_text) AGAINST(? IN NATURAL LANGUAGE MODE) > 0
                   OR {distance_func}(embedding, VEC_FromText(?)) < 1.0
                ORDER BY combined_score
                LIMIT ?
                """
                params = (query, embedding_str, query, embedding_str, top_k)
            else:
                hybrid_sql = f"""
                SELECT id, document_text, metadata
                FROM {table}
                WHERE MATCH(document_text) AGAINST(? IN NATURAL LANGUAGE MODE) > 0
                   OR {distance_func}(embedding, VEC_FromText(?)) < 1.0
                ORDER BY ({text_weight} * (1 - MATCH(document_text) AGAINST(? IN NATURAL LANGUAGE MODE)) +
                         {vector_weight} * {distance_func}(embedding, VEC_FromText(?)))
                LIMIT ?
                """
                params = (query, embedding_str, query, embedding_str, top_k)
            
            results = self.db_manager.execute_query(hybrid_sql, params)
            
            logger.info(f"Hybrid search returned {len(results)} results")
            return results
            
        except Exception as e:
            error_msg = f"Hybrid search failed: {e}"
            logger.error(error_msg)
            raise SearchError(error_msg, query, table) from e
    
    def find_similar_documents(self, document_id: int, table: str, 
                              top_k: int = 5, exclude_self: bool = True,
                              include_distance: bool = False) -> List[Tuple]:
        """
        Find documents similar to a given document.
        
        Args:
            document_id: ID of the reference document
            table: Table to search
            top_k: Number of similar documents to return
            exclude_self: Whether to exclude the reference document
            include_distance: Whether to include distance scores
            
        Returns:
            List of similar document tuples
            
        Raises:
            SearchError: If search fails
        """
        try:
            logger.info(f"Finding documents similar to ID {document_id} in table '{table}'")
            
            # Get the reference document's embedding
            ref_query = f"SELECT embedding FROM {table} WHERE id = ?"
            ref_results = self.db_manager.execute_query(ref_query, (document_id,))
            
            if not ref_results:
                raise SearchError(f"Document with ID {document_id} not found", table=table)
            
            # Get distance function
            distance_type = self.table_distance_types.get(table, 'cosine')
            distance_func = self._get_distance_function(distance_type)
            
            # Build similarity search query
            exclude_clause = f"AND id != {document_id}" if exclude_self else ""
            
            if include_distance:
                similar_sql = f"""
                SELECT id, document_text, metadata,
                    {distance_func}(embedding, (SELECT embedding FROM {table} WHERE id = ?)) as distance
                FROM {table}
                WHERE 1=1 {exclude_clause}
                ORDER BY {distance_func}(embedding, (SELECT embedding FROM {table} WHERE id = ?))
                LIMIT ?
                """
                params = (document_id, document_id, top_k)
            else:
                similar_sql = f"""
                SELECT id, document_text, metadata
                FROM {table}
                WHERE 1=1 {exclude_clause}
                ORDER BY {distance_func}(embedding, (SELECT embedding FROM {table} WHERE id = ?))
                LIMIT ?
                """
                params = (document_id, top_k)
            
            results = self.db_manager.execute_query(similar_sql, params)
            
            logger.info(f"Found {len(results)} similar documents")
            return results
            
        except Exception as e:
            error_msg = f"Failed to find similar documents: {e}"
            logger.error(error_msg)
            raise SearchError(error_msg, table=table) from e
    
    def search_by_metadata(self, table: str, metadata_filter: Dict[str, Any],
                          top_k: int = 100) -> List[Tuple]:
        """
        Search documents by metadata criteria.
        
        Args:
            table: Table to search
            metadata_filter: Metadata filter criteria
            top_k: Maximum number of results
            
        Returns:
            List of matching documents
            
        Raises:
            SearchError: If search fails
        """
        try:
            logger.info(f"Searching by metadata in table '{table}': {metadata_filter}")
            
            # Build metadata filter conditions
            conditions = []
            params = []
            
            for key, value in metadata_filter.items():
                if isinstance(value, str):
                    conditions.append(f"JSON_EXTRACT(metadata, '$.{key}') = ?")
                    params.append(value)
                elif isinstance(value, (int, float)):
                    conditions.append(f"JSON_EXTRACT(metadata, '$.{key}') = ?")
                    params.append(value)
                elif isinstance(value, list):
                    # IN clause for list values
                    placeholders = ','.join(['?' for _ in value])
                    conditions.append(f"JSON_EXTRACT(metadata, '$.{key}') IN ({placeholders})")
                    params.extend(value)
                else:
                    logger.warning(f"Unsupported metadata filter type for key '{key}': {type(value)}")
            
            if not conditions:
                raise SearchError("No valid metadata filter conditions provided")
            
            where_clause = " AND ".join(conditions)
            metadata_sql = f"""
            SELECT id, document_text, metadata, created_at
            FROM {table}
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT ?
            """
            params.append(top_k)
            
            results = self.db_manager.execute_query(metadata_sql, tuple(params))
            
            logger.info(f"Metadata search returned {len(results)} results")
            return results
            
        except Exception as e:
            error_msg = f"Metadata search failed: {e}"
            logger.error(error_msg)
            raise SearchError(error_msg, table=table) from e
    
    def get_search_statistics(self, table: str) -> Dict[str, Any]:
        """
        Get search-related statistics for a table.
        
        Args:
            table: Table name
            
        Returns:
            Dict with statistics
            
        Raises:
            SearchError: If statistics retrieval fails
        """
        try:
            # Get basic table stats
            stats_sql = f"""
            SELECT 
                COUNT(*) as total_documents,
                AVG(CHAR_LENGTH(document_text)) as avg_text_length,
                MIN(created_at) as oldest_document,
                MAX(created_at) as newest_document
            FROM {table}
            """
            
            stats_result = self.db_manager.execute_query(stats_sql)
            
            if stats_result:
                stats = {
                    'total_documents': stats_result[0][0],
                    'avg_text_length': float(stats_result[0][1]) if stats_result[0][1] else 0,
                    'oldest_document': stats_result[0][2],
                    'newest_document': stats_result[0][3]
                }
            else:
                stats = {
                    'total_documents': 0,
                    'avg_text_length': 0,
                    'oldest_document': None,
                    'newest_document': None
                }
            
            # Get metadata field statistics
            metadata_sql = f"""
            SELECT metadata
            FROM {table}
            WHERE metadata IS NOT NULL
            LIMIT 1000
            """
            
            metadata_results = self.db_manager.execute_query(metadata_sql)
            metadata_fields = set()
            
            for row in metadata_results:
                try:
                    metadata = json.loads(row[0])
                    metadata_fields.update(metadata.keys())
                except:
                    continue
            
            stats['metadata_fields'] = list(metadata_fields)
            stats['table_config'] = self.table_distance_types.get(table, 'unknown')
            
            return stats
            
        except Exception as e:
            error_msg = f"Failed to get search statistics: {e}"
            logger.error(error_msg)
            raise SearchError(error_msg, table=table) from e
    
    def set_table_distance_type(self, table: str, distance_type: str) -> None:
        """Set the distance type for a table."""
        self.table_distance_types[table] = distance_type.lower()
    
    def get_table_distance_type(self, table: str) -> str:
        """Get the distance type for a table."""
        return self.table_distance_types.get(table, 'cosine')