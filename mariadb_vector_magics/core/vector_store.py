"""
Vector store management for MariaDB Vector operations.

This module handles vector table creation, management, and operations.
"""

import logging
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .database import DatabaseManager
from .exceptions import VectorStoreError, DatabaseOperationError
from .utils import ValidationUtils

logger = logging.getLogger(__name__)


@dataclass
class VectorStoreConfig:
    """Configuration for vector store creation."""
    table: str
    dimensions: int = 384
    distance: str = 'cosine'
    m_value: int = 16
    drop_if_exists: bool = False


class VectorStoreManager:
    """Manages vector store operations in MariaDB."""
    
    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize vector store manager.
        
        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager
        self.table_configs: Dict[str, VectorStoreConfig] = {}
    
    def create_vector_store(self, config: VectorStoreConfig) -> Dict[str, Any]:
        """
        Create a vector store table with optimized configuration.
        
        Args:
            config: Vector store configuration
            
        Returns:
            Dict with creation details
            
        Raises:
            VectorStoreError: If creation fails
        """
        try:
            # Validate configuration
            self._validate_config(config)
            
            logger.info(f"Creating vector store: {config.table}")
            
            # Drop table if requested
            if config.drop_if_exists:
                self.db_manager.drop_table(config.table)
                logger.info(f"Dropped existing table: {config.table}")
            
            # Create the vector table
            create_sql = self._build_create_table_sql(config)
            self.db_manager.execute_command(create_sql)
            
            # Store configuration
            self.table_configs[config.table] = config
            
            logger.info(f"Vector store created successfully: {config.table}")
            
            return {
                'table': config.table,
                'dimensions': config.dimensions,
                'distance_metric': config.distance,
                'hnsw_m_value': config.m_value,
                'schema': self._get_table_schema()
            }
            
        except DatabaseOperationError as e:
            error_msg = f"Failed to create vector store '{config.table}': {e}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg, config.table) from e
        except Exception as e:
            error_msg = f"Unexpected error creating vector store '{config.table}': {e}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg, config.table) from e
    
    def _validate_config(self, config: VectorStoreConfig) -> None:
        """Validate vector store configuration."""
        # Validate table name
        is_valid, error_msg = ValidationUtils.validate_table_name(config.table)
        if not is_valid:
            raise VectorStoreError(f"Invalid table name: {error_msg}", config.table)
        
        # Validate dimensions
        is_valid, error_msg = ValidationUtils.validate_dimensions(config.dimensions)
        if not is_valid:
            raise VectorStoreError(f"Invalid dimensions: {error_msg}", config.table)
        
        # Validate distance metric
        is_valid, error_msg = ValidationUtils.validate_distance_metric(config.distance)
        if not is_valid:
            raise VectorStoreError(f"Invalid distance metric: {error_msg}", config.table)
        
        # Validate M value
        is_valid, error_msg = ValidationUtils.validate_m_value(config.m_value)
        if not is_valid:
            raise VectorStoreError(f"Invalid M value: {error_msg}", config.table)
    
    def _build_create_table_sql(self, config: VectorStoreConfig) -> str:
        """Build CREATE TABLE SQL for vector store."""
        return f"""
        CREATE TABLE {config.table} (
            id INT AUTO_INCREMENT PRIMARY KEY,
            document_text LONGTEXT NOT NULL,
            metadata JSON,
            embedding VECTOR({config.dimensions}) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            VECTOR INDEX (embedding) M={config.m_value} DISTANCE={config.distance}
        )
        """
    
    def _get_table_schema(self) -> List[str]:
        """Get the standard vector table schema."""
        return [
            'id - Auto-increment primary key',
            'document_text - Text content (LONGTEXT)',
            'metadata - JSON metadata',
            'embedding - Vector embeddings with HNSW index',
            'created_at - Creation timestamp',
            'updated_at - Last update timestamp'
        ]
    
    def insert_embeddings(self, table: str, documents: List[str], 
                         embeddings: List[List[float]], 
                         metadata_list: Optional[List[Dict[str, Any]]] = None) -> int:
        """
        Insert documents with embeddings into vector store.
        
        Args:
            table: Target table name
            documents: List of document texts
            embeddings: List of embedding vectors
            metadata_list: Optional list of metadata dicts
            
        Returns:
            Number of records inserted
            
        Raises:
            VectorStoreError: If insertion fails
        """
        if not self.db_manager.table_exists(table):
            raise VectorStoreError(f"Table '{table}' does not exist", table)
        
        if len(documents) != len(embeddings):
            raise VectorStoreError(
                f"Mismatch between documents ({len(documents)}) and embeddings ({len(embeddings)})",
                table
            )
        
        if metadata_list and len(metadata_list) != len(documents):
            raise VectorStoreError(
                f"Mismatch between documents ({len(documents)}) and metadata ({len(metadata_list)})",
                table
            )
        
        try:
            logger.info(f"Inserting {len(documents)} records into {table}")
            
            insert_sql = f"""
            INSERT INTO {table} (document_text, metadata, embedding)
            VALUES (?, ?, VEC_FromText(?))
            """
            
            inserted_count = 0
            
            with self.db_manager.transaction():
                for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                    # Prepare metadata
                    metadata = metadata_list[i] if metadata_list else {}
                    metadata_json = json.dumps(metadata)
                    
                    # Convert embedding to string format
                    embedding_str = '[' + ','.join(map(str, embedding)) + ']'
                    
                    # Execute insert
                    self.db_manager.execute_command(
                        insert_sql,
                        (doc, metadata_json, embedding_str),
                        commit=False
                    )
                    inserted_count += 1
            
            logger.info(f"Successfully inserted {inserted_count} records into {table}")
            return inserted_count
            
        except Exception as e:
            error_msg = f"Failed to insert embeddings into '{table}': {e}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg, table) from e
    
    def get_table_info(self, table: str) -> Dict[str, Any]:
        """
        Get information about a vector table.
        
        Args:
            table: Table name
            
        Returns:
            Dict with table information
            
        Raises:
            VectorStoreError: If table access fails
        """
        try:
            info = self.db_manager.get_table_info(table)
            
            # Add vector-specific information
            config = self.table_configs.get(table)
            if config:
                info['vector_config'] = {
                    'dimensions': config.dimensions,
                    'distance_metric': config.distance,
                    'hnsw_m_value': config.m_value
                }
            
            return info
            
        except DatabaseOperationError as e:
            raise VectorStoreError(f"Failed to get table info for '{table}': {e}", table) from e
    
    def list_vector_tables(self) -> List[str]:
        """
        List all vector tables.
        
        Returns:
            List of table names
        """
        try:
            all_tables = self.db_manager.list_tables()
            
            # Filter for vector tables (tables with VECTOR columns)
            vector_tables = []
            for table in all_tables:
                try:
                    # Check if table has vector column
                    structure = self.db_manager.execute_query(f"DESCRIBE {table}")
                    for column_info in structure:
                        if 'VECTOR' in str(column_info[1]).upper():
                            vector_tables.append(table)
                            break
                except Exception:
                    # Skip tables we can't access
                    continue
            
            return vector_tables
            
        except Exception as e:
            error_msg = f"Failed to list vector tables: {e}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg) from e
    
    def delete_records(self, table: str, condition: str, params: Optional[Tuple] = None) -> int:
        """
        Delete records from vector table.
        
        Args:
            table: Table name
            condition: WHERE condition
            params: Query parameters
            
        Returns:
            Number of deleted records
            
        Raises:
            VectorStoreError: If deletion fails
        """
        if not self.db_manager.table_exists(table):
            raise VectorStoreError(f"Table '{table}' does not exist", table)
        
        try:
            delete_sql = f"DELETE FROM {table} WHERE {condition}"
            deleted_count = self.db_manager.execute_command(delete_sql, params)
            
            logger.info(f"Deleted {deleted_count} records from {table}")
            return deleted_count
            
        except Exception as e:
            error_msg = f"Failed to delete records from '{table}': {e}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg, table) from e
    
    def update_metadata(self, table: str, record_id: int, metadata: Dict[str, Any]) -> bool:
        """
        Update metadata for a specific record.
        
        Args:
            table: Table name
            record_id: Record ID
            metadata: New metadata
            
        Returns:
            True if update successful
            
        Raises:
            VectorStoreError: If update fails
        """
        if not self.db_manager.table_exists(table):
            raise VectorStoreError(f"Table '{table}' does not exist", table)
        
        try:
            metadata_json = json.dumps(metadata)
            update_sql = f"UPDATE {table} SET metadata = ? WHERE id = ?"
            
            updated_count = self.db_manager.execute_command(
                update_sql, 
                (metadata_json, record_id)
            )
            
            if updated_count == 0:
                logger.warning(f"No record found with ID {record_id} in table {table}")
                return False
            
            logger.info(f"Updated metadata for record {record_id} in {table}")
            return True
            
        except Exception as e:
            error_msg = f"Failed to update metadata in '{table}': {e}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg, table) from e
    
    def get_record_by_id(self, table: str, record_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a specific record by ID.
        
        Args:
            table: Table name
            record_id: Record ID
            
        Returns:
            Record data or None if not found
            
        Raises:
            VectorStoreError: If query fails
        """
        if not self.db_manager.table_exists(table):
            raise VectorStoreError(f"Table '{table}' does not exist", table)
        
        try:
            query_sql = f"""
            SELECT id, document_text, metadata, created_at, updated_at
            FROM {table} WHERE id = ?
            """
            
            results = self.db_manager.execute_query(query_sql, (record_id,))
            
            if not results:
                return None
            
            row = results[0]
            return {
                'id': row[0],
                'document_text': row[1],
                'metadata': json.loads(row[2]) if row[2] else {},
                'created_at': row[3],
                'updated_at': row[4]
            }
            
        except Exception as e:
            error_msg = f"Failed to get record {record_id} from '{table}': {e}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg, table) from e
    
    def optimize_table(self, table: str) -> Dict[str, Any]:
        """
        Optimize vector table for better performance.
        
        Args:
            table: Table name
            
        Returns:
            Optimization results
            
        Raises:
            VectorStoreError: If optimization fails
        """
        if not self.db_manager.table_exists(table):
            raise VectorStoreError(f"Table '{table}' does not exist", table)
        
        try:
            logger.info(f"Optimizing table: {table}")
            
            # Run OPTIMIZE TABLE
            self.db_manager.execute_command(f"OPTIMIZE TABLE {table}")
            
            # Get updated table info
            info = self.get_table_info(table)
            
            logger.info(f"Table optimization completed: {table}")
            return {
                'table': table,
                'status': 'optimized',
                'row_count': info.get('row_count', 0)
            }
            
        except Exception as e:
            error_msg = f"Failed to optimize table '{table}': {e}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg, table) from e
    
    def get_table_config(self, table: str) -> Optional[VectorStoreConfig]:
        """Get stored configuration for a table."""
        return self.table_configs.get(table)
    
    def set_table_config(self, table: str, config: VectorStoreConfig) -> None:
        """Set configuration for a table."""
        self.table_configs[table] = config