"""
Database management for MariaDB Vector operations.

This module handles all database connection and basic operations.
"""

import logging
from typing import Optional, Dict, Any, List, Tuple
from contextlib import contextmanager

try:
    import mariadb
except ImportError:
    raise ImportError("mariadb package is required. Install with: pip install mariadb")

from .exceptions import DatabaseConnectionError, DatabaseOperationError
from .utils import ValidationUtils

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages MariaDB database connections and operations."""
    
    def __init__(self):
        self.connection: Optional[mariadb.Connection] = None
        self.cursor: Optional[mariadb.Cursor] = None
        self._connection_params: Optional[Dict[str, Any]] = None
    
    def connect(self, host: str = '127.0.0.1', port: int = 3306, user: str = 'root', 
                password: str = '', database: str = 'vectordb', **kwargs) -> bool:
        """
        Establish connection to MariaDB server.
        
        Args:
            host: MariaDB host address
            port: MariaDB port number
            user: MariaDB username
            password: MariaDB password
            database: Database name
            **kwargs: Additional connection parameters
            
        Returns:
            bool: True if connection successful
            
        Raises:
            DatabaseConnectionError: If connection fails
        """
        try:
            # Store connection parameters for reconnection
            self._connection_params = {
                'host': host,
                'port': port,
                'user': user,
                'password': password,
                'database': database,
                'local_infile': kwargs.get('local_infile', True),
                **kwargs
            }
            
            logger.info(f"Connecting to MariaDB at {host}:{port}")
            
            self.connection = mariadb.connect(**self._connection_params)
            self.cursor = self.connection.cursor()
            
            # Verify connection and get version
            version_info = self.get_version()
            logger.info(f"Connected to MariaDB {version_info}")
            
            # Test Vector support
            vector_support = self.test_vector_support()
            if vector_support:
                logger.info("Vector support confirmed")
            else:
                logger.warning("Vector support not available")
            
            return True
            
        except mariadb.Error as e:
            error_msg = f"Failed to connect to MariaDB: {str(e)}"
            logger.error(error_msg)
            raise DatabaseConnectionError(error_msg, host, port) from e
        except Exception as e:
            error_msg = f"Unexpected error during connection: {str(e)}"
            logger.error(error_msg)
            raise DatabaseConnectionError(error_msg, host, port) from e
    
    def disconnect(self) -> None:
        """Close database connection."""
        try:
            if self.cursor:
                self.cursor.close()
                self.cursor = None
            
            if self.connection:
                self.connection.close()
                self.connection = None
                
            logger.info("Database connection closed")
            
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")
    
    def reconnect(self) -> bool:
        """Reconnect using stored connection parameters."""
        if not self._connection_params:
            raise DatabaseConnectionError("No connection parameters stored for reconnection")
        
        self.disconnect()
        return self.connect(**self._connection_params)
    
    def refresh_cursor(self) -> None:
        """Refresh the cursor to ensure it's in a clean state."""
        if self.connection and self.is_connected():
            try:
                if self.cursor:
                    self.cursor.close()
                self.cursor = self.connection.cursor()
                logger.debug("Cursor refreshed")
            except Exception as e:
                logger.warning(f"Failed to refresh cursor: {e}")
    
    def is_connected(self) -> bool:
        """Check if database connection is active."""
        try:
            if not self.connection or not self.cursor:
                return False
            
            # Test connection with a simple query
            self.cursor.execute("SELECT 1")
            self.cursor.fetchone()
            return True
            
        except Exception:
            # Try to reconnect if connection is lost
            if self._connection_params:
                try:
                    logger.info("Connection lost, attempting to reconnect...")
                    return self.reconnect()
                except Exception:
                    return False
            return False
    
    def get_version(self) -> str:
        """Get MariaDB version information."""
        if not self.is_connected():
            raise DatabaseConnectionError("Not connected to database")
        
        try:
            self.cursor.execute("SELECT VERSION()")
            version = self.cursor.fetchone()[0]
            return version
        except mariadb.Error as e:
            raise DatabaseOperationError(f"Failed to get version: {e}", "get_version")
    
    def test_vector_support(self) -> bool:
        """Test if Vector support is available."""
        if not self.is_connected():
            return False
        
        try:
            self.cursor.execute("SELECT VEC_FromText('[1,2,3]') as test_vector")
            result = self.cursor.fetchone()
            return result is not None
        except mariadb.Error:
            return False
    
    def execute_query(self, query: str, params: Optional[Tuple] = None) -> List[Tuple]:
        """
        Execute a SELECT query and return results.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            List of result tuples
            
        Raises:
            DatabaseOperationError: If query execution fails
        """
        if not self.is_connected():
            raise DatabaseConnectionError("Not connected to database")
        
        try:
            logger.debug(f"Executing query: {query}")
            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)
            
            results = self.cursor.fetchall()
            logger.debug(f"Query returned {len(results)} rows")
            return results
            
        except mariadb.Error as e:
            error_msg = f"Query execution failed: {e}"
            logger.error(error_msg)
            raise DatabaseOperationError(error_msg, "execute_query") from e
    
    def execute_command(self, command: str, params: Optional[Tuple] = None, commit: bool = True) -> int:
        """
        Execute a non-SELECT command (INSERT, UPDATE, DELETE, etc.).
        
        Args:
            command: SQL command string
            params: Command parameters
            commit: Whether to commit the transaction
            
        Returns:
            Number of affected rows
            
        Raises:
            DatabaseOperationError: If command execution fails
        """
        if not self.is_connected():
            raise DatabaseConnectionError("Not connected to database")
        
        try:
            logger.debug(f"Executing command: {command}")
            if params:
                self.cursor.execute(command, params)
            else:
                self.cursor.execute(command)
            
            affected_rows = self.cursor.rowcount
            
            # Only commit if not in a transaction context and commit is requested
            if commit and self.connection.autocommit:
                self.connection.commit()
                logger.debug(f"Command committed, {affected_rows} rows affected")
            elif not self.connection.autocommit:
                logger.debug(f"Command executed in transaction, {affected_rows} rows affected")
            
            return affected_rows
            
        except mariadb.Error as e:
            # Only rollback if not in a transaction context and commit was requested
            if commit and self.connection.autocommit:
                try:
                    self.connection.rollback()
                except Exception:
                    pass  # Rollback might fail if autocommit is on
            error_msg = f"Command execution failed: {e}"
            logger.error(error_msg)
            raise DatabaseOperationError(error_msg, "execute_command") from e
    
    def list_tables(self) -> List[str]:
        """List all tables in the current database."""
        try:
            results = self.execute_query("SHOW TABLES")
            return [table[0] for table in results]
        except DatabaseOperationError as e:
            logger.error(f"Failed to list tables: {e}")
            raise
    
    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists."""
        # Validate table name
        is_valid, error_msg = ValidationUtils.validate_table_name(table_name)
        if not is_valid:
            raise DatabaseOperationError(f"Invalid table name: {error_msg}")
        
        try:
            results = self.execute_query("SHOW TABLES LIKE %s", (table_name,))
            return len(results) > 0
        except DatabaseOperationError as e:
            logger.error(f"Failed to check table existence: {e}")
            raise
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get information about a table."""
        if not self.table_exists(table_name):
            raise DatabaseOperationError(f"Table '{table_name}' does not exist")
        
        try:
            # Get row count
            count_result = self.execute_query(f"SELECT COUNT(*) FROM {table_name}")
            row_count = count_result[0][0] if count_result else 0
            
            # Get table structure
            structure_result = self.execute_query(f"DESCRIBE {table_name}")
            
            # Get sample data
            sample_result = self.execute_query(
                f"SELECT id, LEFT(document_text, 100) as text_preview FROM {table_name} LIMIT 5"
            )
            
            return {
                'name': table_name,
                'row_count': row_count,
                'structure': structure_result,
                'sample_data': sample_result
            }
            
        except DatabaseOperationError as e:
            logger.error(f"Failed to get table info: {e}")
            raise
    
    def drop_table(self, table_name: str) -> bool:
        """Drop a table if it exists."""
        # Validate table name
        is_valid, error_msg = ValidationUtils.validate_table_name(table_name)
        if not is_valid:
            raise DatabaseOperationError(f"Invalid table name: {error_msg}")
        
        try:
            self.execute_command(f"DROP TABLE IF EXISTS {table_name}")
            logger.info(f"Dropped table: {table_name}")
            return True
        except DatabaseOperationError as e:
            logger.error(f"Failed to drop table {table_name}: {e}")
            raise
    
    @contextmanager
    def transaction(self):
        """Context manager for database transactions."""
        if not self.is_connected():
            raise DatabaseConnectionError("Not connected to database")
        
        # Store original autocommit state
        original_autocommit = self.connection.autocommit
        
        try:
            # Start transaction (MariaDB uses autocommit=False)
            self.connection.autocommit = False
            yield self
            self.connection.commit()
            logger.debug("Transaction committed")
        except Exception as e:
            try:
                self.connection.rollback()
                logger.error(f"Transaction rolled back due to error: {e}")
            except Exception as rollback_error:
                logger.error(f"Failed to rollback transaction: {rollback_error}")
            raise
        finally:
            # Restore original autocommit state
            try:
                self.connection.autocommit = original_autocommit
            except Exception as e:
                logger.error(f"Failed to restore autocommit state: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()