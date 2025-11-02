"""
Custom exceptions for MariaDB Vector Magics.

This module defines all custom exceptions used throughout the application.
"""

from typing import Optional


class MariaDBVectorError(Exception):
    """Base exception for MariaDB Vector operations."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, cause: Optional[Exception] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.cause = cause
    
    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class DatabaseConnectionError(MariaDBVectorError):
    """Raised when database connection fails."""
    
    def __init__(self, message: str, host: Optional[str] = None, port: Optional[int] = None):
        super().__init__(message, "DB_CONNECTION_ERROR")
        self.host = host
        self.port = port


class DatabaseOperationError(MariaDBVectorError):
    """Raised when database operations fail."""
    
    def __init__(self, message: str, operation: Optional[str] = None, table: Optional[str] = None):
        super().__init__(message, "DB_OPERATION_ERROR")
        self.operation = operation
        self.table = table


class EmbeddingError(MariaDBVectorError):
    """Raised when embedding operations fail."""
    
    def __init__(self, message: str, model_name: Optional[str] = None):
        super().__init__(message, "EMBEDDING_ERROR")
        self.model_name = model_name


class VectorStoreError(MariaDBVectorError):
    """Raised when vector store operations fail."""
    
    def __init__(self, message: str, table_name: Optional[str] = None):
        super().__init__(message, "VECTOR_STORE_ERROR")
        self.table_name = table_name


class SearchError(MariaDBVectorError):
    """Raised when search operations fail."""
    
    def __init__(self, message: str, query: Optional[str] = None, table: Optional[str] = None):
        super().__init__(message, "SEARCH_ERROR")
        self.query = query
        self.table = table


class RAGError(MariaDBVectorError):
    """Raised when RAG operations fail."""
    
    def __init__(self, message: str, question: Optional[str] = None, model: Optional[str] = None):
        super().__init__(message, "RAG_ERROR")
        self.question = question
        self.model = model


class ConfigurationError(MariaDBVectorError):
    """Raised when configuration is invalid."""
    
    def __init__(self, message: str, config_key: Optional[str] = None):
        super().__init__(message, "CONFIG_ERROR")
        self.config_key = config_key


class ValidationError(MariaDBVectorError):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Optional[str] = None):
        super().__init__(message, "VALIDATION_ERROR")
        self.field = field
        self.value = value


class SecretsError(MariaDBVectorError):
    """Raised when secrets management fails."""
    
    def __init__(self, message: str, secret_name: Optional[str] = None):
        super().__init__(message, "SECRETS_ERROR")
        self.secret_name = secret_name