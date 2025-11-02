"""
Configuration management for MariaDB Vector Magics.

This module provides centralized configuration management with support for
environment variables, configuration files, and default values.
"""

import os
import configparser
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DatabaseConfig:
    """Database connection configuration."""
    host: str = '127.0.0.1'
    port: int = 3306
    user: str = 'root'
    password: str = ''
    database: str = 'vectordb'
    local_infile: bool = True
    
    @classmethod
    def from_env(cls) -> 'DatabaseConfig':
        """Create configuration from environment variables."""
        return cls(
            host=os.getenv('MARIADB_HOST', cls.host),
            port=int(os.getenv('MARIADB_PORT', str(cls.port))),
            user=os.getenv('MARIADB_USER', cls.user),
            password=os.getenv('MARIADB_PASSWORD', cls.password),
            database=os.getenv('MARIADB_DATABASE', cls.database),
            local_infile=os.getenv('MARIADB_LOCAL_INFILE', 'true').lower() == 'true'
        )


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    model_name: str = 'all-MiniLM-L6-v2'
    batch_size: int = 32
    max_seq_length: int = 256
    cache_folder: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> 'EmbeddingConfig':
        """Create configuration from environment variables."""
        return cls(
            model_name=os.getenv('EMBEDDING_MODEL', cls.model_name),
            batch_size=int(os.getenv('EMBEDDING_BATCH_SIZE', str(cls.batch_size))),
            max_seq_length=int(os.getenv('EMBEDDING_MAX_SEQ_LENGTH', str(cls.max_seq_length))),
            cache_folder=os.getenv('EMBEDDING_CACHE_FOLDER')
        )


@dataclass
class VectorConfig:
    """Vector store configuration."""
    default_dimensions: int = 384
    default_distance: str = 'cosine'
    default_m_value: int = 16
    chunk_size: int = 500
    chunk_overlap: int = 50
    
    @classmethod
    def from_env(cls) -> 'VectorConfig':
        """Create configuration from environment variables."""
        return cls(
            default_dimensions=int(os.getenv('VECTOR_DIMENSIONS', str(cls.default_dimensions))),
            default_distance=os.getenv('VECTOR_DISTANCE', cls.default_distance),
            default_m_value=int(os.getenv('VECTOR_M_VALUE', str(cls.default_m_value))),
            chunk_size=int(os.getenv('VECTOR_CHUNK_SIZE', str(cls.chunk_size))),
            chunk_overlap=int(os.getenv('VECTOR_CHUNK_OVERLAP', str(cls.chunk_overlap)))
        )


@dataclass
class RAGConfig:
    """RAG (Retrieval Augmented Generation) configuration."""
    api_key: Optional[str] = None
    model_name: str = 'llama-3.3-70b-versatile'
    temperature: float = 0.2
    max_tokens: int = 1024
    top_k: int = 3
    
    @classmethod
    def from_env(cls) -> 'RAGConfig':
        """Create configuration from environment variables."""
        return cls(
            api_key=os.getenv('GROQ_API_KEY'),
            model_name=os.getenv('RAG_MODEL', cls.model_name),
            temperature=float(os.getenv('RAG_TEMPERATURE', str(cls.temperature))),
            max_tokens=int(os.getenv('RAG_MAX_TOKENS', str(cls.max_tokens))),
            top_k=int(os.getenv('RAG_TOP_K', str(cls.top_k)))
        )


@dataclass
class AppConfig:
    """Main application configuration."""
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector: VectorConfig = field(default_factory=VectorConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    debug: bool = False
    log_level: str = 'INFO'
    
    @classmethod
    def from_env(cls) -> 'AppConfig':
        """Create configuration from environment variables."""
        return cls(
            database=DatabaseConfig.from_env(),
            embedding=EmbeddingConfig.from_env(),
            vector=VectorConfig.from_env(),
            rag=RAGConfig.from_env(),
            debug=os.getenv('DEBUG', 'false').lower() == 'true',
            log_level=os.getenv('LOG_LEVEL', 'INFO').upper()
        )
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> 'AppConfig':
        """Load configuration from INI file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        parser = configparser.ConfigParser()
        parser.read(config_path)
        
        # Database configuration
        db_section = parser.get('database', fallback={})
        database = DatabaseConfig(
            host=db_section.get('host', DatabaseConfig.host),
            port=int(db_section.get('port', str(DatabaseConfig.port))),
            user=db_section.get('user', DatabaseConfig.user),
            password=db_section.get('password', DatabaseConfig.password),
            database=db_section.get('database', DatabaseConfig.database),
            local_infile=db_section.get('local_infile', 'true').lower() == 'true'
        )
        
        # Embedding configuration
        emb_section = parser.get('embedding', fallback={})
        embedding = EmbeddingConfig(
            model_name=emb_section.get('model_name', EmbeddingConfig.model_name),
            batch_size=int(emb_section.get('batch_size', str(EmbeddingConfig.batch_size))),
            max_seq_length=int(emb_section.get('max_seq_length', str(EmbeddingConfig.max_seq_length))),
            cache_folder=emb_section.get('cache_folder')
        )
        
        # Vector configuration
        vec_section = parser.get('vector', fallback={})
        vector = VectorConfig(
            default_dimensions=int(vec_section.get('default_dimensions', str(VectorConfig.default_dimensions))),
            default_distance=vec_section.get('default_distance', VectorConfig.default_distance),
            default_m_value=int(vec_section.get('default_m_value', str(VectorConfig.default_m_value))),
            chunk_size=int(vec_section.get('chunk_size', str(VectorConfig.chunk_size))),
            chunk_overlap=int(vec_section.get('chunk_overlap', str(VectorConfig.chunk_overlap)))
        )
        
        # RAG configuration
        rag_section = parser.get('rag', fallback={})
        rag = RAGConfig(
            api_key=rag_section.get('api_key'),
            model_name=rag_section.get('model_name', RAGConfig.model_name),
            temperature=float(rag_section.get('temperature', str(RAGConfig.temperature))),
            max_tokens=int(rag_section.get('max_tokens', str(RAGConfig.max_tokens))),
            top_k=int(rag_section.get('top_k', str(RAGConfig.top_k)))
        )
        
        # App configuration
        app_section = parser.get('app', fallback={})
        return cls(
            database=database,
            embedding=embedding,
            vector=vector,
            rag=rag,
            debug=app_section.get('debug', 'false').lower() == 'true',
            log_level=app_section.get('log_level', 'INFO').upper()
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'database': {
                'host': self.database.host,
                'port': self.database.port,
                'user': self.database.user,
                'database': self.database.database,
                'local_infile': self.database.local_infile
            },
            'embedding': {
                'model_name': self.embedding.model_name,
                'batch_size': self.embedding.batch_size,
                'max_seq_length': self.embedding.max_seq_length,
                'cache_folder': self.embedding.cache_folder
            },
            'vector': {
                'default_dimensions': self.vector.default_dimensions,
                'default_distance': self.vector.default_distance,
                'default_m_value': self.vector.default_m_value,
                'chunk_size': self.vector.chunk_size,
                'chunk_overlap': self.vector.chunk_overlap
            },
            'rag': {
                'model_name': self.rag.model_name,
                'temperature': self.rag.temperature,
                'max_tokens': self.rag.max_tokens,
                'top_k': self.rag.top_k
            },
            'app': {
                'debug': self.debug,
                'log_level': self.log_level
            }
        }


class ConfigManager:
    """Configuration manager with multiple sources support."""
    
    def __init__(self):
        self._config: Optional[AppConfig] = None
        self._config_file: Optional[Path] = None
    
    def load_config(self, config_file: Optional[Union[str, Path]] = None) -> AppConfig:
        """
        Load configuration from multiple sources in priority order:
        1. Configuration file (if provided)
        2. Environment variables
        3. Default values
        """
        if config_file:
            config_file = Path(config_file)
            if config_file.exists():
                self._config = AppConfig.from_file(config_file)
                self._config_file = config_file
                return self._config
        
        # Look for default config files
        default_locations = [
            Path.cwd() / 'mariadb_vector.ini',
            Path.home() / '.mariadb_vector.ini',
            Path('/etc/mariadb_vector.ini')
        ]
        
        for location in default_locations:
            if location.exists():
                self._config = AppConfig.from_file(location)
                self._config_file = location
                return self._config
        
        # Fall back to environment variables and defaults
        self._config = AppConfig.from_env()
        return self._config
    
    @property
    def config(self) -> AppConfig:
        """Get current configuration, loading defaults if not already loaded."""
        if self._config is None:
            self.load_config()
        return self._config
    
    def reload_config(self) -> AppConfig:
        """Reload configuration from the same source."""
        if self._config_file:
            return self.load_config(self._config_file)
        else:
            return self.load_config()
    
    def save_config(self, config_file: Union[str, Path]) -> None:
        """Save current configuration to file."""
        if self._config is None:
            raise ValueError("No configuration loaded")
        
        config_file = Path(config_file)
        parser = configparser.ConfigParser()
        
        # Convert config to ConfigParser format
        config_dict = self._config.to_dict()
        for section_name, section_data in config_dict.items():
            parser[section_name] = {k: str(v) for k, v in section_data.items() if v is not None}
        
        with open(config_file, 'w') as f:
            parser.write(f)


# Global configuration manager instance
config_manager = ConfigManager()


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    return config_manager.config


def load_config(config_file: Optional[Union[str, Path]] = None) -> AppConfig:
    """Load configuration from file or environment."""
    return config_manager.load_config(config_file)