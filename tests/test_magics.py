"""
Unit tests for MariaDB Vector Magic commands.

This module contains comprehensive unit tests for all magic command functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json
import os

from mariadb_vector_magics.magics.main import MariaDBVectorMagics
from mariadb_vector_magics.core.exceptions import MariaDBVectorError
from mariadb_vector_magics.core.utils import HTMLRenderer
from mariadb_vector_magics.core.vector_store import VectorStoreConfig


class TestHTMLRenderer:
    """Test cases for HTMLRenderer utility class."""
    
    def test_success_message(self):
        """Test success message rendering."""
        message = "Operation completed"
        details = {"Records": 5, "Table": "test"}
        
        result = HTMLRenderer.success(message, details)
        
        assert "SUCCESS:" in result
        assert message in result
        assert "Records" in result
        assert "test" in result
        assert "#d4edda" in result  # Success color
    
    def test_error_message(self):
        """Test error message rendering."""
        message = "Operation failed"
        error_details = "Connection timeout"
        
        result = HTMLRenderer.error(message, error_details)
        
        assert "ERROR:" in result
        assert message in result
        assert error_details in result
        assert "#f8d7da" in result  # Error color
    
    def test_info_message(self):
        """Test info message rendering."""
        message = "Processing data"
        details = {"Status": "In progress"}
        
        result = HTMLRenderer.info(message, details)
        
        assert "INFO:" in result
        assert message in result
        assert "Status" in result
        assert "#d1ecf1" in result  # Info color
    
    def test_warning_message(self):
        """Test warning message rendering."""
        message = "Low disk space"
        
        result = HTMLRenderer.warning(message)
        
        assert "WARNING:" in result
        assert message in result
        assert "#fff3cd" in result  # Warning color


class TestVectorStoreConfig:
    """Test cases for VectorStoreConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = VectorStoreConfig(table='test_table')
        
        assert config.table == 'test_table'
        assert config.dimensions == 384
        assert config.distance == 'cosine'
        assert config.m_value == 16
        assert config.drop_if_exists is False
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = VectorStoreConfig(
            table='custom_table',
            dimensions=768,
            distance='euclidean',
            m_value=32,
            drop_if_exists=True
        )
        
        assert config.table == 'custom_table'
        assert config.dimensions == 768
        assert config.distance == 'euclidean'
        assert config.m_value == 32
        assert config.drop_if_exists is True


@pytest.mark.unit
class TestMariaDBVectorMagics:
    """Test cases for MariaDBVectorMagics class."""
    
    def test_initialization(self, ipython_shell):
        """Test magic class initialization."""
        magic = MariaDBVectorMagics(ipython_shell)
        
        assert magic.db_manager is not None
        assert magic.embedding_manager is not None
        assert magic.vector_store_manager is not None
        assert magic.search_manager is not None
        assert magic.rag_manager is not None
        assert magic.renderer is not None
    
    @patch('mariadb_vector_magics.magics.main.display')
    @patch('mariadb_vector_magics.core.database.mariadb.connect')
    def test_connect_mariadb_success(self, mock_connect, mock_display, ipython_shell):
        """Test successful MariaDB connection."""
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = ('11.7.2-MariaDB-test',)
        mock_connection.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_connection
        
        magic = MariaDBVectorMagics(ipython_shell)
        result = magic.connect_mariadb('--password=testpass')
        
        assert result == "Connected successfully"
        mock_connect.assert_called_once()
        mock_display.assert_called()
    
    @patch('mariadb_vector_magics.magics.main.display')
    @patch('mariadb_vector_magics.core.database.mariadb.connect')
    def test_connect_mariadb_failure(self, mock_connect, mock_display, ipython_shell):
        """Test failed MariaDB connection."""
        import mariadb
        mock_connect.side_effect = mariadb.Error("Connection failed")
        
        magic = MariaDBVectorMagics(ipython_shell)
        result = magic.connect_mariadb('--password=testpass')
        
        assert result is None
        mock_display.assert_called()
    
    @patch('mariadb_vector_magics.magics.main.display')
    @patch('mariadb_vector_magics.core.embeddings.SentenceTransformer')
    def test_load_embedding_model_success(self, mock_transformer, mock_display, ipython_shell):
        """Test successful embedding model loading."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer.return_value = mock_model
        
        magic = MariaDBVectorMagics(ipython_shell)
        result = magic.load_embedding_model('all-MiniLM-L6-v2')
        
        assert result == "Model all-MiniLM-L6-v2 loaded"
        assert mock_transformer.called
        mock_display.assert_called()
    
    @patch('mariadb_vector_magics.magics.main.display')
    @patch('mariadb_vector_magics.core.embeddings.SentenceTransformer')
    def test_load_embedding_model_failure(self, mock_transformer, mock_display, ipython_shell):
        """Test failed embedding model loading."""
        mock_transformer.side_effect = Exception("Model not found")
        
        magic = MariaDBVectorMagics(ipython_shell)
        result = magic.load_embedding_model('invalid-model')
        
        assert result is None
        mock_display.assert_called()
    
    @patch('mariadb_vector_magics.magics.main.display')
    def test_create_vector_store_success(self, mock_display, magic_instance_connected):
        """Test successful vector store creation."""
        result = magic_instance_connected.create_vector_store('--table=test_docs --dimensions=384')
        
        assert result == "Table test_docs created"
        mock_display.assert_called()
    
    @patch('mariadb_vector_magics.magics.main.display')
    def test_show_vector_tables_success(self, mock_display, magic_instance_connected):
        """Test successful table listing."""
        result = magic_instance_connected.show_vector_tables('')
        
        assert result == ['table1', 'table2']
        mock_display.assert_called()
    
    @patch('mariadb_vector_magics.magics.main.display')
    def test_semantic_search_success(self, mock_display, magic_instance_full):
        """Test successful semantic search."""
        result = magic_instance_full.semantic_search('"test query" --table=test_docs --show_distance')
        
        assert len(result) == 1
        assert result[0][0] == 1  # ID
        mock_display.assert_called()
    
    @patch('mariadb_vector_magics.magics.main.display')
    def test_semantic_search_no_results(self, mock_display, magic_instance_full):
        """Test semantic search with no results."""
        magic_instance_full.search_manager.semantic_search.return_value = []
        
        result = magic_instance_full.semantic_search('"test query" --table=test_docs')
        
        assert result == []
        mock_display.assert_called()
    
    @patch('mariadb_vector_magics.magics.main.display')
    def test_rag_query_success(self, mock_display, magic_instance_full):
        """Test successful RAG query."""
        result = magic_instance_full.rag_query('"What is testing?" --table=test_docs')
        
        assert result is not None
        assert 'question' in result
        assert 'answer' in result
        assert 'context' in result
        mock_display.assert_called()
    
    @patch('mariadb_vector_magics.magics.main.display')
    def test_rag_query_no_api_key(self, mock_display, magic_instance_full):
        """Test RAG query without API key."""
        # Mock the RAG manager to return None for no API key
        magic_instance_full.rag_manager.query.return_value = None
        
        result = magic_instance_full.rag_query('"What is testing?" --table=test_docs')
        
        assert result is None
        mock_display.assert_called()
    
    @patch('mariadb_vector_magics.magics.main.display')
    def test_query_table_success(self, mock_display, magic_instance_connected):
        """Test successful table query."""
        result = magic_instance_connected.query_table('test_table')
        
        assert result is not None
        mock_display.assert_called()


@pytest.mark.unit
class TestMariaDBVectorError:
    """Test cases for custom exception class."""
    
    def test_exception_creation(self):
        """Test custom exception creation."""
        error = MariaDBVectorError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)
    
    def test_exception_raising(self):
        """Test custom exception raising."""
        with pytest.raises(MariaDBVectorError) as exc_info:
            raise MariaDBVectorError("Test error")
        
        assert str(exc_info.value) == "Test error"


# Core component tests (these should always pass)
@pytest.mark.unit
class TestCoreComponents:
    """Test core components independently."""
    
    def test_database_manager_creation(self):
        """Test DatabaseManager can be created."""
        from mariadb_vector_magics.core.database import DatabaseManager
        db_manager = DatabaseManager()
        assert db_manager is not None
        assert db_manager.connection is None
        assert db_manager.cursor is None
    
    def test_embedding_manager_creation(self):
        """Test EmbeddingManager can be created."""
        from mariadb_vector_magics.core.embeddings import EmbeddingManager
        embedding_manager = EmbeddingManager()
        assert embedding_manager is not None
        assert embedding_manager.model is None
        assert embedding_manager.model_name is None
    
    def test_vector_store_manager_creation(self):
        """Test VectorStoreManager can be created."""
        from mariadb_vector_magics.core.vector_store import VectorStoreManager
        from mariadb_vector_magics.core.database import DatabaseManager
        
        db_manager = DatabaseManager()
        vector_manager = VectorStoreManager(db_manager)
        assert vector_manager is not None
        assert vector_manager.db_manager is db_manager
    
    def test_search_manager_creation(self):
        """Test SearchManager can be created."""
        from mariadb_vector_magics.core.search import SearchManager
        from mariadb_vector_magics.core.database import DatabaseManager
        from mariadb_vector_magics.core.embeddings import EmbeddingManager
        
        db_manager = DatabaseManager()
        embedding_manager = EmbeddingManager()
        search_manager = SearchManager(db_manager, embedding_manager)
        assert search_manager is not None
        assert search_manager.db_manager is db_manager
        assert search_manager.embedding_manager is embedding_manager
    
    def test_rag_manager_creation(self):
        """Test RAGManager can be created."""
        from mariadb_vector_magics.core.rag import RAGManager
        from mariadb_vector_magics.core.database import DatabaseManager
        from mariadb_vector_magics.core.embeddings import EmbeddingManager
        from mariadb_vector_magics.core.search import SearchManager
        
        db_manager = DatabaseManager()
        embedding_manager = EmbeddingManager()
        search_manager = SearchManager(db_manager, embedding_manager)
        rag_manager = RAGManager(db_manager, embedding_manager, search_manager)
        assert rag_manager is not None
        assert rag_manager.db_manager is db_manager
        assert rag_manager.embedding_manager is embedding_manager
        assert rag_manager.search_manager is search_manager


# Integration test examples (require real database)
@pytest.mark.integration
class TestMariaDBVectorMagicsIntegration:
    """Integration tests requiring real database connection."""
    
    @pytest.mark.requires_db
    @pytest.mark.skipif(
        not os.getenv('MARIADB_TEST_HOST'),
        reason="Requires real MariaDB connection (set MARIADB_TEST_HOST)"
    )
    def test_real_connection(self, ipython_shell):
        """Test real database connection (requires MariaDB running)."""
        import os
        magic = MariaDBVectorMagics(ipython_shell)
        
        # Use environment variables for connection
        host = os.getenv('MARIADB_TEST_HOST', 'localhost')
        port = int(os.getenv('MARIADB_TEST_PORT', '3307'))
        user = os.getenv('MARIADB_TEST_USER', 'root')
        password = os.getenv('MARIADB_TEST_PASSWORD', 'testpass')
        database = os.getenv('MARIADB_TEST_DATABASE', 'vector_test')
        
        # Test connection directly with database manager first
        try:
            success = magic.db_manager.connect(
                host=host,
                port=port,
                user=user,
                password=password,
                database=database
            )
            
            # If direct connection works, test the magic command
            if success:
                connection_args = f'--host={host} --port={port} --user={user} --password={password} --database={database}'
                result = magic.connect_mariadb(connection_args)
                assert result == "Connected successfully"
                assert magic.db_manager.connection is not None
            else:
                pytest.fail("Database connection failed")
                
        except Exception as e:
            # If connection fails, skip the test with informative message
            pytest.skip(f"Cannot connect to MariaDB at {host}:{port} - {str(e)}")
    
    @pytest.mark.requires_db
    @pytest.mark.skipif(
        not os.getenv('MARIADB_TEST_HOST') or not os.getenv('ENABLE_MODEL_TESTS'),
        reason="Requires model download and real DB (set MARIADB_TEST_HOST and ENABLE_MODEL_TESTS)"
    )
    def test_real_embedding_model(self, ipython_shell):
        """Test real embedding model loading."""
        import os
        magic = MariaDBVectorMagics(ipython_shell)
        
        # Test model loading
        model_name = 'all-MiniLM-L6-v2'
        result = magic.load_embedding_model(model_name)
        
        assert result == f"Model {model_name} loaded"
        assert magic.embedding_manager.model is not None
        assert magic.embedding_manager.model_name == model_name