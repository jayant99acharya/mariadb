"""
Pytest configuration and fixtures for MariaDB Vector Magics tests.

This module provides shared fixtures and configuration for the test suite.
"""

import pytest
import os
import tempfile
from unittest.mock import Mock, MagicMock
from typing import Generator, Dict, Any

# Test configuration
TEST_CONFIG = {
    'host': '127.0.0.1',
    'port': 3306,
    'user': 'root',
    'password': 'test_password',
    'database': 'test_vectordb'
}


@pytest.fixture
def mock_mariadb_connection():
    """Mock MariaDB connection for testing."""
    mock_connection = Mock()
    mock_cursor = Mock()
    
    # Configure cursor mock
    mock_cursor.fetchone.return_value = ('11.7.2-MariaDB-test',)
    mock_cursor.fetchall.return_value = [('test_table',), ('documents',)]
    mock_cursor.execute.return_value = None
    
    # Configure connection mock
    mock_connection.cursor.return_value = mock_cursor
    mock_connection.commit.return_value = None
    mock_connection.close.return_value = None
    
    return mock_connection, mock_cursor


@pytest.fixture
def mock_embedding_model():
    """Mock sentence transformer model for testing."""
    mock_model = Mock()
    mock_model.encode.return_value = [[0.1, 0.2, 0.3, 0.4] * 96]  # 384 dimensions
    mock_model.get_sentence_embedding_dimension.return_value = 384
    mock_model.max_seq_length = 256
    return mock_model


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        "MariaDB is an open-source relational database management system.",
        "Vector search enables semantic similarity matching for AI applications.",
        "RAG combines information retrieval with large language model generation.",
        "Jupyter notebooks provide an interactive computing environment for data science.",
        "Python is a versatile programming language with extensive AI libraries."
    ]


@pytest.fixture
def sample_embeddings():
    """Sample embeddings corresponding to sample documents."""
    return [
        [0.1, 0.2, 0.3, 0.4] * 96,  # 384 dimensions
        [0.2, 0.3, 0.4, 0.5] * 96,
        [0.3, 0.4, 0.5, 0.6] * 96,
        [0.4, 0.5, 0.6, 0.7] * 96,
        [0.5, 0.6, 0.7, 0.8] * 96,
    ]


@pytest.fixture
def temp_config_file() -> Generator[str, None, None]:
    """Create a temporary configuration file for testing."""
    config_content = """
[database]
host = 127.0.0.1
port = 3306
user = root
database = test_vectordb

[embedding]
model = all-MiniLM-L6-v2
batch_size = 32
chunk_size = 500

[search]
default_top_k = 5
default_distance = cosine
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write(config_content)
        temp_file = f.name
    
    yield temp_file
    
    # Cleanup
    if os.path.exists(temp_file):
        os.unlink(temp_file)


@pytest.fixture
def mock_groq_client():
    """Mock Groq client for RAG testing."""
    mock_client = Mock()
    mock_completion = Mock()
    mock_choice = Mock()
    mock_message = Mock()
    
    mock_message.content = "This is a test response from the LLM."
    mock_choice.message = mock_message
    mock_completion.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_completion
    
    return mock_client


@pytest.fixture
def ipython_shell():
    """Mock IPython shell for magic command testing."""
    from IPython.testing import globalipapp
    return globalipapp.get_ipython()


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    original_env = os.environ.copy()
    
    # Set test environment variables
    os.environ['GROQ_API_KEY'] = 'test_api_key'
    os.environ['MARIADB_HOST'] = TEST_CONFIG['host']
    os.environ['MARIADB_PORT'] = str(TEST_CONFIG['port'])
    os.environ['MARIADB_USER'] = TEST_CONFIG['user']
    os.environ['MARIADB_DATABASE'] = TEST_CONFIG['database']
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def magic_instance_connected(ipython_shell):
    """Create a MariaDBVectorMagics instance with mocked connected database."""
    from mariadb_vector_magics.magics.main import MariaDBVectorMagics
    
    magic = MariaDBVectorMagics(ipython_shell)
    
    # Mock the database manager methods
    magic.db_manager.connection = Mock()
    magic.db_manager.cursor = Mock()
    magic.db_manager.is_connected = Mock(return_value=True)
    magic.db_manager.get_version = Mock(return_value='11.7.2-MariaDB-test')
    magic.db_manager.test_vector_support = Mock(return_value=True)
    magic.db_manager.execute_command = Mock()
    magic.db_manager.execute_query = Mock(return_value=[])
    
    # Mock the vector store manager methods
    magic.vector_store_manager.create_vector_store = Mock(return_value={
        'table': 'test_table',
        'dimensions': 384,
        'distance_metric': 'cosine',
        'hnsw_m_value': 16,
        'schema': ['id', 'document_text', 'metadata', 'embedding']
    })
    magic.vector_store_manager.get_table_info = Mock(return_value={
        'name': 'test_table',
        'row_count': 10,
        'sample_data': [(1, "Sample text"), (2, "Another text")]
    })
    magic.vector_store_manager.list_vector_tables = Mock(return_value=['table1', 'table2'])
    
    return magic


@pytest.fixture
def magic_instance_full(ipython_shell, mock_groq_client):
    """Create a fully mocked MariaDBVectorMagics instance."""
    from mariadb_vector_magics.magics.main import MariaDBVectorMagics
    
    magic = MariaDBVectorMagics(ipython_shell)
    
    # Mock the database manager
    magic.db_manager.connection = Mock()
    magic.db_manager.cursor = Mock()
    magic.db_manager.is_connected = Mock(return_value=True)
    magic.db_manager.get_version = Mock(return_value='11.7.2-MariaDB-test')
    magic.db_manager.test_vector_support = Mock(return_value=True)
    
    # Mock the embedding manager
    magic.embedding_manager.model = Mock()
    magic.embedding_manager.is_loaded = Mock(return_value=True)
    magic.embedding_manager.load_model = Mock(return_value={
        'name': 'all-MiniLM-L6-v2',
        'embedding_dimension': 384,
        'load_time': 1.5
    })
    magic.embedding_manager.encode_single = Mock(return_value=[0.1, 0.2, 0.3] * 128)
    
    # Mock the search manager
    magic.search_manager.semantic_search = Mock(return_value=[
        (1, "Test document", '{"test": "metadata"}', 0.5)
    ])
    
    # Mock the RAG manager
    magic.rag_manager.query = Mock(return_value={
        'question': 'What is testing?',
        'answer': 'Testing is a process of verification.',
        'context': ['Test context document', 'Another context document'],
        'context_metadata': [
            {'id': 1, 'distance': 0.3, 'metadata': {}},
            {'id': 2, 'distance': 0.5, 'metadata': {}}
        ],
        'model': 'llama-3.3-70b-versatile',
        'temperature': 0.2,
        'context_count': 2,
        'total_time': 1.5
    })
    
    return magic


# Test markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_db: mark test as requiring database connection"
    )