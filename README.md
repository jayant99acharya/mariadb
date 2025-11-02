# MariaDB Vector Magics

**Professional IPython Magic Commands for MariaDB Vector Operations**

[![MariaDB](https://img.shields.io/badge/MariaDB-11.7%2B-blue)](https://mariadb.org/projects/mariadb-vector/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## Overview

MariaDB Vector Magics is a professional-grade Python package that provides seamless integration between Jupyter notebooks and MariaDB Vector databases. It enables data scientists and developers to perform vector operations, semantic search, and Retrieval Augmented Generation (RAG) workflows through intuitive IPython magic commands.

### Problem Statement

Working with MariaDB Vector databases traditionally requires:
- Complex SQL queries for vector operations
- Manual embedding generation and management
- Extensive Python connector code for database operations
- Custom implementations for semantic search and RAG pipelines

### Solution

This package abstracts the complexity while providing full access to MariaDB Vector capabilities through:
- Simple magic commands for database operations
- Automatic embedding generation using state-of-the-art models
- Built-in semantic search with distance metrics
- Complete RAG pipeline integration with LLM providers

## Features

### Core Functionality
- **Database Connection Management**: Secure connection handling with connection pooling support
- **Embedding Generation**: Automatic text-to-vector conversion using sentence-transformers
- **Vector Store Operations**: Optimized table creation with HNSW indexing
- **Semantic Search**: Natural language queries with cosine and euclidean distance metrics
- **RAG Pipeline**: Complete retrieval-augmented generation with LLM integration
- **Batch Processing**: Efficient handling of large document collections
- **Progress Tracking**: Real-time progress indicators for long-running operations

### Technical Capabilities
- Support for multiple embedding models
- Configurable chunking strategies for large documents
- Automatic metadata management
- Distance threshold filtering
- Batch size optimization
- Error handling and logging
- Type hints and comprehensive documentation

## Installation

### Prerequisites
- Python 3.8 or higher
- MariaDB 11.7+ with Vector support
- Jupyter Notebook or JupyterLab

### Install from PyPI
```bash
pip install mariadb-vector-magics
```

### Install from Source
```bash
git clone https://github.com/jayant99acharya/mariadb.git
cd mariadb
pip install -e .
```

### Development Installation
```bash
git clone https://github.com/jayant99acharya/mariadb.git
cd mariadb
pip install -e ".[dev]"
```

## Quick Start

### 1. Load the Extension
```python
%load_ext mariadb_vector_magics
```

### 2. Connect to MariaDB
```python
%connect_mariadb --password=yourpassword --database=vectordb
```

### 3. Load Embedding Model
```python
%load_embedding_model all-MiniLM-L6-v2
```

### 4. Create Vector Store
```python
%create_vector_store --table documents --dimensions 384 --distance cosine
```

### 5. Add Documents
```python
%%embed_table --table documents --batch_size 32
[
    "MariaDB is an open-source relational database management system.",
    "Vector search enables semantic similarity matching for AI applications.",
    "RAG combines information retrieval with large language model generation."
]
```

### 6. Semantic Search
```python
%semantic_search "database technology" --table documents --top_k 3 --show_distance
```

### 7. RAG Query
```python
%rag_query "What is vector search?" --table documents --model llama-3.3-70b-versatile
```

## API Reference

### Connection Commands

#### `%connect_mariadb`
Establish connection to MariaDB server with Vector support.

**Parameters:**
- `--host`: MariaDB host address (default: 127.0.0.1)
- `--port`: MariaDB port number (default: 3306)
- `--user`: MariaDB username (default: root)
- `--password`: MariaDB password (required)
- `--database`: Database name (default: vectordb)

**Example:**
```python
%connect_mariadb --host=localhost --password=mypass --database=vectordb
```

#### `%load_embedding_model`
Load sentence transformer model for embedding generation.

**Parameters:**
- `model_name`: HuggingFace model name (default: all-MiniLM-L6-v2)

**Example:**
```python
%load_embedding_model sentence-transformers/all-mpnet-base-v2
```

### Vector Store Commands

#### `%create_vector_store`
Create optimized vector table with HNSW indexing.

**Parameters:**
- `--table`: Table name (required)
- `--dimensions`: Vector dimensions (default: 384)
- `--distance`: Distance metric - cosine|euclidean (default: cosine)
- `--m_value`: HNSW M parameter 3-200 (default: 16)
- `--drop_if_exists`: Drop existing table

**Example:**
```python
%create_vector_store --table docs --dimensions 768 --distance euclidean --m_value 32
```

#### `%%embed_table`
Generate embeddings and insert documents (cell magic).

**Parameters:**
- `--table`: Target table name (required)
- `--batch_size`: Embedding batch size (default: 32)
- `--chunk_size`: Max characters per chunk (default: 500)
- `--overlap`: Character overlap between chunks (default: 50)

**Example:**
```python
%%embed_table --table docs --batch_size 64 --chunk_size 1000
import pandas as pd
df = pd.read_csv('documents.csv')
df['content'].tolist()
```

### Search Commands

#### `%semantic_search`
Perform semantic similarity search.

**Parameters:**
- `query`: Search query text (required)
- `--table`: Table to search (required)
- `--top_k`: Number of results (default: 5)
- `--threshold`: Distance threshold filter
- `--show_distance`: Display distance scores

**Example:**
```python
%semantic_search "artificial intelligence" --table docs --top_k 10 --threshold 0.8 --show_distance
```

#### `%rag_query`
Complete RAG pipeline with LLM integration.

**Parameters:**
- `question`: Question to answer (required)
- `--table`: Context table (required)
- `--top_k`: Context documents (default: 3)
- `--api_key`: Groq API key
- `--model`: LLM model (default: llama-3.3-70b-versatile)
- `--temperature`: LLM temperature 0-1 (default: 0.2)

**Example:**
```python
%rag_query "Explain vector databases" --table docs --top_k 5 --model llama-3.1-70b-versatile --temperature 0.1
```

### Utility Commands

#### `%show_vector_tables`
List all tables in current database.

#### `%query_table`
Quick table inspection with sample data.

**Example:**
```python
%query_table documents
```

## Configuration

### Environment Variables
```bash
export GROQ_API_KEY="your_groq_api_key_here"
export MARIADB_HOST="127.0.0.1"
export MARIADB_PORT="3306"
export MARIADB_USER="root"
export MARIADB_DATABASE="vectordb"
```

### Connection Testing
Test your MariaDB connection and Vector support:
```bash
mariadb-vector-test --password=yourpass --database=vectordb --verbose
```

## Architecture

```
┌─────────────────────────────────────────┐
│ Jupyter Notebook Environment           │
├─────────────────────────────────────────┤
│ MariaDB Vector Magic Commands           │
│ ├── Connection Management               │
│ ├── Embedding Generation                │
│ ├── Vector Operations                   │
│ └── Query Interface                     │
├─────────────────────────────────────────┤
│ sentence-transformers                   │
│ (Local Embedding Models)                │
├─────────────────────────────────────────┤
│ MariaDB 11.7+ with Vector Support       │
│ ├── VECTOR data type                    │
│ ├── HNSW indexing                       │
│ ├── Distance functions                  │
│ └── JSON metadata support               │
├─────────────────────────────────────────┤
│ External LLM APIs (Optional)            │
│ └── Groq, OpenAI, etc.                 │
└─────────────────────────────────────────┘
```

## Use Cases

### Academic Research
- Semantic search across research papers
- Literature review automation
- Citation analysis and clustering

### Enterprise Applications
- Corporate knowledge base search
- Document similarity analysis
- Automated FAQ systems

### E-commerce
- Product recommendation systems
- Semantic product search
- Customer review analysis

### Content Management
- News article clustering
- Content recommendation
- Duplicate content detection

## Performance Considerations

### Optimization Guidelines
- Use appropriate batch sizes for your hardware (16-64 typical)
- Choose optimal HNSW M values (16-32 for most use cases)
- Implement proper chunking for large documents
- Monitor memory usage during embedding generation

### Benchmarks
- Embedding generation: ~1000 docs/minute (all-MiniLM-L6-v2)
- Vector search: Sub-millisecond for 100K vectors
- RAG pipeline: 2-5 seconds end-to-end

## Development

### Setting Up Development Environment
```bash
git clone https://github.com/jayant99acharya/mariadb.git
cd mariadb
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

### Running Tests
```bash
pytest tests/ -v --cov=mariadb_vector_magics
```

### Code Quality
```bash
black mariadb_vector_magics/
flake8 mariadb_vector_magics/
mypy mariadb_vector_magics/
```

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Workflow
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## Troubleshooting

### Common Issues

**Connection Errors**
- Verify MariaDB server is running
- Check firewall settings
- Confirm Vector support is enabled

**Embedding Errors**
- Ensure sufficient memory for model loading
- Check internet connectivity for model downloads
- Verify model name is correct

**Performance Issues**
- Reduce batch size if running out of memory
- Optimize chunk size for your documents
- Consider using smaller embedding models

### Getting Help
- Check the [Issues](https://github.com/jayant99acharya/mariadb/issues) page
- Review the [Documentation](https://github.com/jayant99acharya/mariadb/blob/main/README.md)
- Contact the maintainer: jayant99acharya@gmail.com, zakeer1408@gmail.com

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- MariaDB Foundation for Vector implementation
- Hugging Face for sentence-transformers
- Groq for fast LLM inference
- The open-source community for continuous support

## Citation

If you use this package in your research, please cite:

```bibtex
@software{mariadb_vector_magics,
  author = {Acharya, Jayant},{Shaik Mohammed, Zakeer},
  title = {MariaDB Vector Magics: IPython Magic Commands for Vector Operations},
  url = {https://github.com/jayant99acharya/mariadb},
  version = {0.1.0},
  year = {2025}
}
```

---

**Developed with precision for the MariaDB community**
