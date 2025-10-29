# 🚀 MariaDB Vector Intelligence Hub

**AI-Powered Magic Commands for Jupyter Data Science**

[![MariaDB](https://img.shields.io/badge/MariaDB-11.7%2B-blue)](https://mariadb.org/projects/mariadb-vector/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> Winner Submission for MariaDB Bangalore Hackathon 2025 - Integration Track

## 🎯 Overview

MariaDB Vector Intelligence Hub brings the power of vector databases and RAG (Retrieval Augmented Generation) to Jupyter notebooks through intuitive magic commands. Data scientists can now build semantic search and AI applications without writing complex SQL or connector code.

**Problem Solved**: Accessing MariaDB Vector features requires writing complex SQL queries and Python connector code, creating friction for rapid prototyping of AI applications.

**Solution**: Jupyter magic commands that abstract complexity while providing full power of MariaDB Vector, embeddings, and RAG pipelines.

## ✨ Features

- 🔌 **One-line MariaDB connection** with beautiful status feedback
- 🤖 **Auto-embedding generation** using sentence-transformers
- 📊 **Vector store creation** with optimized HNSW indexing
- 🔍 **Semantic search** with natural language queries
- 🧠 **Complete RAG pipeline** with LLM integration (Groq)
- 📈 **Progress tracking** for batch operations
- 🎨 **Rich HTML output** with formatted results

## 🚀 Quick Start

### Installation

Clone the repository
git clone https://github.com/jayant99acharya/mariadb.git
cd mariadb-vector-magic

Install dependencies
pip install -e .

Start MariaDB with Vector support
docker run -d
--name mariadb-vector
-e MYSQL_ROOT_PASSWORD=yourpassword
-e MYSQL_DATABASE=vectordb
-p 3306:3306
mariadb:11.7


### Basic Usage

Load the magic commands
%load_ext mariadb_vector_magics.magics

Connect to MariaDB
%connect_mariadb --password=yourpassword --database=vectordb

Load embedding model
%load_embedding_model all-MiniLM-L6-v2

Create a vector store
%create_vector_store --table documents --dimensions 384

Add documents with automatic embeddings
%%embed_table --table documents
[
"MariaDB is an open-source database.",
"Vector search enables semantic similarity.",
"RAG combines retrieval with generation."
]

Semantic search
%semantic_search "database technology" --table documents --top_k 3

Complete RAG query with LLM
%rag_query "What is vector search?" --table documents --model llama-3.3-70b-versatile



## 📚 Magic Commands Reference

### Connection & Setup

**`%connect_mariadb`** - Connect to MariaDB server

%connect_mariadb --host=127.0.0.1 --password=xxx --database=vectordb


**`%load_embedding_model`** - Load sentence-transformer model

%load_embedding_model all-MiniLM-L6-v2


### Vector Store Operations

**`%create_vector_store`** - Create optimized vector table

%create_vector_store --table docs --dimensions 384 --distance cosine --m_value 16


**`%%embed_table`** - Generate embeddings and insert documents (cell magic)

%%embed_table --table docs --batch_size 32 --chunk_size 500
["document1", "document2", "document3"]


### Search & RAG

**`%semantic_search`** - Semantic similarity search

%semantic_search "query" --table docs --top_k 5 --show_distance


**`%rag_query`** - Complete RAG pipeline


%rag_query "question" --table docs --model llama-3.3-70b-versatile --api_key xxx


### Utilities

**`%show_vector_tables`** - List all tables in database
**`%query_table`** - Quick table inspection

## 🏗️ Architecture

┌─────────────────────────────────────────┐
│ Jupyter Notebook │
├─────────────────────────────────────────┤
│ MariaDB Vector Magic Commands │
│ ├── Connection Management │
│ ├── Embedding Generation │
│ └── Query Interface │
├─────────────────────────────────────────┤
│ sentence-transformers │
│ (Local Embedding Models) │
├─────────────────────────────────────────┤
│ MariaDB 11.7+ with Vector Support │
│ ├── VECTOR data type │
│ ├── HNSW indexing │
│ └── Distance functions │
├─────────────────────────────────────────┤
│ Groq API (LLM for RAG) │
│ └── Llama 3.3 70B │
└─────────────────────────────────────────┘


## 📊 Demo & Examples

See our comprehensive demo notebook: [`demo_notebook.ipynb`](demo_notebook.ipynb)

**Example Use Cases:**
- 📖 Academic research papers semantic search
- 💼 Corporate documentation Q&A systems
- 🛍️ E-commerce product similarity
- 📰 News article clustering and retrieval
- 🎓 Educational content recommendations

## 🔧 Technical Details

**Dependencies:**
- MariaDB 11.7+ (Vector support)
- Python 3.8+
- IPython/Jupyter
- sentence-transformers
- mariadb-connector
- groq (optional, for RAG)

**Performance:**
- Batch embedding generation with progress tracking
- Optimized HNSW indexing for fast similarity search
- Automatic chunking for large documents
- Connection pooling support

## 🎓 Post-Hackathon Roadmap

- [ ] Submit as PR to official MariaDB Jupyter kernel
- [ ] Add LlamaIndex and LangChain integrations
- [ ] Multi-modal support (images, audio)
- [ ] Hybrid search (vector + full-text)
- [ ] Fine-tuned model support
- [ ] Enterprise security features
- [ ] Performance benchmarking suite
- [ ] Video tutorial series

## 🤝 Contributing

Contributions welcome! This project aims to become part of the official MariaDB ecosystem.

## 📝 License

MIT License - see [LICENSE](LICENSE) file

## 🙏 Acknowledgments

- MariaDB Foundation for Vector implementation
- Hugging Face for sentence-transformers
- Groq for fast LLM inference
- MariaDB Bangalore Hackathon organizers

## 📧 Contact

**Author**: Jayant Acharya  
**Email**: jayant99acharya@gmail.com  
**Hackathon**: MariaDB  Hackathon 2025  
**Track**: Integration

---

**Built with ❤️ for the MariaDB community**
