"""
Main magic commands implementation using modular core components.

This module provides the IPython magic command interface that orchestrates
the core functionality modules.
"""

import logging
from typing import Optional, Dict, Any

from IPython.core.magic import (Magics, magics_class, line_magic, cell_magic)
from IPython.core.magic_arguments import (argument, magic_arguments, parse_argstring)
from IPython.display import display, HTML

from ..core import (
    DatabaseManager, EmbeddingManager, VectorStoreManager, 
    SearchManager, RAGManager, HTMLRenderer
)
from ..core.vector_store import VectorStoreConfig
from ..core.exceptions import (
    MariaDBVectorError, DatabaseConnectionError, EmbeddingError,
    VectorStoreError, SearchError, RAGError
)
from ..core.secrets import get_secret, set_secret

logger = logging.getLogger(__name__)


@magics_class
class MariaDBVectorMagics(Magics):
    """
    IPython magic commands for MariaDB Vector operations.
    
    This class provides a clean interface to the modular core functionality,
    handling user input validation, error display, and result formatting.
    """
    
    def __init__(self, shell):
        super(MariaDBVectorMagics, self).__init__(shell)
        
        # Initialize core managers
        self.db_manager = DatabaseManager()
        self.embedding_manager = EmbeddingManager()
        self.vector_store_manager = VectorStoreManager(self.db_manager)
        self.search_manager = SearchManager(self.db_manager, self.embedding_manager)
        self.rag_manager = RAGManager(self.db_manager, self.embedding_manager, self.search_manager)
        
        # HTML renderer for output
        self.renderer = HTMLRenderer()
        
        logger.info("MariaDB Vector Magics initialized with modular architecture")
    
    @line_magic
    @magic_arguments()
    @argument('--host', default='127.0.0.1', help='MariaDB host address')
    @argument('--port', type=int, default=3306, help='MariaDB port number')
    @argument('--user', default='root', help='MariaDB username')
    @argument('--password', required=True, help='MariaDB password')
    @argument('--database', default='vectordb', help='Database name')
    def connect_mariadb(self, line):
        """
        Connect to MariaDB server with Vector support.
        
        Usage: %connect_mariadb --password=yourpassword --database=vectordb
        """
        args = parse_argstring(self.connect_mariadb, line)
        
        try:
            success = self.db_manager.connect(
                host=args.host,
                port=args.port,
                user=args.user,
                password=args.password,
                database=args.database
            )
            
            if success:
                version = self.db_manager.get_version()
                vector_support = self.db_manager.test_vector_support()
                
                details = {
                    'Version': version,
                    'Database': args.database,
                    'Vector Support': 'Available' if vector_support else 'Not Available'
                }
                
                display(HTML(self.renderer.success("Connected to MariaDB", details)))
                return "Connected successfully"
            
        except DatabaseConnectionError as e:
            display(HTML(self.renderer.error("Connection failed", str(e))))
            return None
        except Exception as e:
            display(HTML(self.renderer.error("Unexpected error", str(e))))
            logger.error(f"Unexpected error in connect_mariadb: {e}")
            return None
    
    @line_magic
    def load_embedding_model(self, line):
        """
        Load a sentence transformer model for generating embeddings.
        
        Usage: %load_embedding_model all-MiniLM-L6-v2
        """
        model_name = line.strip() or "all-MiniLM-L6-v2"
        
        try:
            display(HTML(self.renderer.info(f"Loading embedding model: {model_name}")))
            
            model_info = self.embedding_manager.load_model(model_name)
            
            display(HTML(self.renderer.success("Model loaded successfully", model_info)))
            return f"Model {model_name} loaded"
            
        except EmbeddingError as e:
            display(HTML(self.renderer.error("Model loading failed", str(e))))
            return None
        except Exception as e:
            display(HTML(self.renderer.error("Unexpected error", str(e))))
            logger.error(f"Unexpected error in load_embedding_model: {e}")
            return None
    
    @line_magic
    @magic_arguments()
    @argument('--table', required=True, help='Name of the vector table to create')
    @argument('--dimensions', type=int, default=384, help='Vector embedding dimensions')
    @argument('--distance', default='cosine', choices=['cosine', 'euclidean'], help='Distance metric')
    @argument('--m_value', type=int, default=16, help='HNSW M parameter (3-200)')
    @argument('--drop_if_exists', action='store_true', help='Drop table if it exists')
    def create_vector_store(self, line):
        """
        Create a MariaDB table optimized for vector storage with HNSW indexing.
        
        Usage: %create_vector_store --table my_docs --dimensions 384 --distance cosine
        """
        args = parse_argstring(self.create_vector_store, line)
        
        try:
            config = VectorStoreConfig(
                table=args.table,
                dimensions=args.dimensions,
                distance=args.distance,
                m_value=args.m_value,
                drop_if_exists=args.drop_if_exists
            )
            
            result = self.vector_store_manager.create_vector_store(config)
            
            # Store distance type for search operations
            self.search_manager.set_table_distance_type(args.table, args.distance)
            
            display(HTML(self.renderer.success("Vector store created successfully", result)))
            return f"Table {args.table} created"
            
        except VectorStoreError as e:
            display(HTML(self.renderer.error("Vector store creation failed", str(e))))
            return None
        except Exception as e:
            display(HTML(self.renderer.error("Unexpected error", str(e))))
            logger.error(f"Unexpected error in create_vector_store: {e}")
            return None
    
    @line_magic
    def show_vector_tables(self, line):
        """
        Display all vector tables in the current database.
        
        Usage: %show_vector_tables
        """
        try:
            tables = self.vector_store_manager.list_vector_tables()
            
            if tables:
                table_html = '<br>'.join([f"• {table}" for table in tables])
                display(HTML(f"""
                <div style='padding: 10px; background-color: #d1ecf1; border: 1px solid #bee5eb; border-radius: 5px; color: #0c5460;'>
                    <b>Vector tables in database:</b><br><br>
                    {table_html}
                </div>
                """))
                return tables
            else:
                display(HTML(self.renderer.warning("No vector tables found in database")))
                return []
                
        except Exception as e:
            display(HTML(self.renderer.error("Failed to list tables", str(e))))
            logger.error(f"Error in show_vector_tables: {e}")
            return None
    
    @cell_magic
    @magic_arguments()
    @argument('--table', required=True, help='Target vector table name')
    @argument('--batch_size', type=int, default=32, help='Batch size for embedding generation')
    @argument('--chunk_size', type=int, default=500, help='Max characters per text chunk')
    @argument('--overlap', type=int, default=50, help='Character overlap between chunks')
    def embed_table(self, line, cell):
        """
        Generate embeddings and insert documents into vector table.
        
        Usage: 
        %%embed_table --table documents --batch_size 32
        ["text1", "text2", "text3"]
        """
        args = parse_argstring(self.embed_table, line)
        
        try:
            # Validate prerequisites
            if not self.db_manager.is_connected():
                display(HTML(self.renderer.error("Not connected to MariaDB", "Use %connect_mariadb first")))
                return None
            
            if not self.embedding_manager.is_loaded():
                display(HTML(self.renderer.error("No embedding model loaded", "Use %load_embedding_model first")))
                return None
            
            # Execute cell content to get documents
            documents = eval(cell.strip())
            
            if not isinstance(documents, list):
                raise ValueError("Cell must evaluate to a list of strings")
            
            display(HTML(self.renderer.info(
                f"Processing {len(documents)} documents",
                {
                    'Chunk size': f"{args.chunk_size} chars",
                    'Batch size': args.batch_size,
                    'Target table': args.table
                }
            )))
            
            # Process documents with embeddings
            stats, chunks, embeddings, metadata = self.embedding_manager.process_documents(
                documents, args.chunk_size, args.overlap, args.batch_size
            )
            
            # Insert into vector store
            inserted_count = self.vector_store_manager.insert_embeddings(
                args.table, chunks, embeddings, metadata
            )
            
            result_details = {
                'Documents processed': stats.documents_processed,
                'Chunks created': stats.chunks_created,
                'Records inserted': inserted_count,
                'Processing time': f"{stats.processing_time:.2f}s",
                'Table': args.table
            }
            
            display(HTML(self.renderer.success("Documents embedded and inserted successfully", result_details)))
            return f"Inserted {inserted_count} records into {args.table}"
            
        except (VectorStoreError, EmbeddingError) as e:
            display(HTML(self.renderer.error("Document processing failed", str(e))))
            return None
        except SyntaxError as e:
            display(HTML(self.renderer.error(
                "Syntax error in cell content",
                "Cell must contain valid Python code that evaluates to a list of strings"
            )))
            return None
        except Exception as e:
            display(HTML(self.renderer.error("Unexpected error", str(e))))
            logger.error(f"Unexpected error in embed_table: {e}")
            return None
    
    @line_magic
    def query_table(self, line):
        """
        Quick query to inspect table contents.
        
        Usage: %query_table documents
        """
        table_name = line.strip()
        
        if not table_name:
            display(HTML(self.renderer.error("Table name is required")))
            return None
        
        try:
            info = self.vector_store_manager.get_table_info(table_name)
            
            # Format sample data
            sample_rows = []
            for row in info.get('sample_data', []):
                sample_rows.append([str(row[0]), row[1][:100] + "..." if len(row[1]) > 100 else row[1]])
            
            if sample_rows:
                table_html = self.renderer.table(
                    headers=['ID', 'Text Preview'],
                    rows=sample_rows,
                    title=f"Table: {table_name} ({info.get('row_count', 0)} records)"
                )
                display(HTML(table_html))
            else:
                display(HTML(self.renderer.info(f"Table '{table_name}' has {info.get('row_count', 0)} records but no sample data")))
            
            return info
            
        except Exception as e:
            display(HTML(self.renderer.error("Table query failed", str(e))))
            logger.error(f"Error in query_table: {e}")
            return None
    
    @line_magic
    @magic_arguments()
    @argument('query', type=str, help='Search query text')
    @argument('--table', required=True, help='Table to search')
    @argument('--top_k', type=int, default=5, help='Number of results to return')
    @argument('--threshold', type=float, default=None, help='Distance threshold filter')
    @argument('--show_distance', action='store_true', help='Show distance scores')
    def semantic_search(self, line):
        """
        Perform semantic search on vector table.
        
        Usage: %semantic_search "machine learning" --table documents --top_k 3 --show_distance
        """
        args = parse_argstring(self.semantic_search, line)
        
        try:
            # Validate prerequisites
            if not self.db_manager.is_connected():
                display(HTML(self.renderer.error("Not connected to MariaDB")))
                return None
            
            if not self.embedding_manager.is_loaded():
                display(HTML(self.renderer.error("No embedding model loaded")))
                return None
            
            # Perform search
            results = self.search_manager.semantic_search(
                query=args.query.strip('"').strip("'"),
                table=args.table,
                top_k=args.top_k,
                distance_threshold=args.threshold,
                include_distance=args.show_distance
            )
            
            if results:
                # Display formatted results
                results_html = self.renderer.search_results(
                    results, args.show_distance, args.query
                )
                display(HTML(results_html))
            else:
                display(HTML(self.renderer.warning("No results found")))
            
            return results
            
        except SearchError as e:
            display(HTML(self.renderer.error("Search failed", str(e))))
            return None
        except Exception as e:
            display(HTML(self.renderer.error("Unexpected error", str(e))))
            logger.error(f"Unexpected error in semantic_search: {e}")
            return None
    
    @line_magic
    @magic_arguments()
    @argument('question', type=str, help='Question to answer')
    @argument('--table', required=True, help='Table to search for context')
    @argument('--top_k', type=int, default=3, help='Number of context documents')
    @argument('--api_key', default=None, help='LLM API key')
    @argument('--model', default='llama-3.3-70b-versatile', help='LLM model to use')
    @argument('--temperature', type=float, default=0.2, help='LLM temperature (0-1)')
    def rag_query(self, line):
        """
        Complete RAG pipeline: retrieval + augmented generation.
        
        Usage: %rag_query "What is MariaDB?" --table documents --api_key YOUR_KEY
        """
        args = parse_argstring(self.rag_query, line)
        
        try:
            # Validate prerequisites
            if not self.db_manager.is_connected():
                display(HTML(self.renderer.error("Not connected to MariaDB")))
                return None
            
            if not self.embedding_manager.is_loaded():
                display(HTML(self.renderer.error("No embedding model loaded")))
                return None
            
            question = args.question.strip('"').strip("'")
            
            display(HTML(self.renderer.info(
                f"RAG Query: {question}",
                {'Step': '1/3 - Retrieving relevant context'}
            )))
            
            # Perform RAG query
            result = self.rag_manager.query(
                question=question,
                table=args.table,
                top_k=args.top_k,
                api_key=args.api_key,
                model=args.model,
                temperature=args.temperature
            )
            
            # Display formatted result
            self._display_rag_result(result)
            
            return result
            
        except RAGError as e:
            display(HTML(self.renderer.error("RAG query failed", str(e))))
            return None
        except Exception as e:
            display(HTML(self.renderer.error("Unexpected error", str(e))))
            logger.error(f"Unexpected error in rag_query: {e}")
            return None
    
    def _display_rag_result(self, result: Dict[str, Any]):
        """Display formatted RAG result."""
        # Build context HTML
        context_html = ""
        for i, (context_text, metadata) in enumerate(zip(result['context'], result['context_metadata']), 1):
            distance_info = f" (distance: {metadata['distance']:.4f})" if metadata['distance'] is not None else ""
            context_html += f"""
            <div style='padding: 8px; margin: 5px 0; background-color: #f8f9fa; 
            border-left: 3px solid #6c757d; font-size: 0.9em;'>
                <b>Context {i}</b>{distance_info}<br>
                {context_text[:300]}{'...' if len(context_text) > 300 else ''}
            </div>
            """
        
        # Display complete result
        display(HTML(f"""
        <div style='padding: 15px; background-color: white; border: 1px solid #dee2e6; border-radius: 5px; margin-top: 10px;'>
            <div style='padding: 12px; background-color: #d4edda; border: 1px solid #c3e6cb; 
            border-radius: 5px; color: #155724; margin-bottom: 15px;'>
                <b>RAG Answer Complete</b>
            </div>
            
            <div style='margin-bottom: 20px;'>
                <h4 style='color: #007bff; margin-bottom: 10px;'>Answer:</h4>
                <div style='padding: 15px; background-color: #e7f3ff; border-left: 4px solid #007bff; border-radius: 4px;'>
                    {result['answer']}
                </div>
            </div>
            
            <div>
                <h4 style='color: #6c757d; margin-bottom: 10px; font-size: 1em;'>Retrieved Context:</h4>
                {context_html}
            </div>
            
            <div style='margin-top: 15px; padding: 10px; background-color: #f8f9fa; border-radius: 4px; font-size: 0.85em; color: #6c757d;'>
                <b>Model:</b> {result['model']} | 
                <b>Temperature:</b> {result['temperature']} | 
                <b>Context docs:</b> {result['context_count']} |
                <b>Total time:</b> {result['total_time']:.2f}s
            </div>
        </div>
        """))
    
    @line_magic
    @magic_arguments()
    @argument('name', type=str, help='Secret name')
    @argument('value', type=str, help='Secret value')
    def set_secret(self, line):
        """
        Set a secret value in encrypted storage.
        
        Usage: %set_secret GROQ_API_KEY your_api_key_here
        """
        args = parse_argstring(self.set_secret, line)
        
        try:
            set_secret(args.name, args.value)
            display(HTML(self.renderer.success(f"Secret '{args.name}' has been set securely")))
            return True
            
        except Exception as e:
            display(HTML(self.renderer.error("Failed to set secret", str(e))))
            return False
    
    @line_magic
    def get_secret(self, line):
        """
        Get a secret value (for testing - use with caution).
        
        Usage: %get_secret GROQ_API_KEY
        """
        secret_name = line.strip()
        
        if not secret_name:
            display(HTML(self.renderer.error("Secret name is required")))
            return None
        
        try:
            value = get_secret(secret_name)
            if value:
                # Don't display the actual value for security
                display(HTML(self.renderer.success(f"Secret '{secret_name}' found (value hidden for security)")))
                return "***HIDDEN***"
            else:
                display(HTML(self.renderer.warning(f"Secret '{secret_name}' not found")))
                return None
                
        except Exception as e:
            display(HTML(self.renderer.error("Failed to get secret", str(e))))
            return None


def load_ipython_extension(ipython):
    """Load the extension in IPython."""
    ipython.register_magics(MariaDBVectorMagics)
    logger.info("MariaDB Vector Magics extension loaded with modular architecture")


def unload_ipython_extension(ipython):
    """Unload the extension from IPython."""
    logger.info("MariaDB Vector Magics extension unloaded")