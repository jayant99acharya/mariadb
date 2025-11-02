"""
MariaDB Vector Magic Commands for Jupyter Notebooks

This module provides IPython magic commands for seamless integration with MariaDB Vector
databases, enabling vector operations, semantic search, and RAG (Retrieval Augmented Generation)
workflows directly from Jupyter notebooks.

Author: Jayant Acharya
License: MIT
"""

import logging
import json
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from IPython.core.magic import (Magics, magics_class, line_magic, cell_magic)
from IPython.core.magic_arguments import (argument, magic_arguments, parse_argstring)
from IPython.display import display, HTML, clear_output
from tqdm.auto import tqdm

try:
    import mariadb
except ImportError:
    raise ImportError("mariadb package is required. Install with: pip install mariadb")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("sentence-transformers package is required. Install with: pip install sentence-transformers")


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ConnectionConfig:
    """Configuration for MariaDB connection."""
    host: str = '127.0.0.1'
    port: int = 3306
    user: str = 'root'
    password: str = ''
    database: str = 'vectordb'
    local_infile: bool = True


@dataclass
class VectorStoreConfig:
    """Configuration for vector store creation."""
    table: str
    dimensions: int = 384
    distance: str = 'cosine'
    m_value: int = 16
    drop_if_exists: bool = False


class MariaDBVectorError(Exception):
    """Custom exception for MariaDB Vector operations."""
    pass


class HTMLRenderer:
    """Utility class for rendering HTML output in Jupyter notebooks."""
    
    @staticmethod
    def success(message: str, details: Optional[Dict[str, Any]] = None) -> str:
        """Render success message."""
        details_html = ""
        if details:
            details_html = "<br>".join([f"<b>{k}:</b> {v}" for k, v in details.items()])
            details_html = f"<br>{details_html}"
        
        return f"""
        <div style='padding: 10px; background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 5px; color: #155724;'>
            <b>SUCCESS:</b> {message}{details_html}
        </div>
        """
    
    @staticmethod
    def error(message: str, error_details: Optional[str] = None) -> str:
        """Render error message."""
        details_html = f"<br><b>Details:</b> {error_details}" if error_details else ""
        return f"""
        <div style='padding: 10px; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px; color: #721c24;'>
            <b>ERROR:</b> {message}{details_html}
        </div>
        """
    
    @staticmethod
    def info(message: str, details: Optional[Dict[str, Any]] = None) -> str:
        """Render info message."""
        details_html = ""
        if details:
            details_html = "<br>".join([f"<b>{k}:</b> {v}" for k, v in details.items()])
            details_html = f"<br>{details_html}"
        
        return f"""
        <div style='padding: 10px; background-color: #d1ecf1; border: 1px solid #bee5eb; border-radius: 5px; color: #0c5460;'>
            <b>INFO:</b> {message}{details_html}
        </div>
        """
    
    @staticmethod
    def warning(message: str) -> str:
        """Render warning message."""
        return f"""
        <div style='padding: 10px; background-color: #fff3cd; border: 1px solid #ffeeba; border-radius: 5px; color: #856404;'>
            <b>WARNING:</b> {message}
        </div>
        """


@magics_class
class MariaDBVectorMagics(Magics):
    """
    Custom magic commands for MariaDB Vector operations.
    
    This class provides a comprehensive set of magic commands for working with
    MariaDB Vector databases in Jupyter notebooks, including connection management,
    vector store operations, semantic search, and RAG workflows.
    """
    
    def __init__(self, shell):
        super(MariaDBVectorMagics, self).__init__(shell)
        self.connection: Optional[mariadb.Connection] = None
        self.cursor: Optional[mariadb.Cursor] = None
        self.embedding_model: Optional[SentenceTransformer] = None
        self.table_distance_types: Dict[str, str] = {}
        self.renderer = HTMLRenderer()
        
        logger.info("MariaDB Vector Magics initialized")
    
    def _validate_connection(self) -> bool:
        """Validate that database connection exists."""
        if not self.connection:
            display(HTML(self.renderer.error("Not connected to MariaDB. Use %connect_mariadb first.")))
            return False
        return True
    
    def _validate_embedding_model(self) -> bool:
        """Validate that embedding model is loaded."""
        if not self.embedding_model:
            display(HTML(self.renderer.error("No embedding model loaded. Use %load_embedding_model first.")))
            return False
        return True
    
    def _validate_table_exists(self, table_name: str) -> bool:
        """Validate that table exists in database."""
        try:
            self.cursor.execute("SHOW TABLES LIKE %s", (table_name,))
            result = self.cursor.fetchone()
            if not result:
                display(HTML(self.renderer.error(f"Table '{table_name}' does not exist.")))
                return False
            return True
        except mariadb.Error as e:
            display(HTML(self.renderer.error(f"Error checking table existence: {e}")))
            return False
    
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
        
        Args:
            --host: MariaDB host address (default: 127.0.0.1)
            --port: MariaDB port number (default: 3306)
            --user: MariaDB username (default: root)
            --password: MariaDB password (required)
            --database: Database name (default: vectordb)
        
        Returns:
            str: Success message or None on failure
        """
        args = parse_argstring(self.connect_mariadb, line)
        
        try:
            config = ConnectionConfig(
                host=args.host,
                port=args.port,
                user=args.user,
                password=args.password,
                database=args.database
            )
            
            self.connection = mariadb.connect(
                host=config.host,
                port=config.port,
                user=config.user,
                password=config.password,
                database=config.database,
                local_infile=config.local_infile
            )
            self.cursor = self.connection.cursor()
            
            # Verify Vector support
            self.cursor.execute("SELECT VERSION()")
            version = self.cursor.fetchone()[0]
            
            # Test Vector functionality
            try:
                self.cursor.execute("SELECT VEC_FromText('[1,2,3]')")
                vector_support = True
            except mariadb.Error:
                vector_support = False
            
            details = {
                'Version': version,
                'Database': config.database,
                'Vector Support': 'Available' if vector_support else 'Not Available'
            }
            
            display(HTML(self.renderer.success("Connected to MariaDB", details)))
            logger.info(f"Connected to MariaDB {version} at {config.host}:{config.port}")
            
            return "Connected successfully"
            
        except mariadb.Error as e:
            error_msg = f"Connection failed: {str(e)}"
            display(HTML(self.renderer.error("MariaDB connection failed", str(e))))
            logger.error(error_msg)
            return None
        except Exception as e:
            error_msg = f"Unexpected error during connection: {str(e)}"
            display(HTML(self.renderer.error("Unexpected connection error", str(e))))
            logger.error(error_msg)
            return None
    
    @line_magic
    def load_embedding_model(self, line):
        """
        Load a sentence transformer model for generating embeddings.
        
        Usage: %load_embedding_model all-MiniLM-L6-v2
        
        Args:
            line: Model name (default: all-MiniLM-L6-v2)
        
        Returns:
            str: Success message or None on failure
        """
        model_name = line.strip() or "all-MiniLM-L6-v2"
        
        try:
            display(HTML(self.renderer.info(f"Loading embedding model: {model_name}")))
            
            self.embedding_model = SentenceTransformer(model_name)
            self.embedding_model.max_seq_length = 256
            
            details = {
                'Model': model_name,
                'Max sequence length': 256,
                'Embedding dimension': self.embedding_model.get_sentence_embedding_dimension()
            }
            
            display(HTML(self.renderer.success("Model loaded successfully", details)))
            logger.info(f"Loaded embedding model: {model_name}")
            
            return f"Model {model_name} loaded"
            
        except Exception as e:
            error_msg = f"Failed to load model {model_name}: {str(e)}"
            display(HTML(self.renderer.error("Model loading failed", str(e))))
            logger.error(error_msg)
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
        
        Args:
            --table: Name of the vector table to create (required)
            --dimensions: Vector embedding dimensions (default: 384)
            --distance: Distance metric - cosine or euclidean (default: cosine)
            --m_value: HNSW M parameter, range 3-200 (default: 16)
            --drop_if_exists: Drop table if it already exists
        
        Returns:
            str: Success message or None on failure
        """
        args = parse_argstring(self.create_vector_store, line)
        
        if not self._validate_connection():
            return None
        
        try:
            config = VectorStoreConfig(
                table=args.table,
                dimensions=args.dimensions,
                distance=args.distance,
                m_value=args.m_value,
                drop_if_exists=args.drop_if_exists
            )
            
            # Validate M value
            if not (3 <= config.m_value <= 200):
                raise ValueError("M value must be between 3 and 200")
            
            # Drop table if requested
            if config.drop_if_exists:
                self.cursor.execute(f"DROP TABLE IF EXISTS {config.table}")
                display(HTML(self.renderer.warning(f"Dropped existing table: {config.table}")))
                logger.info(f"Dropped existing table: {config.table}")
            
            # Create the vector table with proper schema
            create_table_sql = f"""
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
            
            self.cursor.execute(create_table_sql)
            self.connection.commit()
            
            # Store distance type for later use
            self.table_distance_types[config.table] = config.distance
            
            details = {
                'Table': config.table,
                'Dimensions': config.dimensions,
                'Distance Metric': config.distance,
                'HNSW M Value': config.m_value,
                'Schema': 'id, document_text, metadata, embedding, created_at, updated_at'
            }
            
            display(HTML(self.renderer.success("Vector store created successfully", details)))
            logger.info(f"Created vector store: {config.table}")
            
            return f"Table {config.table} created"
            
        except mariadb.Error as e:
            error_msg = f"Database error creating table: {str(e)}"
            display(HTML(self.renderer.error("Database error", str(e))))
            logger.error(error_msg)
            return None
        except ValueError as e:
            display(HTML(self.renderer.error("Configuration error", str(e))))
            logger.error(f"Configuration error: {str(e)}")
            return None
        except Exception as e:
            error_msg = f"Unexpected error creating table: {str(e)}"
            display(HTML(self.renderer.error("Unexpected error", str(e))))
            logger.error(error_msg)
            return None
    
    @line_magic
    def show_vector_tables(self, line):
        """
        Display all tables in the current database.
        
        Usage: %show_vector_tables
        
        Returns:
            List of table names or None on failure
        """
        if not self._validate_connection():
            return None
        
        try:
            self.cursor.execute("SHOW TABLES")
            tables = self.cursor.fetchall()
            
            if tables:
                table_list = [t[0] for t in tables]
                table_html = '<br>'.join([f"• {table}" for table in table_list])
                
                display(HTML(f"""
                <div style='padding: 10px; background-color: #d1ecf1; border: 1px solid #bee5eb; border-radius: 5px; color: #0c5460;'>
                    <b>Tables in database:</b><br><br>
                    {table_html}
                </div>
                """))
                
                logger.info(f"Found {len(table_list)} tables in database")
                return table_list
            else:
                display(HTML(self.renderer.warning("No tables found in database")))
                return []
                
        except mariadb.Error as e:
            display(HTML(self.renderer.error("Database error", str(e))))
            logger.error(f"Error listing tables: {str(e)}")
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
        
        Args:
            --table: Target vector table name (required)
            --batch_size: Batch size for embedding generation (default: 32)
            --chunk_size: Max characters per text chunk (default: 500)
            --overlap: Character overlap between chunks (default: 50)
        
        Returns:
            str: Success message or None on failure
        """
        args = parse_argstring(self.embed_table, line)
        
        if not self._validate_connection() or not self._validate_embedding_model():
            return None
        
        if not self._validate_table_exists(args.table):
            return None
        
        try:
            # Execute the cell content to get documents list
            documents = eval(cell.strip())
            
            if not isinstance(documents, list):
                raise ValueError("Cell must evaluate to a list of strings")
            
            if len(documents) == 0:
                raise ValueError("Document list is empty")
            
            display(HTML(self.renderer.info(
                f"Processing {len(documents)} documents",
                {'Chunk size': f"{args.chunk_size} chars", 'Batch size': args.batch_size}
            )))
            
            # Process documents into chunks
            chunks, chunk_metadata = self._process_documents(documents, args.chunk_size, args.overlap)
            
            display(HTML(self.renderer.info(f"Created {len(chunks)} chunks from {len(documents)} documents")))
            
            # Generate embeddings in batches
            all_embeddings = self._generate_embeddings_batch(chunks, args.batch_size)
            
            # Insert into database
            inserted_count = self._insert_embeddings(args.table, chunks, all_embeddings, chunk_metadata)
            
            details = {
                'Documents processed': len(documents),
                'Chunks created': len(chunks),
                'Records inserted': inserted_count,
                'Table': args.table
            }
            
            display(HTML(self.renderer.success("Successfully embedded and inserted documents", details)))
            logger.info(f"Inserted {inserted_count} records into {args.table}")
            
            return f"Inserted {inserted_count} records into {args.table}"
            
        except SyntaxError as e:
            display(HTML(self.renderer.error(
                "Syntax error in cell content", 
                "Cell must contain valid Python code that evaluates to a list of strings"
            )))
            return None
        except Exception as e:
            display(HTML(self.renderer.error("Processing error", str(e))))
            logger.error(f"Error processing documents: {str(e)}")
            return None
    
    def _process_documents(self, documents: List[str], chunk_size: int, overlap: int) -> Tuple[List[str], List[Dict]]:
        """Process documents into chunks with metadata."""
        chunks = []
        chunk_metadata = []
        
        for doc_idx, doc in enumerate(documents):
            if chunk_size > 0 and len(doc) > chunk_size:
                # Split into overlapping chunks
                for i in range(0, len(doc), chunk_size - overlap):
                    chunk = doc[i:i + chunk_size]
                    chunks.append(chunk)
                    chunk_metadata.append({
                        'original_doc_index': doc_idx,
                        'chunk_index': len(chunks) - 1,
                        'is_chunk': True,
                        'chunk_start': i,
                        'chunk_end': i + len(chunk)
                    })
            else:
                chunks.append(doc)
                chunk_metadata.append({
                    'original_doc_index': doc_idx,
                    'chunk_index': len(chunks) - 1,
                    'is_chunk': False
                })
        
        return chunks, chunk_metadata
    
    def _generate_embeddings_batch(self, chunks: List[str], batch_size: int) -> List:
        """Generate embeddings for chunks in batches."""
        all_embeddings = []
        
        print("Generating embeddings...")
        for i in tqdm(range(0, len(chunks), batch_size), desc="Batches"):
            batch = chunks[i:i + batch_size]
            embeddings = self.embedding_model.encode(
                batch,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            all_embeddings.extend(embeddings)
        
        return all_embeddings
    
    def _insert_embeddings(self, table: str, chunks: List[str], embeddings: List, metadata: List[Dict]) -> int:
        """Insert embeddings into database."""
        insert_query = f"""
        INSERT INTO {table} (document_text, metadata, embedding)
        VALUES (?, ?, VEC_FromText(?))
        """
        
        inserted_count = 0
        print("Inserting into database...")
        
        for chunk, embedding, meta in tqdm(
            zip(chunks, embeddings, metadata),
            total=len(chunks),
            desc="Inserting"
        ):
            # Convert embedding to string format for MariaDB
            embedding_str = '[' + ','.join(map(str, embedding)) + ']'
            
            self.cursor.execute(
                insert_query,
                (chunk, json.dumps(meta), embedding_str)
            )
            inserted_count += 1
        
        self.connection.commit()
        return inserted_count
    
    @line_magic
    def query_table(self, line):
        """
        Quick query to inspect table contents.
        
        Usage: %query_table documents
        
        Args:
            line: Table name
        
        Returns:
            Table information or None on failure
        """
        table_name = line.strip()
        
        if not self._validate_connection():
            return None
        
        if not table_name:
            display(HTML(self.renderer.error("Table name is required")))
            return None
        
        try:
            # Get count
            self.cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = self.cursor.fetchone()[0]
            
            # Get sample rows
            self.cursor.execute(f"SELECT id, LEFT(document_text, 100) as text_preview FROM {table_name} LIMIT 5")
            rows = self.cursor.fetchall()
            
            rows_html = ""
            for row in rows:
                rows_html += f"""
                <tr>
                    <td style='padding: 5px; border: 1px solid #ddd;'>{row[0]}</td>
                    <td style='padding: 5px; border: 1px solid #ddd;'>{row[1]}...</td>
                </tr>
                """
            
            display(HTML(f"""
            <div style='padding: 10px; background-color: #d1ecf1; border: 1px solid #bee5eb; border-radius: 5px; color: #0c5460;'>
                <b>Table: {table_name}</b><br>
                Total records: <b>{count}</b><br><br>
                <table style='border-collapse: collapse; margin-top: 10px;'>
                    <tr style='background-color: #0c5460; color: white;'>
                        <th style='padding: 5px; border: 1px solid #ddd;'>ID</th>
                        <th style='padding: 5px; border: 1px solid #ddd;'>Text Preview</th>
                    </tr>
                    {rows_html}
                </table>
            </div>
            """))
            
            logger.info(f"Queried table {table_name}: {count} records")
            return {'table': table_name, 'count': count, 'sample_rows': rows}
            
        except mariadb.Error as e:
            display(HTML(self.renderer.error("Database error", str(e))))
            logger.error(f"Error querying table {table_name}: {str(e)}")
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
        Perform semantic search on vector table using cosine or euclidean distance.
        
        Usage: %semantic_search "machine learning" --table documents --top_k 3 --show_distance
        
        Args:
            query: Search query text (required)
            --table: Table to search (required)
            --top_k: Number of results to return (default: 5)
            --threshold: Distance threshold filter (optional)
            --show_distance: Show distance scores in results
        
        Returns:
            List of search results or None on failure
        """
        args = parse_argstring(self.semantic_search, line)
        
        if not self._validate_connection() or not self._validate_embedding_model():
            return None
        
        if not self._validate_table_exists(args.table):
            return None
        
        try:
            query_text = args.query.strip('"').strip("'")
            display(HTML(self.renderer.info(f"Searching for: '{query_text}'")))
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query_text, convert_to_numpy=True)
            embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            # Determine distance function
            distance_type = self.table_distance_types.get(args.table, 'cosine')
            distance_func = "VEC_DISTANCE_COSINE" if distance_type == 'cosine' else "VEC_DISTANCE_EUCLIDEAN"
            
            # Build and execute query
            if args.show_distance:
                query_sql = f"""
                SELECT id, document_text, metadata, 
                    {distance_func}(embedding, VEC_FromText(?)) as distance
                FROM {args.table}
                ORDER BY {distance_func}(embedding, VEC_FromText(?))
                LIMIT ?
                """
                params = (embedding_str, embedding_str, args.top_k)
            else:
                query_sql = f"""
                SELECT id, document_text, metadata
                FROM {args.table}
                ORDER BY {distance_func}(embedding, VEC_FromText(?))
                LIMIT ?
                """
                params = (embedding_str, args.top_k)
            
            self.cursor.execute(query_sql, params)
            results = self.cursor.fetchall()
            
            if not results:
                display(HTML(self.renderer.warning("No results found")))
                return []
            
            # Apply threshold filter if specified
            if args.threshold is not None and args.show_distance:
                results = [r for r in results if r[-1] <= args.threshold]
            
            # Display results
            self._display_search_results(results, args.show_distance)
            
            logger.info(f"Semantic search returned {len(results)} results for query: {query_text}")
            return results
            
        except Exception as e:
            display(HTML(self.renderer.error("Search error", str(e))))
            logger.error(f"Error in semantic search: {str(e)}")
            return None
    
    def _display_search_results(self, results: List[Tuple], show_distance: bool):
        """Display formatted search results."""
        results_html = ""
        
        for idx, result in enumerate(results, 1):
            if show_distance:
                doc_id, text, metadata, distance = result
                distance_badge = f"""
                <span style='background-color: #007bff; color: white; padding: 2px 8px; 
                border-radius: 10px; font-size: 0.85em;'>Distance: {distance:.4f}</span>
                """
            else:
                doc_id, text, metadata = result
                distance_badge = ""
            
            # Parse metadata
            try:
                meta_dict = json.loads(metadata) if metadata else {}
                meta_str = " | ".join([f"{k}: {v}" for k, v in meta_dict.items()])
            except:
                meta_str = ""
            
            results_html += f"""
            <div style='padding: 12px; margin: 10px 0; background-color: #f8f9fa; 
            border-left: 4px solid #007bff; border-radius: 4px;'>
                <div style='margin-bottom: 8px;'>
                    <b>Result #{idx}</b> {distance_badge}
                    <span style='color: #6c757d; font-size: 0.9em; margin-left: 10px;'>ID: {doc_id}</span>
                </div>
                <div style='margin: 8px 0;'>{text[:500]}{'...' if len(text) > 500 else ''}</div>
                {f"<div style='color: #6c757d; font-size: 0.85em; margin-top: 8px;'>{meta_str}</div>" if meta_str else ""}
            </div>
            """
        
        display(HTML(f"""
        <div style='padding: 15px; background-color: white; border: 1px solid #dee2e6; border-radius: 5px;'>
            <div style='padding: 10px; background-color: #d4edda; border: 1px solid #c3e6cb; 
            border-radius: 5px; color: #155724; margin-bottom: 15px;'>
                <b>Found {len(results)} results</b>
            </div>
            {results_html}
        </div>
        """))
    
    @line_magic
    @magic_arguments()
    @argument('question', type=str, help='Question to answer')
    @argument('--table', required=True, help='Table to search for context')
    @argument('--top_k', type=int, default=3, help='Number of context documents')
    @argument('--api_key', default=None, help='Groq API key')
    @argument('--model', default='llama-3.3-70b-versatile', help='Groq model to use')
    @argument('--temperature', type=float, default=0.2, help='LLM temperature (0-1)')
    def rag_query(self, line):
        """
        Complete RAG pipeline: retrieval + augmented generation.
        
        Usage: %rag_query "What is MariaDB?" --table documents --api_key YOUR_KEY
        
        Args:
            question: Question to answer (required)
            --table: Table to search for context (required)
            --top_k: Number of context documents to retrieve (default: 3)
            --api_key: Groq API key (or set GROQ_API_KEY env var)
            --model: Groq model to use (default: llama-3.3-70b-versatile)
            --temperature: LLM temperature 0-1 (default: 0.2)
        
        Returns:
            Dict with question, answer, context, and distances or None on failure
        """
        args = parse_argstring(self.rag_query, line)
        
        if not self._validate_connection() or not self._validate_embedding_model():
            return None
        
        if not self._validate_table_exists(args.table):
            return None
        
        # Get API key
        api_key = args.api_key or os.environ.get('GROQ_API_KEY')
        if not api_key:
            display(HTML(self.renderer.error(
                "Groq API key required",
                "Set GROQ_API_KEY environment variable or use --api_key parameter. Get free key at: https://console.groq.com/"
            )))
            return None
        
        try:
            from groq import Groq
            
            question_text = args.question.strip('"').strip("'")
            
            display(HTML(self.renderer.info(
                f"RAG Query: {question_text}",
                {'Step': '1/3 - Retrieving relevant context'}
            )))
            
            # Step 1: Semantic search for context
            query_embedding = self.embedding_model.encode(question_text, convert_to_numpy=True)
            embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            # Get distance function
            distance_type = self.table_distance_types.get(args.table, 'cosine')
            distance_func = "VEC_DISTANCE_COSINE" if distance_type == 'cosine' else "VEC_DISTANCE_EUCLIDEAN"
            
            query_sql = f"""
            SELECT document_text, {distance_func}(embedding, VEC_FromText(?)) as distance
            FROM {args.table}
            ORDER BY {distance_func}(embedding, VEC_FromText(?))
            LIMIT ?
            """
            
            self.cursor.execute(query_sql, (embedding_str, embedding_str, args.top_k))
            results = self.cursor.fetchall()
            
            if not results:
                display(HTML(self.renderer.warning("No context found in database. Cannot answer question.")))
                return None
            
            # Build context from retrieved documents
            context = "\n\n".join([f"[Context {i+1}]: {doc[0]}" for i, doc in enumerate(results)])
            
            display(HTML(self.renderer.info(
                f"Retrieved {len(results)} relevant documents",
                {'Step': f'2/3 - Generating answer with {args.model}'}
            )))
            
            # Step 2: Build prompt with context
            prompt = f"""You are a helpful AI assistant. Answer the question based on the provided context. If the context doesn't contain enough information to answer the question, say so clearly.

Context:
{context}

Question: {question_text}

Answer:"""
            
            # Step 3: Generate answer using Groq
            client = Groq(api_key=api_key)
            
            completion = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=args.temperature,
                max_tokens=1024
            )
            
            answer = completion.choices[0].message.content
            
            # Display formatted answer
            self._display_rag_results(question_text, answer, results, args.model, args.temperature)
            
            logger.info(f"RAG query completed for: {question_text}")
            
            return {
                'question': question_text,
                'answer': answer,
                'context': [doc[0] for doc in results],
                'distances': [doc[1] for doc in results]
            }
            
        except ImportError:
            display(HTML(self.renderer.error(
                "Groq library not installed",
                "Install with: pip install groq"
            )))
            return None
        except Exception as e:
            display(HTML(self.renderer.error("RAG query error", str(e))))
            logger.error(f"Error in RAG query: {str(e)}")
            return None
    
    def _display_rag_results(self, question: str, answer: str, results: List[Tuple], model: str, temperature: float):
        """Display formatted RAG results."""
        context_html = ""
        for i, (doc_text, distance) in enumerate(results, 1):
            context_html += f"""
            <div style='padding: 8px; margin: 5px 0; background-color: #f8f9fa;
            border-left: 3px solid #6c757d; font-size: 0.9em;'>
                <b>Context {i}</b> <span style='color: #6c757d;'>(distance: {distance:.4f})</span><br>
                {doc_text[:300]}{'...' if len(doc_text) > 300 else ''}
            </div>
            """
        
        display(HTML(f"""
        <div style='padding: 15px; background-color: white; border: 1px solid #dee2e6; border-radius: 5px; margin-top: 10px;'>
            <div style='padding: 12px; background-color: #d4edda; border: 1px solid #c3e6cb;
            border-radius: 5px; color: #155724; margin-bottom: 15px;'>
                <b>RAG Answer Complete</b>
            </div>
            
            <div style='margin-bottom: 20px;'>
                <h4 style='color: #007bff; margin-bottom: 10px;'>Answer:</h4>
                <div style='padding: 15px; background-color: #e7f3ff; border-left: 4px solid #007bff; border-radius: 4px;'>
                    {answer}
                </div>
            </div>
            
            <div>
                <h4 style='color: #6c757d; margin-bottom: 10px; font-size: 1em;'>Retrieved Context:</h4>
                {context_html}
            </div>
            
            <div style='margin-top: 15px; padding: 10px; background-color: #f8f9fa; border-radius: 4px; font-size: 0.85em; color: #6c757d;'>
                <b>Model:</b> {model} | <b>Temperature:</b> {temperature} | <b>Context docs:</b> {len(results)}
            </div>
        </div>
        """))


def load_ipython_extension(ipython):
    """Load the extension in IPython."""
    ipython.register_magics(MariaDBVectorMagics)
    logger.info("MariaDB Vector Magics extension loaded")


def unload_ipython_extension(ipython):
    """Unload the extension from IPython."""
    logger.info("MariaDB Vector Magics extension unloaded")
