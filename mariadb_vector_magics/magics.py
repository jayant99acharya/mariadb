from IPython.core.magic import (Magics, magics_class, line_magic, cell_magic)
from IPython.core.magic_arguments import (argument, magic_arguments, parse_argstring)
from IPython.display import display, HTML, clear_output
from tqdm.auto import tqdm

import mariadb
from sentence_transformers import SentenceTransformer
import json

@magics_class
class MariaDBVectorMagics(Magics):
    """Custom magic commands for MariaDB Vector operations"""
    
    def __init__(self, shell):
        super(MariaDBVectorMagics, self).__init__(shell)
        self.connection = None
        self.cursor = None
        self.embedding_model = None
        
    @line_magic
    @magic_arguments()
    @argument('--host', default='127.0.0.1', help='MariaDB host')  # Changed from localhost
    @argument('--port', type=int, default=3306, help='MariaDB port')
    @argument('--user', default='root', help='MariaDB user')
    @argument('--password', required=True, help='MariaDB password')
    @argument('--database', default='vectordb', help='Database name')
    def connect_mariadb(self, line):
        """
        Connect to MariaDB server
        Usage: %connect_mariadb --password=yourpassword --database=vectordb
        """
        args = parse_argstring(self.connect_mariadb, line)
        
        try:
            self.connection = mariadb.connect(
                host=args.host,
                port=args.port,
                user=args.user,
                password=args.password,
                database=args.database,
                local_infile=True  # Added for file operations later
            )
            self.cursor = self.connection.cursor()
            
            # Check if Vector support is available
            self.cursor.execute("SELECT VERSION()")
            version = self.cursor.fetchone()[0]
            
            display(HTML(f"""
            <div style='padding: 10px; background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 5px; color: #155724;'>
                ✅ <b>Connected to MariaDB!</b><br>
                Version: {version}<br>
                Database: {args.database}
            </div>
            """))
            
            return "Connected successfully"
            
        except mariadb.Error as e:
            display(HTML(f"""
            <div style='padding: 10px; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px; color: #721c24;'>
                ❌ <b>Connection Error:</b> {e}
            </div>
            """))
            return None
    
    @line_magic
    def load_embedding_model(self, line):
        """
        Load a sentence transformer model for embeddings
        Usage: %load_embedding_model all-MiniLM-L6-v2
        """
        model_name = line.strip() or "all-MiniLM-L6-v2"
        
        try:
            display(HTML(f"""
            <div style='padding: 10px; background-color: #d1ecf1; border: 1px solid #bee5eb; border-radius: 5px; color: #0c5460;'>
                ⏳ Loading embedding model: <b>{model_name}</b>...
            </div>
            """))
            
            self.embedding_model = SentenceTransformer(model_name)
            self.embedding_model.max_seq_length = 256
            
            display(HTML(f"""
            <div style='padding: 10px; background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 5px; color: #155724;'>
                ✅ <b>Model loaded successfully!</b><br>
                Model: {model_name}<br>
                Max sequence length: 256<br>
                Embedding dimension: {self.embedding_model.get_sentence_embedding_dimension()}
            </div>
            """))
            
            return f"Model {model_name} loaded"
            
        except Exception as e:
            display(HTML(f"""
            <div style='padding: 10px; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px; color: #721c24;'>
                ❌ <b>Error loading model:</b> {e}
            </div>
            """))
            return None
        
    @line_magic
    @magic_arguments()
    @argument('--table', required=True, help='Name of the vector table to create')
    @argument('--dimensions', type=int, default=384, help='Vector embedding dimensions (default: 384 for all-MiniLM-L6-v2)')
    @argument('--distance', default='cosine', choices=['cosine', 'euclidean'], help='Distance metric for vector search')
    @argument('--m_value', type=int, default=16, help='HNSW M parameter (3-200, higher=more accurate but slower)')
    @argument('--drop_if_exists', action='store_true', help='Drop table if it already exists')
    def create_vector_store(self, line):
        """
        Create a MariaDB table optimized for vector storage
        Usage: %create_vector_store --table my_docs --dimensions 384 --distance cosine
        """
        args = parse_argstring(self.create_vector_store, line)

        if not self.connection:
            display(HTML("""
            <div style='padding: 10px; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px; color: #721c24;'>
                ❌ <b>Error:</b> Not connected to MariaDB. Use %connect_mariadb first.
            </div>
            """))
            return None

        try:
            # Validate M value
            if args.m_value < 3 or args.m_value > 200:
                raise ValueError("M value must be between 3 and 200")
            
            # Drop table if requested
            if args.drop_if_exists:
                self.cursor.execute(f"DROP TABLE IF EXISTS {args.table}")
                display(HTML(f"""
                <div style='padding: 10px; background-color: #fff3cd; border: 1px solid #ffeeba; border-radius: 5px; color: #856404;'>
                    ⚠️ Dropped existing table: <b>{args.table}</b>
                </div>
                """))
            
            # Create the vector table
            create_table_sql = f"""
            CREATE TABLE {args.table} (
                id INT AUTO_INCREMENT PRIMARY KEY,
                document_text LONGTEXT NOT NULL,
                metadata JSON,
                embedding VECTOR({args.dimensions}) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                VECTOR INDEX (embedding) M={args.m_value} DISTANCE={args.distance}
            )
            """
            
            self.cursor.execute(create_table_sql)
            self.connection.commit()
            if not hasattr(self, 'table_distance_types'):
                self.table_distance_types = {}
                self.table_distance_types[args.table] = args.distance
            
            display(HTML(f"""
            <div style='padding: 10px; background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 5px; color: #155724;'>
                ✅ <b>Vector Store Created Successfully!</b><br><br>
                <table style='margin-top: 10px; border-collapse: collapse;'>
                    <tr><td style='padding: 5px;'><b>Table:</b></td><td style='padding: 5px;'>{args.table}</td></tr>
                    <tr><td style='padding: 5px;'><b>Dimensions:</b></td><td style='padding: 5px;'>{args.dimensions}</td></tr>
                    <tr><td style='padding: 5px;'><b>Distance Metric:</b></td><td style='padding: 5px;'>{args.distance}</td></tr>
                    <tr><td style='padding: 5px;'><b>HNSW M Value:</b></td><td style='padding: 5px;'>{args.m_value}</td></tr>
                </table>
                <br>
                <b>Schema:</b>
                <ul style='margin-top: 5px;'>
                    <li><code>id</code> - Auto-increment primary key</li>
                    <li><code>document_text</code> - Your text content</li>
                    <li><code>metadata</code> - JSON for additional info</li>
                    <li><code>embedding</code> - Vector({args.dimensions}) with index</li>
                    <li><code>created_at</code> - Timestamp</li>
                </ul>
            </div>
            """))
            
            return f"Table {args.table} created"
            
        except mariadb.Error as e:
            display(HTML(f"""
            <div style='padding: 10px; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px; color: #721c24;'>
                ❌ <b>Database Error:</b> {e}
            </div>
            """))
            return None
        except Exception as e:
            display(HTML(f"""
            <div style='padding: 10px; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px; color: #721c24;'>
                ❌ <b>Error:</b> {e}
            </div>
            """))
            return None
    
    @line_magic
    def show_vector_tables(self, line):
        """Show all tables in current database"""
        if not self.connection:
            display(HTML("""
            <div style='padding: 10px; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px; color: #721c24;'>
                ❌ <b>Error:</b> Not connected to MariaDB.
            </div>
            """))
            return None
        
        self.cursor.execute("SHOW TABLES")
        tables = self.cursor.fetchall()
        
        if tables:
            table_list = '<br>'.join([f"• {t[0]}" for t in tables])
            display(HTML(f"""
            <div style='padding: 10px; background-color: #d1ecf1; border: 1px solid #bee5eb; border-radius: 5px; color: #0c5460;'>
                <b>📊 Tables in database:</b><br><br>
                {table_list}
            </div>
            """))
        else:
            display(HTML("""
            <div style='padding: 10px; background-color: #fff3cd; border: 1px solid #ffeeba; border-radius: 5px; color: #856404;'>
                No tables found in database.
            </div>
            """))

    @cell_magic
    @magic_arguments()
    @argument('--table', required=True, help='Target vector table name')
    @argument('--batch_size', type=int, default=32, help='Batch size for embedding generation')
    @argument('--chunk_size', type=int, default=500, help='Max characters per text chunk (0 to disable chunking)')
    @argument('--overlap', type=int, default=50, help='Character overlap between chunks')
    def embed_table(self, line, cell):
        """
        Generate embeddings and insert documents into vector table
        Usage: %%embed_table --table documents --batch_size 32
            ["text1", "text2", "text3"]
        Or:    %%embed_table --table documents
            import pandas as pd
            df = pd.read_csv('data.csv')
            df['text_column'].tolist()
        """
        args = parse_argstring(self.embed_table, line)
        
        # Validation checks
        if not self.connection:
            display(HTML("""
            <div style='padding: 10px; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px; color: #721c24;'>
                ❌ <b>Error:</b> Not connected to MariaDB. Use %connect_mariadb first.
            </div>
            """))
            return None
        
        if not self.embedding_model:
            display(HTML("""
            <div style='padding: 10px; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px; color: #721c24;'>
                ❌ <b>Error:</b> No embedding model loaded. Use %load_embedding_model first.
            </div>
            """))
            return None
        
        try:
            # Execute the cell content to get documents list
            documents = eval(cell.strip())
            
            if not isinstance(documents, list):
                raise ValueError("Cell must evaluate to a list of strings")
            
            if len(documents) == 0:
                raise ValueError("Document list is empty")
            
            display(HTML(f"""
            <div style='padding: 10px; background-color: #d1ecf1; border: 1px solid #bee5eb; border-radius: 5px; color: #0c5460;'>
                📚 Processing <b>{len(documents)}</b> documents...<br>
                Chunk size: {args.chunk_size} chars | Batch size: {args.batch_size}
            </div>
            """))
            
            # Chunk documents if needed
            chunks = []
            chunk_metadata = []
            
            for doc_idx, doc in enumerate(documents):
                if args.chunk_size > 0 and len(doc) > args.chunk_size:
                    # Split into overlapping chunks
                    for i in range(0, len(doc), args.chunk_size - args.overlap):
                        chunk = doc[i:i + args.chunk_size]
                        chunks.append(chunk)
                        chunk_metadata.append({
                            'original_doc_index': doc_idx,
                            'chunk_index': len(chunks) - 1,
                            'is_chunk': True
                        })
                else:
                    chunks.append(doc)
                    chunk_metadata.append({
                        'original_doc_index': doc_idx,
                        'chunk_index': len(chunks) - 1,
                        'is_chunk': False
                    })
            
            display(HTML(f"""
            <div style='padding: 10px; background-color: #d1ecf1; border: 1px solid #bee5eb; border-radius: 5px; color: #0c5460;'>
                📝 Created <b>{len(chunks)}</b> chunks from {len(documents)} documents
            </div>
            """))
            
            # Generate embeddings in batches
            all_embeddings = []
            
            print("🔄 Generating embeddings...")
            for i in tqdm(range(0, len(chunks), args.batch_size), desc="Batches"):
                batch = chunks[i:i + args.batch_size]
                embeddings = self.embedding_model.encode(
                    batch,
                    batch_size=args.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                all_embeddings.extend(embeddings)
            
            # Insert into database
            print("\n💾 Inserting into database...")
            insert_query = f"""
            INSERT INTO {args.table} (document_text, metadata, embedding)
            VALUES (?, ?, VEC_FromText(?))
            """
            
            inserted_count = 0
            for chunk, embedding, metadata in tqdm(
                zip(chunks, all_embeddings, chunk_metadata),
                total=len(chunks),
                desc="Inserting"
            ):
                # Convert embedding to string format for MariaDB
                embedding_str = '[' + ','.join(map(str, embedding)) + ']'
                
                self.cursor.execute(
                    insert_query,
                    (chunk, json.dumps(metadata), embedding_str)
                )
                inserted_count += 1
            
            self.connection.commit()
            
            display(HTML(f"""
            <div style='padding: 10px; background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 5px; color: #155724;'>
                ✅ <b>Successfully embedded and inserted!</b><br><br>
                <table style='margin-top: 10px; border-collapse: collapse;'>
                    <tr><td style='padding: 5px;'><b>Documents processed:</b></td><td style='padding: 5px;'>{len(documents)}</td></tr>
                    <tr><td style='padding: 5px;'><b>Chunks created:</b></td><td style='padding: 5px;'>{len(chunks)}</td></tr>
                    <tr><td style='padding: 5px;'><b>Records inserted:</b></td><td style='padding: 5px;'>{inserted_count}</td></tr>
                    <tr><td style='padding: 5px;'><b>Table:</b></td><td style='padding: 5px;'>{args.table}</td></tr>
                </table>
            </div>
            """))
            
            return f"Inserted {inserted_count} records into {args.table}"
            
        except SyntaxError as e:
            display(HTML(f"""
            <div style='padding: 10px; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px; color: #721c24;'>
                ❌ <b>Syntax Error:</b> {e}<br>
                Cell must contain valid Python code that evaluates to a list of strings.
            </div>
            """))
            return None
        except Exception as e:
            display(HTML(f"""
            <div style='padding: 10px; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px; color: #721c24;'>
                ❌ <b>Error:</b> {e}
            </div>
            """))
            return None

    @line_magic
    def query_table(self, line):
        """
        Quick query to check table contents
        Usage: %query_table documents
        """
        table_name = line.strip()
        
        if not self.connection:
            display(HTML("""
            <div style='padding: 10px; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px; color: #721c24;'>
                ❌ <b>Error:</b> Not connected to MariaDB.
            </div>
            """))
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
                rows_html += f"<tr><td style='padding: 5px; border: 1px solid #ddd;'>{row[0]}</td><td style='padding: 5px; border: 1px solid #ddd;'>{row[1]}...</td></tr>"
            
            display(HTML(f"""
            <div style='padding: 10px; background-color: #d1ecf1; border: 1px solid #bee5eb; border-radius: 5px; color: #0c5460;'>
                <b>📊 Table: {table_name}</b><br>
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
            
        except Exception as e:
            display(HTML(f"""
            <div style='padding: 10px; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px; color: #721c24;'>
                ❌ <b>Error:</b> {e}
            </div>
            """))

    @line_magic
    @magic_arguments()
    @argument('query', type=str, help='Search query text')
    @argument('--table', required=True, help='Table to search')
    @argument('--top_k', type=int, default=5, help='Number of results to return')
    @argument('--threshold', type=float, default=None, help='Distance threshold (optional)')
    @argument('--show_distance', action='store_true', help='Show distance scores')
    def semantic_search(self, line):
        """
        Perform semantic search on vector table
        Usage: %semantic_search "machine learning" --table documents --top_k 3
        """
        args = parse_argstring(self.semantic_search, line)
        
        # Validation
        if not self.connection:
            display(HTML("""
            <div style='padding: 10px; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px; color: #721c24;'>
                ❌ <b>Error:</b> Not connected to MariaDB.
            </div>
            """))
            return None
        
        if not self.embedding_model:
            display(HTML("""
            <div style='padding: 10px; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px; color: #721c24;'>
                ❌ <b>Error:</b> No embedding model loaded.
            </div>
            """))
            return None
        
        try:
            # Generate query embedding
            query_text = args.query.strip('"').strip("'")
            display(HTML(f"""
            <div style='padding: 10px; background-color: #d1ecf1; border: 1px solid #bee5eb; border-radius: 5px; color: #0c5460;'>
                🔍 Searching for: <b>"{query_text}"</b>
            </div>
            """))
            
            query_embedding = self.embedding_model.encode(query_text, convert_to_numpy=True)
            
            # Convert to MariaDB format
            embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            # Determine distance function - use explicit functions
            if hasattr(self, 'table_distance_types'):
                distance_type = self.table_distance_types.get(args.table, 'cosine')
            else:
                distance_type = 'cosine'  # Default
            
            distance_func = "VEC_DISTANCE_COSINE" if distance_type == 'cosine' else "VEC_DISTANCE_EUCLIDEAN"
            
            # Build query with explicit distance function
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
            
            # Execute search
            self.cursor.execute(query_sql, params)
            results = self.cursor.fetchall()
            
            if not results:
                display(HTML("""
                <div style='padding: 10px; background-color: #fff3cd; border: 1px solid #ffeeba; border-radius: 5px; color: #856404;'>
                    ⚠️ No results found.
                </div>
                """))
                return []
            
            # Apply threshold filter if specified
            if args.threshold is not None and args.show_distance:
                results = [r for r in results if r[-1] <= args.threshold]
            
            # Display results
            results_html = ""
            for idx, result in enumerate(results, 1):
                if args.show_distance:
                    doc_id, text, metadata, distance = result
                    distance_badge = f"<span style='background-color: #007bff; color: white; padding: 2px 8px; border-radius: 10px; font-size: 0.85em;'>Distance: {distance:.4f}</span>"
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
                <div style='padding: 12px; margin: 10px 0; background-color: #f8f9fa; border-left: 4px solid #007bff; border-radius: 4px;'>
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
                <div style='padding: 10px; background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 5px; color: #155724; margin-bottom: 15px;'>
                    ✅ Found <b>{len(results)}</b> results
                </div>
                {results_html}
            </div>
            """))
            
            # Return results as list for programmatic access
            return results
            
        except Exception as e:
            display(HTML(f"""
            <div style='padding: 10px; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px; color: #721c24;'>
                ❌ <b>Error:</b> {e}
            </div>
            """))
            import traceback
            traceback.print_exc()
            return None

    @line_magic
    @magic_arguments()
    @argument('question', type=str, help='Question to answer')
    @argument('--table', required=True, help='Table to search for context')
    @argument('--top_k', type=int, default=3, help='Number of context documents to retrieve')
    @argument('--api_key', default=None, help='Groq API key (or set GROQ_API_KEY env var)')
    @argument('--model', default='llama-3.3-70b-versatile', help='Groq model to use')
    @argument('--temperature', type=float, default=0.2, help='LLM temperature (0-1)')
    def rag_query(self, line):
        """
        Complete RAG pipeline: search + context + LLM answer
        Usage: %rag_query "What is MariaDB?" --table documents --api_key YOUR_KEY
        """
        args = parse_argstring(self.rag_query, line)
        
        # Validation
        if not self.connection:
            display(HTML("""
            <div style='padding: 10px; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px; color: #721c24;'>
                ❌ <b>Error:</b> Not connected to MariaDB.
            </div>
            """))
            return None
        
        if not self.embedding_model:
            display(HTML("""
            <div style='padding: 10px; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px; color: #721c24;'>
                ❌ <b>Error:</b> No embedding model loaded.
            </div>
            """))
            return None
        
        # Get API key
        import os
        api_key = args.api_key or os.environ.get('GROQ_API_KEY')
        if not api_key:
            display(HTML("""
            <div style='padding: 10px; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px; color: #721c24;'>
                ❌ <b>Error:</b> Groq API key required. Set GROQ_API_KEY env var or use --api_key<br>
                Get free key at: <a href="https://console.groq.com/" target="_blank">https://console.groq.com/</a>
            </div>
            """))
            return None
        
        try:
            from groq import Groq
            
            question_text = args.question.strip('"').strip("'")
            
            display(HTML(f"""
            <div style='padding: 10px; background-color: #d1ecf1; border: 1px solid #bee5eb; border-radius: 5px; color: #0c5460;'>
                🤔 <b>Question:</b> {question_text}<br>
                🔍 <b>Step 1/3:</b> Retrieving relevant context...
            </div>
            """))
            
            # Step 1: Semantic search for context
            query_embedding = self.embedding_model.encode(question_text, convert_to_numpy=True)
            embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            # Get distance function
            if hasattr(self, 'table_distance_types'):
                distance_type = self.table_distance_types.get(args.table, 'cosine')
            else:
                distance_type = 'cosine'
            
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
                display(HTML("""
                <div style='padding: 10px; background-color: #fff3cd; border: 1px solid #ffeeba; border-radius: 5px; color: #856404;'>
                    ⚠️ No context found in database. Cannot answer question.
                </div>
                """))
                return None
            
            # Build context from retrieved documents
            context = "\n\n".join([f"[Context {i+1}]: {doc[0]}" for i, doc in enumerate(results)])
            
            display(HTML(f"""
            <div style='padding: 10px; background-color: #d1ecf1; border: 1px solid #bee5eb; border-radius: 5px; color: #0c5460;'>
                ✅ <b>Retrieved {len(results)} relevant documents</b><br>
                🤖 <b>Step 2/3:</b> Generating answer with {args.model}...
            </div>
            """))
            
            # Step 2: Build prompt with context
            prompt = f"""You are a helpful AI assistant. Answer the question based on the provided context. If the context doesn't contain enough information to answer the question, say so.

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
            context_html = ""
            for i, (doc_text, distance) in enumerate(results, 1):
                context_html += f"""
                <div style='padding: 8px; margin: 5px 0; background-color: #f8f9fa; border-left: 3px solid #6c757d; font-size: 0.9em;'>
                    <b>Context {i}</b> <span style='color: #6c757d;'>(distance: {distance:.4f})</span><br>
                    {doc_text[:300]}{'...' if len(doc_text) > 300 else ''}
                </div>
                """
            
            display(HTML(f"""
            <div style='padding: 15px; background-color: white; border: 1px solid #dee2e6; border-radius: 5px; margin-top: 10px;'>
                <div style='padding: 12px; background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 5px; color: #155724; margin-bottom: 15px;'>
                    🎯 <b>RAG Answer Complete!</b>
                </div>
                
                <div style='margin-bottom: 20px;'>
                    <h4 style='color: #007bff; margin-bottom: 10px;'>💬 Answer:</h4>
                    <div style='padding: 15px; background-color: #e7f3ff; border-left: 4px solid #007bff; border-radius: 4px;'>
                        {answer}
                    </div>
                </div>
                
                <div>
                    <h4 style='color: #6c757d; margin-bottom: 10px; font-size: 1em;'>📚 Retrieved Context:</h4>
                    {context_html}
                </div>
                
                <div style='margin-top: 15px; padding: 10px; background-color: #f8f9fa; border-radius: 4px; font-size: 0.85em; color: #6c757d;'>
                    <b>Model:</b> {args.model} | <b>Temperature:</b> {args.temperature} | <b>Context docs:</b> {len(results)}
                </div>
            </div>
            """))
            
            return {
                'question': question_text,
                'answer': answer,
                'context': [doc[0] for doc in results],
                'distances': [doc[1] for doc in results]
            }
            
        except ImportError:
            display(HTML("""
            <div style='padding: 10px; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px; color: #721c24;'>
                ❌ <b>Error:</b> Groq library not installed. Run: pip install groq
            </div>
            """))
            return None
        except Exception as e:
            display(HTML(f"""
            <div style='padding: 10px; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px; color: #721c24;'>
                ❌ <b>Error:</b> {e}
            </div>
            """))
            import traceback
            traceback.print_exc()
            return None







def load_ipython_extension(ipython):
    """Load the extension in IPython."""
    ipython.register_magics(MariaDBVectorMagics)
