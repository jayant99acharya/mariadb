"""
RAG (Retrieval Augmented Generation) management for MariaDB Vector Magics.

This module handles the complete RAG pipeline including retrieval and generation.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import time

from .database import DatabaseManager
from .embeddings import EmbeddingManager
from .search import SearchManager
from .secrets import get_secret
from .exceptions import RAGError, SearchError, EmbeddingError

logger = logging.getLogger(__name__)


class RAGManager:
    """Manages RAG (Retrieval Augmented Generation) operations."""
    
    def __init__(self, db_manager: DatabaseManager, embedding_manager: EmbeddingManager, 
                 search_manager: SearchManager):
        """
        Initialize RAG manager.
        
        Args:
            db_manager: Database manager instance
            embedding_manager: Embedding manager instance
            search_manager: Search manager instance
        """
        self.db_manager = db_manager
        self.embedding_manager = embedding_manager
        self.search_manager = search_manager
        self._llm_client = None
    
    def query(self, question: str, table: str, top_k: int = 3,
              api_key: Optional[str] = None, model: str = 'llama-3.3-70b-versatile',
              temperature: float = 0.2, max_tokens: int = 1024,
              system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform complete RAG query: retrieval + generation.
        
        Args:
            question: Question to answer
            table: Table to search for context
            top_k: Number of context documents to retrieve
            api_key: LLM API key (if None, uses secrets manager)
            model: LLM model to use
            temperature: LLM temperature (0-1)
            max_tokens: Maximum tokens in response
            system_prompt: Custom system prompt
            
        Returns:
            Dict with question, answer, context, and metadata
            
        Raises:
            RAGError: If RAG operation fails
        """
        try:
            start_time = time.time()
            
            logger.info(f"Starting RAG query: '{question}' using table '{table}'")
            
            # Step 1: Retrieve relevant context
            context_docs = self._retrieve_context(question, table, top_k)
            
            if not context_docs:
                raise RAGError(
                    "No relevant context found in database. Cannot answer question.",
                    question, model
                )
            
            retrieval_time = time.time() - start_time
            logger.info(f"Retrieved {len(context_docs)} context documents in {retrieval_time:.2f}s")
            
            # Step 2: Generate answer using LLM
            generation_start = time.time()
            answer = self._generate_answer(
                question, context_docs, api_key, model, 
                temperature, max_tokens, system_prompt
            )
            generation_time = time.time() - generation_start
            
            total_time = time.time() - start_time
            
            logger.info(f"RAG query completed in {total_time:.2f}s")
            
            return {
                'question': question,
                'answer': answer,
                'context': [doc[1] for doc in context_docs],  # Extract text content
                'context_metadata': [
                    {
                        'id': doc[0],
                        'distance': doc[3] if len(doc) > 3 else None,
                        'metadata': doc[2] if len(doc) > 2 else None
                    }
                    for doc in context_docs
                ],
                'model': model,
                'temperature': temperature,
                'retrieval_time': retrieval_time,
                'generation_time': generation_time,
                'total_time': total_time,
                'context_count': len(context_docs)
            }
            
        except (SearchError, EmbeddingError) as e:
            error_msg = f"RAG query failed during retrieval: {e}"
            logger.error(error_msg)
            raise RAGError(error_msg, question, model) from e
        except Exception as e:
            error_msg = f"RAG query failed: {e}"
            logger.error(error_msg)
            raise RAGError(error_msg, question, model) from e
    
    def _retrieve_context(self, question: str, table: str, top_k: int) -> List[Tuple]:
        """Retrieve relevant context documents for the question."""
        try:
            # Perform semantic search to get relevant documents
            context_docs = self.search_manager.semantic_search(
                query=question,
                table=table,
                top_k=top_k,
                include_distance=True
            )
            
            return context_docs
            
        except SearchError as e:
            logger.error(f"Context retrieval failed: {e}")
            raise
    
    def _generate_answer(self, question: str, context_docs: List[Tuple],
                        api_key: Optional[str], model: str, temperature: float,
                        max_tokens: int, system_prompt: Optional[str]) -> str:
        """Generate answer using LLM with retrieved context."""
        try:
            # Get API key
            effective_api_key = api_key or get_secret('GROQ_API_KEY')
            if not effective_api_key:
                raise RAGError(
                    "LLM API key required. Set GROQ_API_KEY in secrets or provide --api_key parameter. "
                    "Get free key at: https://console.groq.com/",
                    question, model
                )
            
            # Build context from retrieved documents
            context_text = self._build_context_text(context_docs)
            
            # Build prompt
            prompt = self._build_prompt(question, context_text, system_prompt)
            
            # Generate answer using LLM
            answer = self._call_llm(
                prompt, effective_api_key, model, temperature, max_tokens
            )
            
            return answer
            
        except Exception as e:
            error_msg = f"Answer generation failed: {e}"
            logger.error(error_msg)
            raise RAGError(error_msg, question, model) from e
    
    def _build_context_text(self, context_docs: List[Tuple]) -> str:
        """Build formatted context text from retrieved documents."""
        context_parts = []
        
        for i, doc in enumerate(context_docs, 1):
            doc_text = doc[1]  # Document text is at index 1
            
            # Truncate very long documents
            if len(doc_text) > 1000:
                doc_text = doc_text[:1000] + "..."
            
            context_parts.append(f"[Context {i}]: {doc_text}")
        
        return "\n\n".join(context_parts)
    
    def _build_prompt(self, question: str, context: str, system_prompt: Optional[str]) -> str:
        """Build the complete prompt for the LLM."""
        default_system = (
            "You are a helpful AI assistant. Answer the question based on the provided context. "
            "If the context doesn't contain enough information to answer the question, say so clearly. "
            "Be concise and accurate in your response."
        )
        
        system_text = system_prompt or default_system
        
        return f"""{system_text}

Context:
{context}

Question: {question}

Answer:"""
    
    def _call_llm(self, prompt: str, api_key: str, model: str, 
                  temperature: float, max_tokens: int) -> str:
        """Call the LLM API to generate an answer."""
        try:
            # Import Groq client
            try:
                from groq import Groq
            except ImportError:
                raise RAGError(
                    "Groq library not installed. Install with: pip install groq",
                    model=model
                )
            
            # Initialize client if needed
            if self._llm_client is None or getattr(self._llm_client, '_api_key', None) != api_key:
                self._llm_client = Groq(api_key=api_key)
                self._llm_client._api_key = api_key  # Store for comparison
            
            # Make API call
            completion = self._llm_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            answer = completion.choices[0].message.content
            
            if not answer or not answer.strip():
                raise RAGError("LLM returned empty response", model=model)
            
            return answer.strip()
            
        except Exception as e:
            if "groq" in str(e).lower() or "api" in str(e).lower():
                error_msg = f"LLM API call failed: {e}"
            else:
                error_msg = f"Unexpected error calling LLM: {e}"
            
            logger.error(error_msg)
            raise RAGError(error_msg, model=model) from e
    
    def batch_query(self, questions: List[str], table: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Process multiple RAG queries in batch.
        
        Args:
            questions: List of questions to process
            table: Table to search for context
            **kwargs: Additional parameters for individual queries
            
        Returns:
            List of RAG results
            
        Raises:
            RAGError: If batch processing fails
        """
        try:
            logger.info(f"Processing batch of {len(questions)} RAG queries")
            
            results = []
            errors = []
            
            for i, question in enumerate(questions):
                try:
                    result = self.query(question, table, **kwargs)
                    results.append(result)
                    logger.debug(f"Completed query {i+1}/{len(questions)}")
                except Exception as e:
                    error_info = {
                        'question_index': i,
                        'question': question,
                        'error': str(e)
                    }
                    errors.append(error_info)
                    logger.error(f"Failed query {i+1}/{len(questions)}: {e}")
            
            logger.info(f"Batch processing completed: {len(results)} successful, {len(errors)} failed")
            
            if errors:
                logger.warning(f"Batch had {len(errors)} errors: {errors}")
            
            return results
            
        except Exception as e:
            error_msg = f"Batch RAG processing failed: {e}"
            logger.error(error_msg)
            raise RAGError(error_msg) from e
    
    def evaluate_answer_quality(self, question: str, answer: str, context: List[str],
                               ground_truth: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate the quality of a RAG answer.
        
        Args:
            question: Original question
            answer: Generated answer
            context: Retrieved context documents
            ground_truth: Optional ground truth answer for comparison
            
        Returns:
            Dict with quality metrics
        """
        try:
            metrics = {
                'answer_length': len(answer),
                'context_count': len(context),
                'context_total_length': sum(len(doc) for doc in context),
                'answer_to_context_ratio': len(answer) / max(sum(len(doc) for doc in context), 1)
            }
            
            # Check if answer indicates insufficient context
            insufficient_indicators = [
                "don't have enough information",
                "context doesn't contain",
                "cannot answer",
                "insufficient information",
                "not enough context"
            ]
            
            metrics['indicates_insufficient_context'] = any(
                indicator in answer.lower() for indicator in insufficient_indicators
            )
            
            # Basic relevance check (keyword overlap)
            question_words = set(question.lower().split())
            answer_words = set(answer.lower().split())
            context_words = set()
            for doc in context:
                context_words.update(doc.lower().split())
            
            metrics['question_answer_overlap'] = len(question_words & answer_words) / max(len(question_words), 1)
            metrics['context_answer_overlap'] = len(context_words & answer_words) / max(len(answer_words), 1)
            
            # If ground truth is provided, calculate similarity
            if ground_truth:
                ground_truth_words = set(ground_truth.lower().split())
                metrics['ground_truth_overlap'] = len(ground_truth_words & answer_words) / max(len(ground_truth_words), 1)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Answer quality evaluation failed: {e}")
            return {'error': str(e)}
    
    def get_rag_statistics(self, table: str) -> Dict[str, Any]:
        """
        Get RAG-related statistics for a table.
        
        Args:
            table: Table name
            
        Returns:
            Dict with RAG statistics
        """
        try:
            # Get search statistics
            search_stats = self.search_manager.get_search_statistics(table)
            
            # Add RAG-specific information
            rag_stats = {
                'table': table,
                'total_documents': search_stats.get('total_documents', 0),
                'avg_document_length': search_stats.get('avg_text_length', 0),
                'metadata_fields': search_stats.get('metadata_fields', []),
                'distance_metric': search_stats.get('table_config', 'unknown'),
                'embedding_model': self.embedding_manager.model_name if self.embedding_manager.is_loaded() else None,
                'embedding_dimension': self.embedding_manager.get_embedding_dimension() if self.embedding_manager.is_loaded() else None
            }
            
            return rag_stats
            
        except Exception as e:
            error_msg = f"Failed to get RAG statistics: {e}"
            logger.error(error_msg)
            raise RAGError(error_msg, table=table) from e
    
    def clear_llm_client(self) -> None:
        """Clear the cached LLM client."""
        self._llm_client = None
        logger.debug("LLM client cache cleared")