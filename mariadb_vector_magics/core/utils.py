"""
Utility functions and classes for MariaDB Vector Magics.

This module contains shared utilities used across the application.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class HTMLRenderer:
    """Utility class for rendering HTML output in Jupyter notebooks."""
    
    @staticmethod
    def success(message: str, details: Optional[Dict[str, Any]] = None) -> str:
        """Render success message with optional details."""
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
        """Render error message with optional details."""
        details_html = f"<br><b>Details:</b> {error_details}" if error_details else ""
        return f"""
        <div style='padding: 10px; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px; color: #721c24;'>
            <b>ERROR:</b> {message}{details_html}
        </div>
        """
    
    @staticmethod
    def info(message: str, details: Optional[Dict[str, Any]] = None) -> str:
        """Render info message with optional details."""
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
    
    @staticmethod
    def table(headers: List[str], rows: List[List[str]], title: Optional[str] = None) -> str:
        """Render a formatted table."""
        title_html = f"<h4>{title}</h4>" if title else ""
        
        header_html = "".join([f"<th style='padding: 8px; border: 1px solid #ddd; background-color: #f8f9fa;'>{h}</th>" for h in headers])
        
        rows_html = ""
        for row in rows:
            row_html = "".join([f"<td style='padding: 8px; border: 1px solid #ddd;'>{cell}</td>" for cell in row])
            rows_html += f"<tr>{row_html}</tr>"
        
        return f"""
        <div style='padding: 10px; background-color: white; border: 1px solid #dee2e6; border-radius: 5px;'>
            {title_html}
            <table style='border-collapse: collapse; width: 100%;'>
                <thead><tr>{header_html}</tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
        """
    
    @staticmethod
    def search_results(results: List[Tuple], show_distance: bool = False, query: Optional[str] = None) -> str:
        """Render search results in a formatted layout."""
        if not results:
            return HTMLRenderer.warning("No results found")
        
        query_html = f"<b>Query:</b> '{query}'<br>" if query else ""
        
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
                import json
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
        
        return f"""
        <div style='padding: 15px; background-color: white; border: 1px solid #dee2e6; border-radius: 5px;'>
            <div style='padding: 10px; background-color: #d4edda; border: 1px solid #c3e6cb; 
            border-radius: 5px; color: #155724; margin-bottom: 15px;'>
                {query_html}<b>Found {len(results)} results</b>
            </div>
            {results_html}
        </div>
        """


@dataclass
class ProcessingStats:
    """Statistics for document processing operations."""
    documents_processed: int = 0
    chunks_created: int = 0
    embeddings_generated: int = 0
    records_inserted: int = 0
    processing_time: float = 0.0
    errors: int = 0


class TextProcessor:
    """Utility class for text processing operations."""
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int, overlap: int = 0) -> List[str]:
        """Split text into overlapping chunks."""
        if chunk_size <= 0:
            return [text]
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            
            if end >= len(text):
                break
                
            start = end - overlap
        
        return chunks
    
    @staticmethod
    def validate_documents(documents: List[str]) -> Tuple[bool, Optional[str]]:
        """Validate a list of documents."""
        if not isinstance(documents, list):
            return False, "Documents must be a list"
        
        if len(documents) == 0:
            return False, "Document list cannot be empty"
        
        for i, doc in enumerate(documents):
            if not isinstance(doc, str):
                return False, f"Document at index {i} must be a string"
            
            if len(doc.strip()) == 0:
                return False, f"Document at index {i} cannot be empty"
        
        return True, None
    
    @staticmethod
    def sanitize_text(text: str) -> str:
        """Sanitize text for safe processing."""
        # Remove null bytes and other problematic characters
        text = text.replace('\x00', '')
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text.strip()


class ValidationUtils:
    """Utility class for input validation."""
    
    @staticmethod
    def validate_table_name(table_name: str) -> Tuple[bool, Optional[str]]:
        """Validate table name for SQL safety."""
        if not table_name:
            return False, "Table name cannot be empty"
        
        if not table_name.replace('_', '').replace('-', '').isalnum():
            return False, "Table name can only contain letters, numbers, underscores, and hyphens"
        
        if len(table_name) > 64:
            return False, "Table name cannot exceed 64 characters"
        
        if table_name[0].isdigit():
            return False, "Table name cannot start with a number"
        
        return True, None
    
    @staticmethod
    def validate_dimensions(dimensions: int) -> Tuple[bool, Optional[str]]:
        """Validate vector dimensions."""
        if dimensions <= 0:
            return False, "Dimensions must be positive"
        
        if dimensions > 65535:
            return False, "Dimensions cannot exceed 65535"
        
        return True, None
    
    @staticmethod
    def validate_distance_metric(distance: str) -> Tuple[bool, Optional[str]]:
        """Validate distance metric."""
        valid_metrics = ['cosine', 'euclidean']
        if distance.lower() not in valid_metrics:
            return False, f"Distance metric must be one of: {', '.join(valid_metrics)}"
        
        return True, None
    
    @staticmethod
    def validate_m_value(m_value: int) -> Tuple[bool, Optional[str]]:
        """Validate HNSW M value."""
        if not (3 <= m_value <= 200):
            return False, "M value must be between 3 and 200"
        
        return True, None


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"