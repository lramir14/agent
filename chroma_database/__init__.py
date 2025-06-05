"""
ChromaDB Document Management Package

This package provides tools for managing document collections in ChromaDB:
- CSV and PDF document uploading
- Semantic querying with Ollama integration
- Persistent storage management

Example usage:
    >>> from chroma_db.manager import ChromaManager
    >>> db = ChromaManager()
    >>> db.upload_csv("data.csv")
"""
from .manager import ChromaManager

# Package version
__version__ = "1.0.0"
__all__ = ['ChromaManager']