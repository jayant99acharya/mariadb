"""
MariaDB Vector Magics - IPython magic commands for MariaDB Vector operations.

This package provides seamless integration between Jupyter notebooks and MariaDB Vector
databases, enabling vector operations, semantic search, and RAG workflows through
intuitive magic commands.

Author: Jayant Acharya
License: MIT
Version: 0.1.0
"""

__version__ = '0.1.0'
__author__ = 'Jayant Acharya'
__email__ = 'jayant99acharya@gmail.com'
__license__ = 'MIT'

# Import main classes for easy access
from .magics.main import MariaDBVectorMagics
from .core.exceptions import MariaDBVectorError
from .core.utils import HTMLRenderer
from .core.secrets import get_secret, set_secret, get_groq_api_key, set_groq_api_key

# Define what gets imported with "from mariadb_vector_magics import *"
__all__ = [
    'MariaDBVectorMagics',
    'MariaDBVectorError',
    'HTMLRenderer',
    'get_secret',
    'set_secret',
    'get_groq_api_key',
    'set_groq_api_key',
    '__version__',
    '__author__',
    '__email__',
    '__license__',
]

def load_ipython_extension(ipython):
    """Load the extension in IPython/Jupyter."""
    from .magics.main import load_ipython_extension
    return load_ipython_extension(ipython)

def unload_ipython_extension(ipython):
    """Unload the extension from IPython/Jupyter."""
    from .magics.main import unload_ipython_extension
    return unload_ipython_extension(ipython)