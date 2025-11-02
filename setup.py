"""
Setup configuration for MariaDB Vector Magics package.

This package provides IPython magic commands for seamless integration with MariaDB Vector
databases, enabling vector operations, semantic search, and RAG workflows.
"""

from setuptools import setup, find_packages
import os
import re

# Read version from __init__.py
def get_version():
    """Extract version from package __init__.py file."""
    init_file = os.path.join(os.path.dirname(__file__), 'mariadb_vector_magics', '__init__.py')
    if os.path.exists(init_file):
        with open(init_file, 'r', encoding='utf-8') as f:
            content = f.read()
            version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", content, re.M)
            if version_match:
                return version_match.group(1)
    return '0.1.0'

# Read long description from README
def get_long_description():
    """Read long description from README.md file."""
    readme_file = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_file):
        with open(readme_file, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

# Read requirements from requirements.txt
def get_requirements():
    """Read requirements from requirements.txt file."""
    requirements_file = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    requirements = []
    if os.path.exists(requirements_file):
        with open(requirements_file, 'r', encoding='utf-8') as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return requirements

setup(
    # Package metadata
    name='mariadb-vector-magics',
    version=get_version(),
    author='Jayant Acharya',
    author_email='jayant99acharya@gmail.com',
    description='IPython magic commands for MariaDB Vector operations',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    url='https://github.com/jayant99acharya/mariadb',
    project_urls={
        'Bug Reports': 'https://github.com/jayant99acharya/mariadb/issues',
        'Source': 'https://github.com/jayant99acharya/mariadb',
        'Documentation': 'https://github.com/jayant99acharya/mariadb/blob/main/README.md',
    },
    
    # Package configuration
    packages=find_packages(exclude=['tests*', 'docs*', 'examples*']),
    include_package_data=True,
    zip_safe=False,
    
    # Dependencies
    install_requires=get_requirements(),
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'black>=22.0.0',
            'flake8>=5.0.0',
            'mypy>=1.0.0',
        ],
        'docs': [
            'sphinx>=5.0.0',
            'sphinx-rtd-theme>=1.0.0',
            'myst-parser>=0.18.0',
        ],
        'rag': [
            'groq>=0.4.0',
        ],
    },
    
    # Python version requirement
    python_requires='>=3.8',
    
    # Package classification
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Database',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Framework :: IPython',
        'Framework :: Jupyter',
    ],
    
    # Keywords for package discovery
    keywords=[
        'mariadb',
        'vector',
        'database',
        'jupyter',
        'ipython',
        'magic',
        'ai',
        'ml',
        'rag',
        'semantic-search',
        'embeddings',
        'similarity-search',
    ],
    
    # Entry points for magic commands
    entry_points={
        'console_scripts': [
            'mariadb-vector-test=mariadb_vector_magics.cli:test_connection',
            'mariadb-vector-secrets=mariadb_vector_magics.cli:manage_secrets',
        ],
    },
    
    # Package data
    package_data={
        'mariadb_vector_magics': [
            'py.typed',
            '*.md',
        ],
    },
    
    # Test configuration
    test_suite='tests',
    tests_require=[
        'pytest>=7.0.0',
        'pytest-cov>=4.0.0',
    ],
)
