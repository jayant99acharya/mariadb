# Contributing to MariaDB Vector Magics

We welcome contributions to MariaDB Vector Magics! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Code Style](#code-style)
- [Documentation](#documentation)

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to jayant99acharya@gmail.com.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up the development environment
4. Create a branch for your changes
5. Make your changes
6. Test your changes
7. Submit a pull request

## Development Setup

### Prerequisites

- Python 3.8 or higher
- MariaDB 11.7+ with Vector support
- Git

### Setup Instructions

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/mariadb.git
cd mariadb

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests to verify setup
pytest tests/ -v
```

### Database Setup

For testing, you'll need a MariaDB instance with Vector support:

```bash
# Using Docker
docker run -d \
  --name mariadb-vector-test \
  -e MYSQL_ROOT_PASSWORD=test_password \
  -e MYSQL_DATABASE=test_vectordb \
  -p 3306:3306 \
  mariadb:11.7

# Test connection
mariadb-vector-test --password=test_password --database=test_vectordb
```

## Making Changes

### Branch Naming

Use descriptive branch names:
- `feature/add-new-magic-command`
- `bugfix/fix-connection-error`
- `docs/update-readme`
- `test/add-integration-tests`

### Commit Messages

Follow conventional commit format:
```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `style`: Code style changes
- `chore`: Maintenance tasks

Examples:
```
feat(magics): add support for custom distance functions
fix(connection): handle connection timeout gracefully
docs(readme): update installation instructions
test(magics): add unit tests for semantic search
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=mariadb_vector_magics

# Run specific test file
pytest tests/test_magics.py

# Run tests with specific markers
pytest -m unit
pytest -m integration
```

### Test Categories

- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Test complete workflows

### Writing Tests

- Place tests in the `tests/` directory
- Use descriptive test names
- Follow the AAA pattern (Arrange, Act, Assert)
- Mock external dependencies
- Test both success and failure cases

Example:
```python
def test_connect_mariadb_success(self, mock_mariadb, magic_instance):
    """Test successful MariaDB connection."""
    # Arrange
    mock_connection = Mock()
    mock_mariadb.connect.return_value = mock_connection
    
    # Act
    result = magic_instance.connect_mariadb('--password=test')
    
    # Assert
    assert result == "Connected successfully"
    mock_mariadb.connect.assert_called_once()
```

## Submitting Changes

### Pull Request Process

1. **Update Documentation**: Ensure documentation reflects your changes
2. **Add Tests**: Include tests for new functionality
3. **Update Changelog**: Add entry to CHANGELOG.md
4. **Check CI**: Ensure all CI checks pass
5. **Request Review**: Tag relevant maintainers

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Other (please describe)

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests pass locally
- [ ] No breaking changes (or documented)
```

## Code Style

### Python Style Guide

We follow PEP 8 with some modifications:
- Line length: 100 characters
- Use Black for formatting
- Use isort for import sorting
- Use type hints where appropriate

### Pre-commit Hooks

Pre-commit hooks automatically check:
- Code formatting (Black)
- Import sorting (isort)
- Linting (flake8)
- Type checking (mypy)
- Security scanning (bandit)

### Code Quality Tools

```bash
# Format code
black mariadb_vector_magics/

# Sort imports
isort mariadb_vector_magics/

# Lint code
flake8 mariadb_vector_magics/

# Type checking
mypy mariadb_vector_magics/

# Security scan
bandit -r mariadb_vector_magics/
```

## Documentation

### Documentation Standards

- Use Google-style docstrings
- Include type hints
- Provide examples for public APIs
- Update README.md for user-facing changes
- Add docstrings for all public functions/classes

### Docstring Example

```python
def semantic_search(self, query: str, table: str, top_k: int = 5) -> List[Tuple]:
    """
    Perform semantic search on vector table.
    
    Args:
        query: Search query text
        table: Table name to search
        top_k: Number of results to return
        
    Returns:
        List of tuples containing (id, text, metadata, distance)
        
    Raises:
        MariaDBVectorError: If search fails
        
    Example:
        >>> results = magic.semantic_search("AI technology", "documents", 3)
        >>> print(f"Found {len(results)} results")
    """
```

### Building Documentation

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation
cd docs/
make html

# Serve documentation locally
python -m http.server 8000 -d _build/html/
```

## Issue Guidelines

### Reporting Bugs

Include:
- Python version
- MariaDB version
- Package version
- Minimal reproduction example
- Error messages/stack traces
- Expected vs actual behavior

### Feature Requests

Include:
- Use case description
- Proposed API/interface
- Implementation considerations
- Backward compatibility impact

## Release Process

Releases are handled by maintainers:

1. Update version in `__init__.py`
2. Update CHANGELOG.md
3. Create release tag
4. GitHub Actions handles PyPI publication

## Getting Help

- **Documentation**: Check README.md and code comments
- **Issues**: Search existing issues before creating new ones
- **Discussions**: Use GitHub Discussions for questions
- **Email**: Contact jayant99acharya@gmail.com for sensitive issues

## Recognition

Contributors are recognized in:
- CONTRIBUTORS.md file
- Release notes
- GitHub contributors page

Thank you for contributing to MariaDB Vector Magics!