# Contributing to HardVAE

Thank you for your interest in contributing to HardVAE! This document provides guidelines and instructions for contributing.

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please be respectful and constructive in all interactions.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected behavior
- Actual behavior
- Your environment (Python version, OS, installed packages)

### Suggesting Enhancements

For feature requests or improvements:
- Clearly describe the enhancement
- Explain the motivation and use case
- Provide examples if applicable
- Discuss potential implementation approaches

### Pull Requests

1. **Fork the repository** and create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the code style guidelines below

3. **Write or update tests** for your changes

4. **Update documentation** as needed

5. **Commit with clear messages**:
   ```bash
   git commit -m "Add feature: description of changes"
   ```

6. **Push to your fork** and **create a Pull Request**

## Code Style Guidelines

### Python Style

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with these specific guidelines:

- **Line length**: Maximum 100 characters
- **Indentation**: 4 spaces
- **Imports**: Organize as stdlib, third-party, local
- **Naming**: 
  - Classes: `PascalCase`
  - Functions/variables: `snake_case`
  - Constants: `UPPER_CASE`

### Code Formatting

Use `black` for code formatting:
```bash
pip install black
black hardvae/
```

Check code style with `flake8`:
```bash
pip install flake8
flake8 hardvae/
```

### Type Hints

Use type hints for function signatures:
```python
def calculate_hardness(X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Calculate hardness scores."""
    pass
```

### Docstrings

Use Google-style docstrings:
```python
def function_name(param1: str, param2: int) -> bool:
    """
    Brief description of function.
    
    Longer description if needed, explaining the purpose,
    behavior, and any important details.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: When param1 is invalid
        TypeError: When param2 is not an integer
    
    Examples:
        >>> result = function_name("test", 42)
        >>> print(result)
        True
    """
    pass
```

## Testing

### Writing Tests

- Place tests in the `tests/` directory
- Use `pytest` framework
- Test file naming: `test_*.py` or `*_test.py`
- Test function naming: `test_*`

Example test:
```python
import pytest
from hardvae.core import HardnessCalculator

def test_hardness_calculation():
    """Test basic hardness calculation."""
    calc = HardnessCalculator(random_state=42)
    X = np.random.randn(100, 10)
    y = np.ones(100)
    
    scores = calc.calculate_hardness_scores(X, y, metrics=['feature_kDN'])
    
    assert scores is not None
    assert len(scores) == 100
    assert not scores.isnull().any().any()
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_hardness.py

# Run with coverage
pytest --cov=hardvae tests/

# Run with verbose output
pytest -v
```

## Documentation

### Updating Documentation

- Update docstrings when changing function signatures
- Update README.md for major changes
- Add examples to `examples/` directory for new features
- Update API reference in `docs/api_reference.md` (if  created )

### Building Documentation

```bash
# Install sphinx
pip install sphinx sphinx-rtd-theme

# Build HTML docs
cd docs
make html
```

## Commit Messages

Use clear, descriptive commit messages:

```
[Type] Brief description (50 chars max)

Longer explanation if needed (72 chars per line).
Explain what changed and why.

- Bullet points for multiple changes
- Keep it concise and informative

Closes #123
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (no logic change)
- `refactor`: Code refactoring
- `test`: Test additions/changes
- `chore`: Build, dependencies, etc.

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Bouka12/HardVAE.git
   cd HardVAE
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in development mode**:
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks** (optional):
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Release Process

Maintainers follow this process for releases:

1. Update version in `setup.py` and `hardvae/__init__.py`
2. Update CHANGELOG.md
3. Create git tag: `git tag v1.0.0`
4. Push tag: `git push origin v1.0.0`
5. Build and upload to PyPI

## Questions?

- Check `original_code/` for the original working code
- Check existing issues and discussions
- Read the documentation in `docs/`
- Look at examples in `examples/`
- Open a discussion for questions

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- GitHub contributors page
- Release notes for significant contributions

Thank you for contributing to HardVAE!
