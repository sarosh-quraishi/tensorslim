# Contributing to TensorSlim

Thank you for your interest in contributing to TensorSlim! We welcome contributions from the community and are excited to collaborate with you.

## 🎯 How to Contribute

There are many ways to contribute to TensorSlim:

- 🐛 **Report bugs** and suggest fixes
- 💡 **Propose new features** or improvements
- 📖 **Improve documentation** and examples
- 🧪 **Add tests** and improve test coverage
- 🔧 **Submit code** improvements and bug fixes
- 🚀 **Share your use cases** and success stories

## 🚀 Getting Started

### 1. Development Environment Setup

1. **Fork and clone** the repository:
   ```bash
   git clone https://github.com/your-username/tensorslim.git
   cd tensorslim
   ```

2. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Set up development environment**:
   ```bash
   # Create virtual environment and install dependencies
   uv sync --extra dev --extra all
   
   # Install pre-commit hooks
   uv run pre-commit install
   ```

4. **Run tests** to ensure everything works:
   ```bash
   uv run pytest tests/
   ```

### 2. Development Workflow

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

2. **Make your changes** following our coding standards

3. **Run tests and checks**:
   ```bash
   # Run tests
   uv run pytest tests/ -v
   
   # Run linting
   uv run flake8 src/tensorslim tests/
   
   # Run formatting
   uv run black src/tensorslim tests/
   uv run isort src/tensorslim tests/
   
   # Run type checking
   uv run mypy src/tensorslim
   ```

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   # or
   git commit -m "fix: fix your bug description"
   ```

5. **Push and create pull request**:
   ```bash
   git push origin feature/your-feature-name
   ```

## 📝 Contribution Guidelines

### Code Standards

- **Python version**: Support Python 3.8+
- **Code style**: We use Black for formatting with 88 character line length
- **Import sorting**: We use isort with Black profile
- **Type hints**: All public functions should have type hints
- **Docstrings**: All public functions and classes must have docstrings
- **Testing**: New features must include tests

### Commit Message Format

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```bash
feat: add support for quantized models
fix: resolve memory leak in SVD computation
docs: update README with new examples
test: add integration tests for HuggingFace models
```

### Code Review Process

1. All contributions require **code review**
2. Ensure **CI checks pass** (tests, linting, type checking)
3. Maintain **test coverage** above 90%
4. Update **documentation** if needed
5. Get **approval from maintainers**

## 🧪 Testing Guidelines

### Running Tests

```bash
# Run all tests
uv run pytest tests/

# Run specific test file
uv run pytest tests/test_core.py

# Run with coverage
uv run pytest tests/ --cov=tensorslim --cov-report=html

# Run only fast tests (skip slow benchmarks)
uv run pytest tests/ -m "not slow"
```

### Writing Tests

- Place tests in the `tests/` directory
- Use descriptive test function names
- Include both positive and negative test cases
- Test edge cases and error conditions
- Use fixtures for common test setup

**Example test structure:**
```python
def test_compress_model_basic():
    """Test basic model compression functionality."""
    # Arrange
    model = create_test_model()
    
    # Act
    compressed = compress_model(model, compression_ratio=0.5)
    
    # Assert
    assert compressed is not None
    original_params = sum(p.numel() for p in model.parameters())
    compressed_params = sum(p.numel() for p in compressed.parameters())
    assert compressed_params < original_params
```

### Test Categories

- **Unit tests**: Test individual functions and classes
- **Integration tests**: Test component interactions
- **End-to-end tests**: Test complete workflows
- **Performance tests**: Benchmark and regression tests

## 📖 Documentation

### Documentation Types

- **API Documentation**: Docstrings for all public APIs
- **User Guides**: High-level usage documentation
- **Examples**: Practical usage examples
- **Tutorials**: Step-by-step learning materials

### Writing Documentation

- Use clear, concise language
- Include code examples
- Test all code examples
- Update relevant documentation when making changes

### Docstring Format

We use Google-style docstrings:

```python
def compress_model(
    model: nn.Module,
    compression_ratio: float = 0.5,
    quality_threshold: float = 0.95
) -> nn.Module:
    """
    Compress a PyTorch model using randomized SVD.
    
    Args:
        model: PyTorch model to compress
        compression_ratio: Target compression ratio (0-1)
        quality_threshold: Minimum quality to maintain
        
    Returns:
        Compressed model with reduced parameters
        
    Raises:
        ValueError: If compression_ratio is not between 0 and 1
        
    Example:
        >>> model = torch.nn.Linear(100, 50)
        >>> compressed = compress_model(model, compression_ratio=0.5)
        >>> print(f"Compression: {model.weight.numel() / compressed.weight.numel():.1f}x")
    """
```

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Environment information**:
   - Python version
   - TensorSlim version
   - PyTorch version
   - Operating system

2. **Bug description**:
   - What you expected to happen
   - What actually happened
   - Steps to reproduce

3. **Minimal code example** that reproduces the issue

4. **Error messages** and stack traces

**Use our bug report template:**

```markdown
## Bug Description
A clear description of what the bug is.

## Environment
- Python: 3.x.x
- TensorSlim: x.x.x
- PyTorch: x.x.x
- OS: Ubuntu/Windows/macOS

## Steps to Reproduce
1. Step one
2. Step two
3. See error

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened.

## Code Example
```python
# Minimal code that reproduces the issue
```

## Error Message
```
Full error message and stack trace
```
```

## 💡 Feature Requests

When proposing features:

1. **Check existing issues** to avoid duplicates
2. **Describe the motivation** and use case
3. **Propose an API design** if applicable
4. **Consider backwards compatibility**
5. **Discuss implementation complexity**

## 🏆 Recognition

Contributors are recognized in several ways:

- **README acknowledgments** for significant contributions
- **Release notes** mention notable contributors
- **GitHub badges** for various contribution types
- **Community highlights** on social media

## 📞 Getting Help

If you need help contributing:

- 💬 **Discord**: Join our [community server](https://discord.gg/tensorslim)
- 🐛 **GitHub Issues**: Ask questions in issues
- 📧 **Email**: Contact maintainers at [contributors@tensorslim.ai](mailto:contributors@tensorslim.ai)

## 🎉 First-Time Contributors

Welcome! Here are some good first issues:

- **Documentation improvements**: Fix typos, improve examples
- **Test coverage**: Add tests for existing functionality
- **Small bug fixes**: Fix minor issues in the codebase
- **Example scripts**: Create new usage examples

Look for issues labeled `good first issue` or `help wanted`.

## 📋 Pull Request Checklist

Before submitting a pull request:

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] New code is covered by tests
- [ ] Documentation is updated
- [ ] Commit messages follow convention
- [ ] No merge conflicts
- [ ] PR description explains changes

## 🛡️ Security

To report security vulnerabilities:

- **Do not** open public issues
- Email security concerns to [security@tensorslim.ai](mailto:security@tensorslim.ai)
- Include detailed reproduction steps
- Allow time for response before disclosure

## 📜 Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

### Our Standards

- **Be respectful** and inclusive
- **Be collaborative** and constructive
- **Be patient** with newcomers
- **Be professional** in all interactions

## 🙏 Thank You

Thank you for contributing to TensorSlim! Your efforts help make neural network compression accessible to everyone.

---

*Happy coding! 🚀*