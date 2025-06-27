"""
Test suite for TensorSlim.

This package contains comprehensive tests for all TensorSlim functionality
including core algorithms, model compression, integrations, and utilities.

Test modules:
- test_core: Core compression algorithms and SVD functionality
- test_models: Model-specific compression and analysis
- test_integrations: PyTorch and HuggingFace integration tests
- test_utils: Utility functions and quality metrics

Run tests with:
    pytest tests/
    
Run specific test modules:
    pytest tests/test_core.py
    pytest tests/test_models.py
    pytest tests/test_integrations.py
    
Run with coverage:
    pytest --cov=tensorslim tests/
"""

import pytest
import sys
from pathlib import Path

# Add src directory to path for testing
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Test configuration
pytest_plugins = []

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "huggingface: marks tests that require HuggingFace transformers"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    import pytest
    
    for item in items:
        # Mark slow tests
        if "slow" in item.nodeid or "benchmark" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        
        # Mark GPU tests
        if "gpu" in item.nodeid or "cuda" in item.nodeid:
            item.add_marker(pytest.mark.gpu)
        
        # Mark integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Mark HuggingFace tests
        if "huggingface" in item.nodeid or "hf" in item.nodeid:
            item.add_marker(pytest.mark.huggingface)