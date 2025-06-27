"""
TensorSlim: Fast neural network compression using randomized SVD.

Make your models runway-ready with 10-100x faster compression and <2% quality loss.

Example usage:
    >>> import torch
    >>> from tensorslim import compress_model
    >>> 
    >>> # Simple compression
    >>> model = torch.load('my_model.pth')
    >>> compressed_model = compress_model(model, compression_ratio=0.5)
    >>> 
    >>> # HuggingFace integration
    >>> from tensorslim.integrations import compress_huggingface_model
    >>> compressed_bert, tokenizer = compress_huggingface_model("bert-base-uncased")
"""

__version__ = "0.1.0"
__author__ = "Sarosh Quraishi"
__email__ = "sarosh.quraishi@gmail.com"

# Core compression functionality
from .core import (
    # SVD algorithms
    RandomizedSVD,
    AdaptiveRandomizedSVD,
    randomized_svd,
    estimate_rank,
    
    # Model compression
    TensorSlim,
    ModelCompressor,
    compress_model,
    analyze_model_compression,
    
    # Compressed layers
    SlimLinear,
    SlimConv2d,
    SlimSeparableConv2d,
    SlimEmbedding,
    convert_layer_to_slim
)

# Model-specific compression
from .models import (
    TransformerSlim,
    AttentionSlim,
    BaseModelCompressor,
    LayerAnalyzer,
    CompressionStrategy,
    ModelProfiler
)

# Integration modules
from .integrations import (
    PyTorchCompressor,
    optimize_model_for_inference,
    analyze_model_layers,
    create_model_summary,
    ModelConverter,
    benchmark_model_inference,
    compare_models
)

# Utility functions
from .utils import (
    # Quality metrics
    cosine_similarity,
    mse_loss,
    relative_error,
    frobenius_distance,
    spectral_distance,
    CompressionMetrics,
    evaluate_compression_quality,
    create_compression_report,
    
    # I/O utilities
    ModelSaver,
    ConfigurationManager,
    ExperimentLogger,
    export_model_summary,
    create_compression_report_file
)

# Try to import HuggingFace integration (optional)
try:
    from .integrations import (
        HuggingFaceCompressor,
        compress_huggingface_model,
        analyze_huggingface_model,
        HuggingFaceModelWrapper,
        load_compressed_huggingface_model
    )
    
    _HF_AVAILABLE = True
    
    # Add HuggingFace imports to __all__
    _hf_exports = [
        'HuggingFaceCompressor',
        'compress_huggingface_model',
        'analyze_huggingface_model', 
        'HuggingFaceModelWrapper',
        'load_compressed_huggingface_model'
    ]
    
except ImportError:
    _HF_AVAILABLE = False
    _hf_exports = []

# Define public API
__all__ = [
    # Core compression
    'RandomizedSVD',
    'AdaptiveRandomizedSVD',
    'randomized_svd',
    'estimate_rank',
    'TensorSlim',
    'ModelCompressor', 
    'compress_model',
    'analyze_model_compression',
    
    # Compressed layers
    'SlimLinear',
    'SlimConv2d',
    'SlimSeparableConv2d',
    'SlimEmbedding',
    'convert_layer_to_slim',
    
    # Model-specific compression
    'TransformerSlim',
    'AttentionSlim',
    'BaseModelCompressor',
    'LayerAnalyzer',
    'CompressionStrategy',
    'ModelProfiler',
    
    # PyTorch integration
    'PyTorchCompressor',
    'optimize_model_for_inference',
    'analyze_model_layers',
    'create_model_summary',
    'ModelConverter',
    'benchmark_model_inference',
    'compare_models',
    
    # Quality metrics
    'cosine_similarity',
    'mse_loss',
    'relative_error',
    'frobenius_distance',
    'spectral_distance',
    'CompressionMetrics',
    'evaluate_compression_quality',
    'create_compression_report',
    
    # I/O utilities
    'ModelSaver',
    'ConfigurationManager',
    'ExperimentLogger',
    'export_model_summary',
    'create_compression_report_file',
    
] + _hf_exports

# Package metadata
__description__ = "Fast neural network compression using randomized SVD"
__url__ = "https://github.com/tensorslim/tensorslim"
__license__ = "MIT"
__keywords__ = ["machine-learning", "deep-learning", "model-compression", "svd", "pytorch", "transformers"]


def get_version() -> str:
    """Get TensorSlim version."""
    return __version__


def check_dependencies() -> dict:
    """
    Check availability of optional dependencies.
    
    Returns:
        Dictionary indicating which optional dependencies are available
    """
    dependencies = {
        'torch': False,
        'numpy': False,
        'transformers': False,
        'sklearn': False,
        'matplotlib': False
    }
    
    try:
        import torch
        dependencies['torch'] = True
    except ImportError:
        pass
    
    try:
        import numpy
        dependencies['numpy'] = True
    except ImportError:
        pass
    
    try:
        import transformers
        dependencies['transformers'] = True
    except ImportError:
        pass
    
    try:
        import sklearn
        dependencies['sklearn'] = True
    except ImportError:
        pass
    
    try:
        import matplotlib
        dependencies['matplotlib'] = True
    except ImportError:
        pass
    
    return dependencies


def print_info():
    """Print TensorSlim package information."""
    deps = check_dependencies()
    
    print(f"""
TensorSlim {__version__}
{__description__}

Authors: {__author__}
License: {__license__}
Homepage: {__url__}

Dependencies:
  ✓ torch: {'Available' if deps['torch'] else 'Not found'}
  ✓ numpy: {'Available' if deps['numpy'] else 'Not found'}
  {'✓' if deps['transformers'] else '○'} transformers: {'Available' if deps['transformers'] else 'Not installed (optional)'}
  {'✓' if deps['sklearn'] else '○'} scikit-learn: {'Available' if deps['sklearn'] else 'Not installed (optional)'}
  {'✓' if deps['matplotlib'] else '○'} matplotlib: {'Available' if deps['matplotlib'] else 'Not installed (optional)'}

Quick Start:
  >>> import tensorslim
  >>> compressed_model = tensorslim.compress_model(model, compression_ratio=0.5)

For more examples, visit: {__url__}
""")


# Configure logging
import logging

logger = logging.getLogger(__name__)

# Set up default logging configuration
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Welcome message for interactive sessions
import sys
if hasattr(sys, 'ps1'):  # Interactive session
    print(f"TensorSlim {__version__} - Make your models runway-ready!")
    print("Type 'tensorslim.print_info()' for more information.")