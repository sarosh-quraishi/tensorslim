"""
Integration modules for popular ML frameworks and libraries.
"""

from .pytorch import (
    PyTorchCompressor,
    optimize_model_for_inference,
    analyze_model_layers,
    create_model_summary,
    ModelConverter,
    benchmark_model_inference,
    compare_models
)

# HuggingFace integration (optional import)
try:
    from .huggingface import (
        HuggingFaceCompressor,
        compress_huggingface_model,
        analyze_huggingface_model,
        HuggingFaceModelWrapper,
        load_compressed_huggingface_model
    )
    
    __all__ = [
        # PyTorch integration
        'PyTorchCompressor',
        'optimize_model_for_inference',
        'analyze_model_layers',
        'create_model_summary',
        'ModelConverter',
        'benchmark_model_inference',
        'compare_models',
        
        # HuggingFace integration
        'HuggingFaceCompressor',
        'compress_huggingface_model', 
        'analyze_huggingface_model',
        'HuggingFaceModelWrapper',
        'load_compressed_huggingface_model'
    ]
    
except ImportError:
    # HuggingFace not available
    __all__ = [
        # PyTorch integration only
        'PyTorchCompressor',
        'optimize_model_for_inference', 
        'analyze_model_layers',
        'create_model_summary',
        'ModelConverter',
        'benchmark_model_inference',
        'compare_models'
    ]