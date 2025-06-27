"""
Core compression algorithms and layers for TensorSlim.
"""

from .randomized_svd import (
    RandomizedSVD,
    AdaptiveRandomizedSVD, 
    randomized_svd,
    estimate_rank
)

from .compression import (
    TensorSlim,
    ModelCompressor,
    compress_model,
    analyze_model_compression
)

from .layers import (
    SlimLinear,
    SlimConv2d,
    SlimSeparableConv2d,
    SlimEmbedding,
    convert_layer_to_slim
)

__all__ = [
    # SVD algorithms
    'RandomizedSVD',
    'AdaptiveRandomizedSVD',
    'randomized_svd',
    'estimate_rank',
    
    # Model compression
    'TensorSlim',
    'ModelCompressor',
    'compress_model',
    'analyze_model_compression',
    
    # Compressed layers
    'SlimLinear',
    'SlimConv2d', 
    'SlimSeparableConv2d',
    'SlimEmbedding',
    'convert_layer_to_slim'
]