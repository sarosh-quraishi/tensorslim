"""
Utility modules for TensorSlim.
"""

from .metrics import (
    cosine_similarity,
    mse_loss,
    relative_error,
    frobenius_distance,
    spectral_distance,
    CompressionMetrics,
    evaluate_compression_quality,
    create_compression_report
)

from .io import (
    ModelSaver,
    ConfigurationManager,
    ExperimentLogger,
    export_model_summary,
    create_compression_report_file
)

__all__ = [
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
    'create_compression_report_file'
]