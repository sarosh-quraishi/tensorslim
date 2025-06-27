"""
HuggingFace Transformers integration for TensorSlim.

This module provides seamless integration with HuggingFace models,
allowing direct compression of popular transformer models.
"""

from typing import Dict, Optional, Union, Any, List, Tuple
import logging
import warnings

try:
    import transformers
    from transformers import (
        AutoModel, AutoTokenizer, AutoConfig,
        PreTrainedModel, PreTrainedTokenizer
    )
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    AutoModel = None
    AutoTokenizer = None
    AutoConfig = None
    PreTrainedModel = None
    PreTrainedTokenizer = None

import torch
import torch.nn as nn

from ..models import TransformerSlim, ModelProfiler
from ..core import compress_model, TensorSlim

logger = logging.getLogger(__name__)


def check_huggingface_availability():
    """Check if HuggingFace transformers is available."""
    if not HF_AVAILABLE:
        raise ImportError(
            "HuggingFace transformers not available. Install with: "
            "pip install transformers or uv add tensorslim[huggingface]"
        )


class HuggingFaceCompressor:
    """
    Specialized compressor for HuggingFace transformer models.
    
    This class provides optimized compression for popular HuggingFace models
    with architecture-specific optimizations and quality preservation.
    
    Args:
        model_name: HuggingFace model name or path
        compression_ratio: Target compression ratio
        preserve_embeddings: Whether to preserve embedding layers
        preserve_layer_norm: Whether to preserve layer normalization
        quality_threshold: Minimum quality to maintain
        device: Device for computation
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        compression_ratio: Union[int, float] = 0.5,
        preserve_embeddings: bool = True,
        preserve_layer_norm: bool = True,
        quality_threshold: float = 0.95,
        device: Optional[Union[str, torch.device]] = None
    ):
        check_huggingface_availability()
        
        self.model_name = model_name
        self.compression_ratio = compression_ratio
        self.preserve_embeddings = preserve_embeddings
        self.preserve_layer_norm = preserve_layer_norm
        self.quality_threshold = quality_threshold
        self.device = device
        
        # Model-specific configurations
        self.model_configs = self._get_model_configs()
        
    def compress_from_pretrained(
        self,
        model_name: Optional[str] = None,
        **model_kwargs
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer, Dict[str, Any]]:
        """
        Load and compress a HuggingFace model from pretrained.
        
        Args:
            model_name: HuggingFace model name (overrides instance default)
            **model_kwargs: Additional arguments for model loading
            
        Returns:
            Tuple of (compressed_model, tokenizer, compression_info)
        """
        model_name = model_name or self.model_name
        if not model_name:
            raise ValueError("model_name must be provided")
        
        logger.info(f"Loading model: {model_name}")
        
        # Load model and tokenizer
        model = AutoModel.from_pretrained(model_name, **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Move to device if specified
        if self.device is not None:
            model = model.to(self.device)
        
        # Compress model
        compressed_model, compression_info = self.compress_model(model)
        
        return compressed_model, tokenizer, compression_info
    
    def compress_model(
        self,
        model: PreTrainedModel
    ) -> Tuple[PreTrainedModel, Dict[str, Any]]:
        """
        Compress a loaded HuggingFace model.
        
        Args:
            model: HuggingFace model to compress
            
        Returns:
            Tuple of (compressed_model, compression_info)
        """
        # Get model-specific configuration
        model_type = getattr(model.config, 'model_type', 'unknown')
        config = self.model_configs.get(model_type, self.model_configs['default'])
        
        logger.info(f"Compressing {model_type} model with architecture-specific optimizations")
        
        # Profile original model
        original_profile = ModelProfiler.profile_model(model)
        
        # Create transformer-specific compressor
        compressor = TransformerSlim(
            attention_rank=self._get_attention_rank(config),
            ffn_rank=self._get_ffn_rank(config),
            output_rank=self._get_output_rank(config),
            preserve_embeddings=self.preserve_embeddings,
            preserve_layernorm=self.preserve_layer_norm,
            quality_threshold=self.quality_threshold,
            device=self.device
        )
        
        # Compress model
        compressed_model = compressor.compress(model, inplace=False)
        
        # Profile compressed model
        compressed_profile = ModelProfiler.profile_model(compressed_model)
        
        # Prepare compression info
        compression_info = {
            'model_type': model_type,
            'compression_method': 'transformer_slim',
            'original_profile': original_profile,
            'compressed_profile': compressed_profile,
            'compression_stats': compressor.compression_stats,
            'parameter_reduction': 1 - (compressed_profile['total_parameters'] / original_profile['total_parameters']),
            'size_reduction_mb': original_profile['model_size_mb'] - compressed_profile['model_size_mb']
        }
        
        logger.info(f"Compression complete: {compression_info['parameter_reduction']*100:.1f}% parameter reduction")
        
        return compressed_model, compression_info
    
    def _get_model_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get model-specific compression configurations."""
        return {
            'bert': {
                'attention_rank_ratio': 0.6,
                'ffn_rank_ratio': 0.4,
                'output_rank_ratio': 0.8,
                'preserve_cls': True
            },
            'gpt2': {
                'attention_rank_ratio': 0.5,
                'ffn_rank_ratio': 0.3,
                'output_rank_ratio': 0.7,  
                'preserve_lm_head': True
            },
            'roberta': {
                'attention_rank_ratio': 0.6,
                'ffn_rank_ratio': 0.4,
                'output_rank_ratio': 0.8,
                'preserve_cls': True
            },
            't5': {
                'attention_rank_ratio': 0.5,
                'ffn_rank_ratio': 0.3,
                'output_rank_ratio': 0.6,
                'encoder_decoder_balance': True
            },
            'distilbert': {
                'attention_rank_ratio': 0.7,
                'ffn_rank_ratio': 0.5,
                'output_rank_ratio': 0.8,
                'preserve_distillation': True
            },
            'default': {
                'attention_rank_ratio': 0.5,
                'ffn_rank_ratio': 0.4,
                'output_rank_ratio': 0.7
            }
        }
    
    def _get_attention_rank(self, config: Dict[str, Any]) -> Union[int, float]:
        """Calculate attention rank based on compression ratio and model config."""
        ratio = config.get('attention_rank_ratio', 0.5)
        if isinstance(self.compression_ratio, float):
            return self.compression_ratio * ratio
        else:
            return int(self.compression_ratio * ratio)
    
    def _get_ffn_rank(self, config: Dict[str, Any]) -> Union[int, float]:
        """Calculate FFN rank based on compression ratio and model config."""
        ratio = config.get('ffn_rank_ratio', 0.4)
        if isinstance(self.compression_ratio, float):
            return self.compression_ratio * ratio
        else:
            return int(self.compression_ratio * ratio)
    
    def _get_output_rank(self, config: Dict[str, Any]) -> Union[int, float]:
        """Calculate output rank based on compression ratio and model config."""
        ratio = config.get('output_rank_ratio', 0.7)
        if isinstance(self.compression_ratio, float):
            return self.compression_ratio * ratio
        else:
            return int(self.compression_ratio * ratio)


def compress_huggingface_model(
    model_name: str,
    compression_ratio: Union[int, float] = 0.5,
    quality_threshold: float = 0.95,
    preserve_embeddings: bool = True,
    device: Optional[Union[str, torch.device]] = None,
    **model_kwargs
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Convenience function to compress HuggingFace models.
    
    Args:
        model_name: HuggingFace model name or path
        compression_ratio: Target compression ratio
        quality_threshold: Minimum quality threshold
        preserve_embeddings: Whether to preserve embeddings
        device: Device for computation
        **model_kwargs: Additional model loading arguments
        
    Returns:
        Tuple of (compressed_model, tokenizer)
    """
    compressor = HuggingFaceCompressor(
        model_name=model_name,
        compression_ratio=compression_ratio,
        quality_threshold=quality_threshold,
        preserve_embeddings=preserve_embeddings,
        device=device
    )
    
    compressed_model, tokenizer, _ = compressor.compress_from_pretrained(**model_kwargs)
    return compressed_model, tokenizer


def analyze_huggingface_model(
    model_name: str,
    compression_ratios: Optional[List[float]] = None,
    device: Optional[Union[str, torch.device]] = None
) -> Dict[str, Any]:
    """
    Analyze compression potential for a HuggingFace model.
    
    Args:
        model_name: HuggingFace model name
        compression_ratios: List of ratios to analyze
        device: Device for computation
        
    Returns:
        Analysis results dictionary
    """
    check_huggingface_availability()
    
    if compression_ratios is None:
        compression_ratios = [0.1, 0.25, 0.5, 0.75]
    
    # Load model
    model = AutoModel.from_pretrained(model_name)
    if device is not None:
        model = model.to(device)
    
    # Profile original model
    original_profile = ModelProfiler.profile_model(model)
    
    results = {
        'model_name': model_name,
        'model_type': getattr(model.config, 'model_type', 'unknown'),
        'original_profile': original_profile,
        'compression_analysis': {}
    }
    
    # Test different compression ratios
    for ratio in compression_ratios:
        logger.info(f"Analyzing compression ratio: {ratio}")
        
        compressor = HuggingFaceCompressor(
            model_name=model_name,
            compression_ratio=ratio,
            device=device
        )
        
        try:
            compressed_model, compression_info = compressor.compress_model(model)
            
            results['compression_analysis'][f'ratio_{ratio}'] = {
                'parameter_reduction': compression_info['parameter_reduction'],
                'size_reduction_mb': compression_info['size_reduction_mb'],
                'layers_compressed': len(compression_info['compression_stats']),
                'success': True
            }
            
        except Exception as e:
            logger.warning(f"Failed to analyze ratio {ratio}: {e}")
            results['compression_analysis'][f'ratio_{ratio}'] = {
                'success': False,
                'error': str(e)
            }
    
    return results


class HuggingFaceModelWrapper:
    """
    Wrapper for compressed HuggingFace models to maintain API compatibility.
    
    This wrapper ensures that compressed models can be used as drop-in
    replacements for original HuggingFace models.
    """
    
    def __init__(
        self,
        compressed_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        compression_info: Dict[str, Any]
    ):
        self.model = compressed_model
        self.tokenizer = tokenizer
        self.compression_info = compression_info
        
        # Expose model attributes
        self.config = compressed_model.config
        
    def __call__(self, *args, **kwargs):
        """Forward call to the compressed model."""
        return self.model(*args, **kwargs)
    
    def __getattr__(self, name):
        """Delegate attribute access to the compressed model."""
        return getattr(self.model, name)
    
    def save_compressed(
        self,
        save_directory: str,
        save_compression_info: bool = True
    ) -> None:
        """
        Save compressed model and tokenizer.
        
        Args:
            save_directory: Directory to save to
            save_compression_info: Whether to save compression metadata
        """
        # Save model and tokenizer using HuggingFace methods
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        
        # Save compression info
        if save_compression_info:
            import json
            import os
            
            info_path = os.path.join(save_directory, 'tensorslim_compression_info.json')
            with open(info_path, 'w') as f:
                # Make compression info JSON serializable
                serializable_info = self._make_serializable(self.compression_info)
                json.dump(serializable_info, f, indent=2)
        
        logger.info(f"Compressed model saved to: {save_directory}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert object to JSON serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj
    
    def get_compression_summary(self) -> str:
        """Get a human-readable compression summary."""
        info = self.compression_info
        
        summary = f"""
TensorSlim Compression Summary
============================
Model: {info.get('model_type', 'Unknown')}
Method: {info.get('compression_method', 'Unknown')}

Original Parameters: {info['original_profile']['total_parameters']:,}
Compressed Parameters: {info['compressed_profile']['total_parameters']:,}
Parameter Reduction: {info['parameter_reduction']*100:.1f}%

Original Size: {info['original_profile']['model_size_mb']:.1f} MB
Compressed Size: {info['compressed_profile']['model_size_mb']:.1f} MB
Size Reduction: {info['size_reduction_mb']:.1f} MB

Layers Compressed: {len(info['compression_stats'])}
"""
        return summary.strip()


def load_compressed_huggingface_model(
    model_path: str,
    load_compression_info: bool = True
) -> HuggingFaceModelWrapper:
    """
    Load a compressed HuggingFace model saved by TensorSlim.
    
    Args:
        model_path: Path to saved model directory
        load_compression_info: Whether to load compression metadata
        
    Returns:
        Wrapped compressed model
    """
    check_huggingface_availability()
    
    # Load model and tokenizer
    model = AutoModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load compression info
    compression_info = {}
    if load_compression_info:
        import json
        import os
        
        info_path = os.path.join(model_path, 'tensorslim_compression_info.json')
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                compression_info = json.load(f)
        else:
            logger.warning("Compression info not found, model may not be compressed")
    
    return HuggingFaceModelWrapper(model, tokenizer, compression_info)