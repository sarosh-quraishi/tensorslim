"""
High-level model compression using randomized SVD.

This module provides the main compression interface for PyTorch models,
handling layer detection, compression, and reconstruction automatically.
"""

from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import torch
import torch.nn as nn
from torch import Tensor
import copy
import logging
from tqdm import tqdm

from .randomized_svd import RandomizedSVD, AdaptiveRandomizedSVD, estimate_rank
from .layers import SlimLinear, SlimConv2d

logger = logging.getLogger(__name__)


class TensorSlim:
    """
    Main model compression class using randomized SVD.
    
    This class handles automatic detection of compressible layers, applies
    randomized SVD compression, and replaces layers with efficient equivalents.
    
    Args:
        rank: Target rank for compression (if int) or compression ratio (if float 0-1)
        method: Compression method ('randomized_svd' or 'adaptive')
        target_layers: Layer types to compress (default: linear and conv layers)
        preserve_layers: Layer names/types to preserve without compression
        quality_threshold: Minimum quality to maintain (0-1)
        device: Device for computation
        progress_bar: Show progress during compression
    """
    
    def __init__(
        self,
        rank: Union[int, float] = 0.5,
        method: str = "randomized_svd",
        target_layers: Optional[List[str]] = None,
        preserve_layers: Optional[List[str]] = None,
        quality_threshold: float = 0.95,
        device: Optional[Union[str, torch.device]] = None,
        progress_bar: bool = True,
        **svd_kwargs
    ):
        self.rank = rank
        self.method = method
        self.target_layers = target_layers or ['Linear', 'Conv2d']
        self.preserve_layers = preserve_layers or []
        self.quality_threshold = quality_threshold
        self.device = device
        self.progress_bar = progress_bar
        self.svd_kwargs = svd_kwargs
        
        # Compression statistics
        self.compression_stats = {}
        
    def compress(
        self, 
        model: nn.Module, 
        inplace: bool = False
    ) -> nn.Module:
        """
        Compress a PyTorch model using randomized SVD.
        
        Args:
            model: PyTorch model to compress
            inplace: Modify model in-place (default: False)
            
        Returns:
            Compressed model
        """
        if not inplace:
            model = copy.deepcopy(model)
            
        # Move to target device
        if self.device is not None:
            model = model.to(self.device)
            
        # Find compressible layers
        compressible_layers = self._find_compressible_layers(model)
        
        if not compressible_layers:
            logger.warning("No compressible layers found in model")
            return model
            
        logger.info(f"Found {len(compressible_layers)} compressible layers")
        
        # Compress layers
        self.compression_stats = {}
        
        pbar = None
        if self.progress_bar:
            pbar = tqdm(
                compressible_layers, 
                desc="Compressing layers",
                unit="layer"
            )
            
        for layer_name, layer in (pbar or compressible_layers):
            try:
                compressed_layer = self._compress_layer(layer, layer_name)
                self._replace_layer(model, layer_name, compressed_layer)
                
                if pbar:
                    pbar.set_postfix({
                        'current': layer_name.split('.')[-1],
                        'compression': f"{self.compression_stats.get(layer_name, {}).get('compression_ratio', 0):.1f}x"
                    })
                    
            except Exception as e:
                logger.warning(f"Failed to compress layer {layer_name}: {e}")
                continue
                
        if pbar:
            pbar.close()
            
        # Log compression summary
        self._log_compression_summary()
        
        return model
    
    def _find_compressible_layers(
        self, 
        model: nn.Module
    ) -> List[Tuple[str, nn.Module]]:
        """Find layers that can be compressed."""
        compressible = []
        
        for name, module in model.named_modules():
            # Skip if in preserve list
            if any(preserve in name for preserve in self.preserve_layers):
                continue
                
            # Check if layer type is target for compression  
            layer_type = type(module).__name__
            if layer_type in self.target_layers:
                compressible.append((name, module))
                
        return compressible
    
    def _compress_layer(
        self, 
        layer: nn.Module, 
        layer_name: str
    ) -> nn.Module:
        """Compress a single layer using SVD."""
        layer_type = type(layer).__name__
        
        if layer_type == 'Linear':
            return self._compress_linear(layer, layer_name)
        elif layer_type == 'Conv2d':
            return self._compress_conv2d(layer, layer_name)
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")
    
    def _compress_linear(
        self, 
        layer: nn.Linear, 
        layer_name: str
    ) -> SlimLinear:
        """Compress a linear layer."""
        weight = layer.weight.data
        bias = layer.bias.data if layer.bias is not None else None
        
        # Determine compression rank
        rank = self._get_compression_rank(weight.shape)
        
        # Create SVD compressor
        svd = self._create_svd_compressor(rank)
        
        # Compress weight matrix
        U, s, Vt = svd.fit_transform(weight)
        
        # Calculate compression metrics
        original_params = weight.numel() + (bias.numel() if bias is not None else 0)
        compressed_params = U.numel() + s.numel() + Vt.numel() + (bias.numel() if bias is not None else 0)
        compression_ratio = original_params / compressed_params
        
        # Estimate quality loss
        reconstructed = svd.reconstruct(U, s, Vt)
        quality_loss = svd.relative_error(weight, reconstructed)
        
        # Store statistics
        self.compression_stats[layer_name] = {
            'original_params': original_params,
            'compressed_params': compressed_params,
            'compression_ratio': compression_ratio,
            'quality_loss': quality_loss,
            'rank': rank
        }
        
        # Create compressed layer
        return SlimLinear(
            U=U,
            s=s,
            Vt=Vt,
            bias=bias,
            original_shape=weight.shape
        )
    
    def _compress_conv2d(
        self, 
        layer: nn.Conv2d, 
        layer_name: str
    ) -> SlimConv2d:
        """Compress a 2D convolutional layer."""
        weight = layer.weight.data  # (out_channels, in_channels, kernel_h, kernel_w)
        bias = layer.bias.data if layer.bias is not None else None
        
        # Reshape weight for SVD: (out_channels, in_channels * kernel_h * kernel_w)
        original_shape = weight.shape
        out_channels, in_channels, kernel_h, kernel_w = original_shape
        
        weight_2d = weight.view(out_channels, in_channels * kernel_h * kernel_w)
        
        # Determine compression rank
        rank = self._get_compression_rank(weight_2d.shape)
        
        # Create SVD compressor
        svd = self._create_svd_compressor(rank)
        
        # Compress weight matrix
        U, s, Vt = svd.fit_transform(weight_2d)
        
        # Calculate compression metrics
        original_params = weight.numel() + (bias.numel() if bias is not None else 0)
        compressed_params = U.numel() + s.numel() + Vt.numel() + (bias.numel() if bias is not None else 0)
        compression_ratio = original_params / compressed_params
        
        # Estimate quality loss
        reconstructed = svd.reconstruct(U, s, Vt)
        quality_loss = svd.relative_error(weight_2d, reconstructed)
        
        # Store statistics
        self.compression_stats[layer_name] = {
            'original_params': original_params,
            'compressed_params': compressed_params,
            'compression_ratio': compression_ratio,
            'quality_loss': quality_loss,
            'rank': rank
        }
        
        # Create compressed layer
        return SlimConv2d(
            U=U,
            s=s,
            Vt=Vt,
            bias=bias,
            original_shape=original_shape,
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=layer.groups,
            padding_mode=layer.padding_mode
        )
    
    def _get_compression_rank(self, weight_shape: Tuple[int, ...]) -> int:
        """Determine compression rank based on configuration."""
        if isinstance(self.rank, int):
            return min(self.rank, min(weight_shape))
        elif isinstance(self.rank, float):
            # Float rank can be interpreted as compression ratio (0-1)
            # If rank < 1, treat as compression ratio (fraction of original rank)
            # If rank >= 1, treat as direct compression factor
            m, n = weight_shape
            max_rank = min(m, n)
            
            if self.rank < 1.0:
                # For high-quality compression, use much more conservative ratios
                # Map compression ratios to maintain <2% quality loss
                if self.rank >= 0.95:
                    # Very conservative: maintain ~98% of max rank for <2% quality loss
                    estimated_rank = int(max_rank * 0.98)
                elif self.rank >= 0.9:
                    # Conservative: maintain ~95% of max rank
                    estimated_rank = int(max_rank * 0.95)
                elif self.rank >= 0.8:
                    # Moderate: maintain ~90% of max rank
                    estimated_rank = int(max_rank * 0.90)
                else:
                    # Aggressive: but still keep at least 80% for reasonable quality
                    estimated_rank = int(max_rank * 0.80)
            else:
                # Direct compression factor: solve for rank that achieves this compression
                original_size = m * n
                # Solve: original_size / (rank * (m + n + 1)) = compression_factor
                # rank = original_size / (compression_factor * (m + n + 1))
                estimated_rank = int(original_size / (self.rank * (m + n + 1)))
            
            return max(1, min(estimated_rank, max_rank))
        else:
            raise ValueError(f"Invalid rank type: {type(self.rank)}")
    
    def _create_svd_compressor(self, rank: int) -> Union[RandomizedSVD, AdaptiveRandomizedSVD]:
        """Create appropriate SVD compressor."""
        if self.method == "adaptive":
            return AdaptiveRandomizedSVD(
                rank=rank,
                target_quality=self.quality_threshold,
                device=self.device,
                **self.svd_kwargs
            )
        else:
            return RandomizedSVD(
                rank=rank,
                device=self.device,
                **self.svd_kwargs
            )
    
    def _replace_layer(
        self, 
        model: nn.Module, 
        layer_name: str, 
        new_layer: nn.Module
    ) -> None:
        """Replace a layer in the model with a new layer."""
        # Navigate to parent module
        parts = layer_name.split('.')
        parent = model
        
        for part in parts[:-1]:
            parent = getattr(parent, part)
            
        # Replace the layer
        setattr(parent, parts[-1], new_layer)
    
    def _log_compression_summary(self) -> None:
        """Log compression statistics summary."""
        if not self.compression_stats:
            return
            
        total_original = sum(stats['original_params'] for stats in self.compression_stats.values())
        total_compressed = sum(stats['compressed_params'] for stats in self.compression_stats.values())
        overall_ratio = total_original / total_compressed if total_compressed > 0 else 0
        
        avg_quality_loss = sum(stats['quality_loss'] for stats in self.compression_stats.values()) / len(self.compression_stats)
        
        logger.info(f"Compression Summary:")
        logger.info(f"  Layers compressed: {len(self.compression_stats)}")
        logger.info(f"  Overall compression ratio: {overall_ratio:.2f}x")
        logger.info(f"  Parameter reduction: {(1 - total_compressed/total_original)*100:.1f}%")
        logger.info(f"  Average quality loss: {avg_quality_loss:.2f}%")


class ModelCompressor:
    """
    High-level interface for model compression with automatic optimization.
    
    This class provides a simple interface for compressing models with
    automatic parameter tuning and quality monitoring.
    """
    
    def __init__(
        self,
        quality_threshold: float = 0.95,
        max_compression_ratio: float = 10.0,
        device: Optional[Union[str, torch.device]] = None
    ):
        self.quality_threshold = quality_threshold
        self.max_compression_ratio = max_compression_ratio
        self.device = device
    
    def compress(
        self, 
        model: nn.Module,
        compression_ratio: Optional[float] = None,
        target_size: Optional[int] = None,
        inplace: bool = False
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Compress model with automatic optimization.
        
        Args:
            model: Model to compress
            compression_ratio: Target compression ratio (e.g., 0.25 for 4x compression)
            target_size: Target model size in parameters
            inplace: Modify model in-place
            
        Returns:
            Tuple of (compressed_model, compression_info)
        """
        if compression_ratio is None and target_size is None:
            compression_ratio = 0.5  # Default 2x compression
            
        # Calculate compression ratio from target size if needed
        if target_size is not None:
            original_size = sum(p.numel() for p in model.parameters())
            compression_ratio = target_size / original_size
            
        # Use adaptive compression for quality preservation
        compressor = TensorSlim(
            rank=compression_ratio,
            method="adaptive",
            quality_threshold=self.quality_threshold,
            device=self.device
        )
        
        compressed_model = compressor.compress(model, inplace=inplace)
        
        # Prepare compression info
        compression_info = {
            'compression_stats': compressor.compression_stats,
            'method': 'adaptive_randomized_svd',
            'quality_threshold': self.quality_threshold,
            'target_compression_ratio': compression_ratio
        }
        
        return compressed_model, compression_info


def compress_model(
    model: nn.Module,
    compression_ratio: Union[int, float] = 0.5,
    quality_threshold: float = 0.95,
    method: str = "randomized_svd",
    inplace: bool = False,
    **kwargs
) -> nn.Module:
    """
    Convenience function for model compression.
    
    Args:
        model: PyTorch model to compress
        compression_ratio: Target compression (int for rank, float for ratio)
        quality_threshold: Minimum quality to maintain
        method: Compression method ('randomized_svd' or 'adaptive')
        inplace: Modify model in-place
        **kwargs: Additional arguments for TensorSlim
        
    Returns:
        Compressed model
    """
    compressor = TensorSlim(
        rank=compression_ratio,
        quality_threshold=quality_threshold,
        method=method,
        **kwargs
    )
    
    return compressor.compress(model, inplace=inplace)


def analyze_model_compression(
    model: nn.Module,
    compression_ratios: List[float] = None,
    device: Optional[Union[str, torch.device]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Analyze compression potential for different ratios.
    
    Args:
        model: Model to analyze
        compression_ratios: List of compression ratios to test
        device: Device for computation
        
    Returns:
        Dictionary with compression analysis results
    """
    if compression_ratios is None:
        compression_ratios = [0.1, 0.25, 0.5, 0.75]
        
    results = {}
    
    for ratio in compression_ratios:
        compressor = TensorSlim(
            rank=ratio,
            method="adaptive",
            device=device,
            progress_bar=False
        )
        
        # Compress copy of model
        test_model = copy.deepcopy(model)
        compressed = compressor.compress(test_model)
        
        # Calculate metrics
        original_params = sum(p.numel() for p in model.parameters())
        compressed_params = sum(p.numel() for p in compressed.parameters())
        actual_ratio = original_params / compressed_params
        
        results[f"ratio_{ratio}"] = {
            'target_ratio': 1.0 / ratio,
            'actual_ratio': actual_ratio,
            'parameter_reduction': (1 - compressed_params / original_params) * 100,
            'layers_compressed': len(compressor.compression_stats)
        }
        
    return results