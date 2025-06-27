"""
Base classes for model compression.

This module provides abstract base classes and common functionality
for model-specific compression implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Tuple, Any
import torch
import torch.nn as nn
from torch import Tensor


class BaseModelCompressor(ABC):
    """
    Abstract base class for model-specific compressors.
    
    This class defines the interface that all model compressors should implement,
    providing a consistent API across different architectures.
    """
    
    def __init__(
        self,
        compression_ratio: Union[int, float] = 0.5,
        quality_threshold: float = 0.95,
        device: Optional[Union[str, torch.device]] = None,
        progress_bar: bool = True
    ):
        self.compression_ratio = compression_ratio
        self.quality_threshold = quality_threshold
        self.device = device
        self.progress_bar = progress_bar
        
        # Statistics tracking
        self.compression_stats = {}
        
    @abstractmethod
    def compress(
        self, 
        model: nn.Module, 
        inplace: bool = False
    ) -> nn.Module:
        """
        Compress the given model.
        
        Args:
            model: Model to compress
            inplace: Whether to modify the model in-place
            
        Returns:
            Compressed model
        """
        pass
    
    @abstractmethod
    def analyze_model(self, model: nn.Module) -> Dict[str, Any]:
        """
        Analyze model structure for compression planning.
        
        Args:
            model: Model to analyze
            
        Returns:
            Dictionary with model analysis results
        """
        pass
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics from the last compression operation."""
        return self.compression_stats.copy()
    
    def estimate_compression_benefit(
        self, 
        model: nn.Module
    ) -> Dict[str, float]:
        """
        Estimate the potential benefits of compression.
        
        Args:
            model: Model to analyze
            
        Returns:
            Dictionary with estimated benefits
        """
        analysis = self.analyze_model(model)
        
        # Calculate potential parameter reduction
        compressible_params = analysis.get('compressible_parameters', 0)
        total_params = analysis.get('total_parameters', 0)
        
        if total_params == 0:
            return {'parameter_reduction': 0.0, 'memory_savings': 0.0}
        
        # Estimate compression based on configuration
        if isinstance(self.compression_ratio, float):
            estimated_compressed_params = compressible_params * self.compression_ratio
        else:
            # For rank-based compression, estimate based on typical ratios
            estimated_compressed_params = compressible_params * 0.3  # Conservative estimate
        
        total_compressed_params = (total_params - compressible_params) + estimated_compressed_params
        
        parameter_reduction = 1.0 - (total_compressed_params / total_params)
        memory_savings = parameter_reduction * analysis.get('model_size_mb', 0)
        
        return {
            'parameter_reduction': parameter_reduction,
            'memory_savings_mb': memory_savings,
            'estimated_compression_ratio': total_params / total_compressed_params
        }


class LayerAnalyzer:
    """
    Utility class for analyzing neural network layers.
    
    This class provides methods to analyze layer properties, compression potential,
    and relationships between layers.
    """
    
    @staticmethod
    def get_layer_info(layer: nn.Module) -> Dict[str, Any]:
        """
        Extract comprehensive information about a layer.
        
        Args:
            layer: PyTorch layer to analyze
            
        Returns:
            Dictionary with layer information
        """
        info = {
            'type': type(layer).__name__,
            'parameters': sum(p.numel() for p in layer.parameters()),
            'trainable_parameters': sum(p.numel() for p in layer.parameters() if p.requires_grad),
            'compressible': False,
            'compression_potential': 0.0
        }
        
        # Layer-specific information
        if isinstance(layer, nn.Linear):
            info.update({
                'in_features': layer.in_features,
                'out_features': layer.out_features,
                'has_bias': layer.bias is not None,
                'compressible': True,
                'compression_potential': LayerAnalyzer._estimate_linear_compression_potential(layer)
            })
            
        elif isinstance(layer, nn.Conv2d):
            info.update({
                'in_channels': layer.in_channels,
                'out_channels': layer.out_channels,
                'kernel_size': layer.kernel_size,
                'stride': layer.stride,
                'padding': layer.padding,
                'has_bias': layer.bias is not None,
                'compressible': True,
                'compression_potential': LayerAnalyzer._estimate_conv_compression_potential(layer)
            })
            
        elif isinstance(layer, nn.Embedding):
            info.update({
                'num_embeddings': layer.num_embeddings,
                'embedding_dim': layer.embedding_dim,
                'padding_idx': layer.padding_idx,
                'compressible': True,
                'compression_potential': LayerAnalyzer._estimate_embedding_compression_potential(layer)
            })
        
        return info
    
    @staticmethod
    def _estimate_linear_compression_potential(layer: nn.Linear) -> float:
        """Estimate compression potential for linear layers."""
        in_dim, out_dim = layer.in_features, layer.out_features
        min_dim = min(in_dim, out_dim)
        
        # Higher potential for larger, more square matrices
        size_factor = (in_dim * out_dim) / (1000 * 1000)  # Normalize by 1M parameters
        aspect_ratio = min(in_dim, out_dim) / max(in_dim, out_dim)
        
        potential = min(0.9, 0.3 + 0.4 * aspect_ratio + 0.2 * min(1.0, size_factor))
        return potential
    
    @staticmethod
    def _estimate_conv_compression_potential(layer: nn.Conv2d) -> float:
        """Estimate compression potential for convolutional layers."""
        in_ch, out_ch = layer.in_channels, layer.out_channels
        kernel_size = layer.kernel_size[0] * layer.kernel_size[1]
        
        # Larger kernels and more channels generally compress better
        size_factor = (in_ch * out_ch * kernel_size) / 10000  # Normalize
        kernel_factor = min(1.0, kernel_size / 9)  # 3x3 kernel as baseline
        
        potential = min(0.8, 0.2 + 0.4 * kernel_factor + 0.2 * min(1.0, size_factor))
        return potential
    
    @staticmethod
    def _estimate_embedding_compression_potential(layer: nn.Embedding) -> float:
        """Estimate compression potential for embedding layers."""
        vocab_size, embed_dim = layer.num_embeddings, layer.embedding_dim
        
        # Large embedding tables have high compression potential
        size_factor = (vocab_size * embed_dim) / (50000 * 768)  # BERT-base as baseline
        
        potential = min(0.9, 0.4 + 0.5 * min(1.0, size_factor))
        return potential
    
    @staticmethod
    def identify_layer_groups(model: nn.Module) -> Dict[str, List[str]]:
        """
        Group layers by their role in the model architecture.
        
        Args:
            model: Model to analyze
            
        Returns:
            Dictionary mapping group names to lists of layer names
        """
        groups = {
            'embeddings': [],
            'attention': [],
            'feedforward': [],
            'normalization': [],
            'output': [],
            'other': []
        }
        
        for name, module in model.named_modules():
            if not isinstance(module, (nn.Linear, nn.Conv2d, nn.Embedding, nn.LayerNorm, nn.BatchNorm2d)):
                continue
                
            name_lower = name.lower()
            
            # Classify based on name patterns
            if any(pattern in name_lower for pattern in ['embed', 'embedding']):
                groups['embeddings'].append(name)
            elif any(pattern in name_lower for pattern in ['attn', 'attention', 'query', 'key', 'value']):
                groups['attention'].append(name)
            elif any(pattern in name_lower for pattern in ['ffn', 'mlp', 'feed', 'intermediate']):
                groups['feedforward'].append(name)
            elif any(pattern in name_lower for pattern in ['norm', 'ln', 'bn']):
                groups['normalization'].append(name)
            elif any(pattern in name_lower for pattern in ['output', 'head', 'classifier']):
                groups['output'].append(name)
            else:
                groups['other'].append(name)
        
        return groups


class CompressionStrategy:
    """
    Defines compression strategies for different model architectures.
    
    This class encapsulates the logic for choosing compression parameters
    based on model characteristics and target requirements.
    """
    
    def __init__(
        self,
        base_compression_ratio: float = 0.5,
        layer_specific_ratios: Optional[Dict[str, float]] = None,
        quality_threshold: float = 0.95,
        preserve_patterns: Optional[List[str]] = None
    ):
        self.base_compression_ratio = base_compression_ratio
        self.layer_specific_ratios = layer_specific_ratios or {}
        self.quality_threshold = quality_threshold
        self.preserve_patterns = preserve_patterns or ['norm', 'bias']
    
    def get_layer_compression_ratio(
        self, 
        layer_name: str, 
        layer_info: Dict[str, Any]
    ) -> Optional[float]:
        """
        Determine compression ratio for a specific layer.
        
        Args:
            layer_name: Name of the layer
            layer_info: Layer information from LayerAnalyzer
            
        Returns:
            Compression ratio or None if layer should be preserved
        """
        # Check if layer should be preserved
        name_lower = layer_name.lower()
        if any(pattern in name_lower for pattern in self.preserve_patterns):
            return None
        
        # Check layer-specific overrides
        for pattern, ratio in self.layer_specific_ratios.items():
            if pattern in name_lower:
                return ratio
        
        # Use base ratio adjusted by compression potential
        potential = layer_info.get('compression_potential', 0.5)
        adjusted_ratio = self.base_compression_ratio * (0.5 + 0.5 * potential)
        
        return adjusted_ratio
    
    def create_compression_plan(
        self, 
        model: nn.Module
    ) -> Dict[str, Dict[str, Any]]:
        """
        Create a comprehensive compression plan for the model.
        
        Args:
            model: Model to create plan for
            
        Returns:
            Dictionary mapping layer names to compression configurations
        """
        plan = {}
        
        for name, module in model.named_modules():
            if not isinstance(module, (nn.Linear, nn.Conv2d, nn.Embedding)):
                continue
            
            layer_info = LayerAnalyzer.get_layer_info(module)
            compression_ratio = self.get_layer_compression_ratio(name, layer_info)
            
            if compression_ratio is not None:
                plan[name] = {
                    'compression_ratio': compression_ratio,
                    'layer_info': layer_info,
                    'method': 'randomized_svd',
                    'quality_threshold': self.quality_threshold
                }
        
        return plan


class ModelProfiler:
    """
    Profiles model performance and memory usage.
    
    This class provides utilities for measuring model size, inference speed,
    and memory consumption before and after compression.
    """
    
    @staticmethod
    def profile_model(
        model: nn.Module,
        sample_input: Optional[Tensor] = None,
        device: Optional[Union[str, torch.device]] = None
    ) -> Dict[str, Any]:
        """
        Profile model performance characteristics.
        
        Args:
            model: Model to profile
            sample_input: Sample input for timing measurements
            device: Device to run profiling on
            
        Returns:
            Dictionary with profiling results
        """
        if device is not None:
            model = model.to(device)
            if sample_input is not None:
                sample_input = sample_input.to(device)
        
        profile = {
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'model_size_mb': ModelProfiler._calculate_model_size(model),
            'layer_count': len(list(model.modules())),
            'linear_layers': len([m for m in model.modules() if isinstance(m, nn.Linear)]),
            'conv_layers': len([m for m in model.modules() if isinstance(m, nn.Conv2d)]),
            'embedding_layers': len([m for m in model.modules() if isinstance(m, nn.Embedding)])
        }
        
        # Performance profiling if sample input provided
        if sample_input is not None:
            profile.update(ModelProfiler._profile_inference(model, sample_input))
        
        return profile
    
    @staticmethod
    def _calculate_model_size(model: nn.Module) -> float:
        """Calculate model size in MB."""
        total_bytes = 0
        for param in model.parameters():
            total_bytes += param.numel() * param.element_size()
        return total_bytes / (1024 * 1024)  # Convert to MB
    
    @staticmethod
    def _profile_inference(model: nn.Module, sample_input: Tensor) -> Dict[str, float]:
        """Profile inference performance."""
        import time
        
        model.eval()
        
        # Warm up
        with torch.no_grad():
            for _ in range(3):
                _ = model(sample_input)
        
        # Time inference
        times = []
        with torch.no_grad():
            for _ in range(10):
                start_time = time.time()
                _ = model(sample_input)
                end_time = time.time()
                times.append(end_time - start_time)
        
        return {
            'avg_inference_time_ms': sum(times) / len(times) * 1000,
            'min_inference_time_ms': min(times) * 1000,
            'max_inference_time_ms': max(times) * 1000
        }