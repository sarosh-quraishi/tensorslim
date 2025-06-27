"""
PyTorch integration utilities for TensorSlim.

This module provides utilities for working with PyTorch models,
including model analysis, optimization, and integration helpers.
"""

from typing import Dict, List, Optional, Union, Any, Callable
import torch
import torch.nn as nn
from torch import Tensor
import logging

from ..core import TensorSlim, compress_model
from ..models import ModelProfiler, LayerAnalyzer

logger = logging.getLogger(__name__)


class PyTorchCompressor:
    """
    General-purpose PyTorch model compressor.
    
    This class provides a unified interface for compressing arbitrary PyTorch
    models with automatic layer detection and optimization.
    """
    
    def __init__(
        self,
        compression_ratio: Union[int, float] = 0.5,
        target_layers: Optional[List[str]] = None,
        preserve_layers: Optional[List[str]] = None,
        quality_threshold: float = 0.95,
        device: Optional[Union[str, torch.device]] = None
    ):
        self.compression_ratio = compression_ratio
        self.target_layers = target_layers or ['Linear', 'Conv2d', 'Embedding']
        self.preserve_layers = preserve_layers or ['BatchNorm', 'LayerNorm', 'Dropout']
        self.quality_threshold = quality_threshold
        self.device = device
    
    def compress(
        self,
        model: nn.Module,
        inplace: bool = False,
        validation_fn: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Compress PyTorch model with quality validation.
        
        Args:
            model: PyTorch model to compress
            inplace: Whether to modify model in-place
            validation_fn: Optional function to validate quality
            
        Returns:
            Dictionary with compressed model and compression info
        """
        # Profile original model
        original_profile = ModelProfiler.profile_model(model)
        
        # Create compressor
        compressor = TensorSlim(
            rank=self.compression_ratio,
            target_layers=self.target_layers,
            preserve_layers=self.preserve_layers,
            quality_threshold=self.quality_threshold,
            device=self.device
        )
        
        # Compress model
        compressed_model = compressor.compress(model, inplace=inplace)
        
        # Validate quality if function provided
        quality_metrics = {}
        if validation_fn is not None:
            try:
                quality_metrics = validation_fn(model, compressed_model)
            except Exception as e:
                logger.warning(f"Quality validation failed: {e}")
        
        # Profile compressed model
        compressed_profile = ModelProfiler.profile_model(compressed_model)
        
        return {
            'compressed_model': compressed_model,
            'original_profile': original_profile,
            'compressed_profile': compressed_profile,
            'compression_stats': compressor.compression_stats,
            'quality_metrics': quality_metrics,
            'parameter_reduction': 1 - (compressed_profile['total_parameters'] / original_profile['total_parameters'])
        }


def optimize_model_for_inference(
    model: nn.Module,
    sample_input: Optional[Tensor] = None,
    enable_fusion: bool = True,
    enable_quantization: bool = False
) -> nn.Module:
    """
    Optimize PyTorch model for inference performance.
    
    Args:
        model: Model to optimize
        sample_input: Sample input for optimization
        enable_fusion: Whether to enable operator fusion
        enable_quantization: Whether to enable quantization
        
    Returns:
        Optimized model
    """
    model.eval()
    
    # Apply torch.jit optimization if sample input provided
    if sample_input is not None:
        try:
            model = torch.jit.trace(model, sample_input)
            logger.info("Applied TorchScript tracing optimization")
        except Exception as e:
            logger.warning(f"TorchScript tracing failed: {e}")
    
    # Apply operator fusion
    if enable_fusion:
        try:
            model = torch.jit.optimize_for_inference(model)
            logger.info("Applied operator fusion optimization")
        except Exception as e:
            logger.warning(f"Operator fusion failed: {e}")
    
    # Apply quantization
    if enable_quantization:
        try:
            model = torch.quantization.quantize_dynamic(
                model, 
                {nn.Linear, nn.Conv2d}, 
                dtype=torch.qint8
            )
            logger.info("Applied dynamic quantization")
        except Exception as e:
            logger.warning(f"Quantization failed: {e}")
    
    return model


def analyze_model_layers(model: nn.Module) -> Dict[str, Any]:
    """
    Comprehensive analysis of model layers.
    
    Args:
        model: Model to analyze
        
    Returns:
        Dictionary with layer analysis results
    """
    analysis = {
        'total_layers': 0,
        'layer_types': {},
        'compressible_layers': [],
        'parameter_distribution': {},
        'compression_potential': {}
    }
    
    total_params = 0
    compressible_params = 0
    
    for name, module in model.named_modules():
        if len(list(module.children())) > 0:
            continue  # Skip container modules
        
        analysis['total_layers'] += 1
        
        # Analyze layer
        layer_info = LayerAnalyzer.get_layer_info(module)
        layer_type = layer_info['type']
        
        # Count layer types
        if layer_type not in analysis['layer_types']:
            analysis['layer_types'][layer_type] = 0
        analysis['layer_types'][layer_type] += 1
        
        # Track parameters
        layer_params = layer_info['parameters']
        total_params += layer_params
        
        if layer_info['compressible']:
            compressible_params += layer_params
            analysis['compressible_layers'].append({
                'name': name,
                'type': layer_type,
                'parameters': layer_params,
                'compression_potential': layer_info['compression_potential']
            })
        
        # Parameter distribution
        if layer_type not in analysis['parameter_distribution']:
            analysis['parameter_distribution'][layer_type] = 0
        analysis['parameter_distribution'][layer_type] += layer_params
    
    # Calculate compression potential
    analysis['total_parameters'] = total_params
    analysis['compressible_parameters'] = compressible_params
    analysis['compressible_ratio'] = compressible_params / total_params if total_params > 0 else 0
    
    # Average compression potential
    if analysis['compressible_layers']:
        avg_potential = sum(
            layer['compression_potential'] for layer in analysis['compressible_layers']
        ) / len(analysis['compressible_layers'])
        analysis['average_compression_potential'] = avg_potential
    else:
        analysis['average_compression_potential'] = 0
    
    return analysis


def create_model_summary(model: nn.Module, input_size: Optional[tuple] = None) -> str:
    """
    Create a detailed model summary.
    
    Args:
        model: Model to summarize
        input_size: Input size for parameter calculation
        
    Returns:
        String summary of the model
    """
    analysis = analyze_model_layers(model)
    profile = ModelProfiler.profile_model(model)
    
    summary = f"""
PyTorch Model Summary
====================
Total Layers: {analysis['total_layers']}
Total Parameters: {profile['total_parameters']:,}
Trainable Parameters: {profile['trainable_parameters']:,}
Model Size: {profile['model_size_mb']:.2f} MB

Layer Distribution:
"""
    
    for layer_type, count in analysis['layer_types'].items():
        param_count = analysis['parameter_distribution'].get(layer_type, 0)
        param_pct = (param_count / analysis['total_parameters']) * 100 if analysis['total_parameters'] > 0 else 0
        summary += f"  {layer_type}: {count} layers, {param_count:,} params ({param_pct:.1f}%)\n"
    
    summary += f"""
Compression Analysis:
  Compressible Parameters: {analysis['compressible_parameters']:,} ({analysis['compressible_ratio']*100:.1f}%)
  Average Compression Potential: {analysis['average_compression_potential']*100:.1f}%
  Compressible Layers: {len(analysis['compressible_layers'])}
"""
    
    return summary.strip()


class ModelConverter:
    """
    Utility class for converting models between different formats.
    """
    
    @staticmethod
    def to_onnx(
        model: nn.Module,
        sample_input: Tensor,
        onnx_path: str,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None
    ) -> None:
        """
        Export PyTorch model to ONNX format.
        
        Args:
            model: PyTorch model to export
            sample_input: Sample input tensor
            onnx_path: Path to save ONNX model
            input_names: Names for input nodes
            output_names: Names for output nodes
            dynamic_axes: Dynamic axis specifications
        """
        model.eval()
        
        with torch.no_grad():
            torch.onnx.export(
                model,
                sample_input,
                onnx_path,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=11
            )
        
        logger.info(f"Model exported to ONNX: {onnx_path}")
    
    @staticmethod
    def to_torchscript(
        model: nn.Module,
        sample_input: Optional[Tensor] = None,
        scripted_path: Optional[str] = None
    ) -> torch.jit.ScriptModule:
        """
        Convert PyTorch model to TorchScript.
        
        Args:
            model: PyTorch model to convert
            sample_input: Sample input for tracing
            scripted_path: Optional path to save scripted model
            
        Returns:
            TorchScript model
        """
        model.eval()
        
        if sample_input is not None:
            # Use tracing
            scripted_model = torch.jit.trace(model, sample_input)
        else:
            # Use scripting
            scripted_model = torch.jit.script(model)
        
        if scripted_path is not None:
            scripted_model.save(scripted_path)
            logger.info(f"TorchScript model saved: {scripted_path}")
        
        return scripted_model


def benchmark_model_inference(
    model: nn.Module,
    sample_input: Tensor,
    num_runs: int = 100,
    warmup_runs: int = 10
) -> Dict[str, float]:
    """
    Benchmark model inference performance.
    
    Args:
        model: Model to benchmark
        sample_input: Input tensor for benchmarking
        num_runs: Number of inference runs
        warmup_runs: Number of warmup runs
        
    Returns:
        Dictionary with benchmark results
    """
    import time
    
    model.eval()
    device = next(model.parameters()).device
    sample_input = sample_input.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(sample_input)
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            _ = model(sample_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            times.append(end_time - start_time)
    
    return {
        'avg_time_ms': (sum(times) / len(times)) * 1000,
        'min_time_ms': min(times) * 1000,
        'max_time_ms': max(times) * 1000,
        'std_time_ms': (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5 * 1000,
        'throughput_fps': 1.0 / (sum(times) / len(times))
    }


def compare_models(
    original_model: nn.Module,
    compressed_model: nn.Module,
    sample_input: Tensor,
    similarity_fn: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Compare original and compressed models.
    
    Args:
        original_model: Original model
        compressed_model: Compressed model
        sample_input: Sample input for comparison
        similarity_fn: Function to compute output similarity
        
    Returns:
        Dictionary with comparison results
    """
    # Profile both models
    original_profile = ModelProfiler.profile_model(original_model, sample_input)
    compressed_profile = ModelProfiler.profile_model(compressed_model, sample_input)
    
    # Benchmark both models
    original_benchmark = benchmark_model_inference(original_model, sample_input)
    compressed_benchmark = benchmark_model_inference(compressed_model, sample_input)
    
    # Compare outputs
    output_similarity = None
    if similarity_fn is not None:
        original_model.eval()
        compressed_model.eval()
        
        with torch.no_grad():
            original_output = original_model(sample_input)
            compressed_output = compressed_model(sample_input)
            output_similarity = similarity_fn(original_output, compressed_output)
    
    return {
        'size_comparison': {
            'original_params': original_profile['total_parameters'],
            'compressed_params': compressed_profile['total_parameters'],
            'parameter_reduction': 1 - (compressed_profile['total_parameters'] / original_profile['total_parameters']),
            'original_size_mb': original_profile['model_size_mb'],
            'compressed_size_mb': compressed_profile['model_size_mb'],
            'size_reduction_mb': original_profile['model_size_mb'] - compressed_profile['model_size_mb']
        },
        'performance_comparison': {
            'original_avg_time_ms': original_benchmark['avg_time_ms'],
            'compressed_avg_time_ms': compressed_benchmark['avg_time_ms'],
            'speedup': original_benchmark['avg_time_ms'] / compressed_benchmark['avg_time_ms'],
            'original_throughput_fps': original_benchmark['throughput_fps'],
            'compressed_throughput_fps': compressed_benchmark['throughput_fps']
        },
        'output_similarity': output_similarity,
        'original_benchmark': original_benchmark,
        'compressed_benchmark': compressed_benchmark
    }