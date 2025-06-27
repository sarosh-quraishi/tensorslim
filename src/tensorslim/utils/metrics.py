"""
Quality metrics and evaluation utilities for model compression.

This module provides various metrics to assess the quality of compressed models
and compare them with original models.
"""

from typing import Dict, List, Optional, Union, Callable, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import logging

logger = logging.getLogger(__name__)


def cosine_similarity(x: Tensor, y: Tensor, dim: int = -1) -> Tensor:
    """
    Compute cosine similarity between two tensors.
    
    Args:
        x: First tensor
        y: Second tensor  
        dim: Dimension along which to compute similarity
        
    Returns:
        Cosine similarity tensor
    """
    return F.cosine_similarity(x, y, dim=dim)


def mse_loss(x: Tensor, y: Tensor) -> Tensor:
    """
    Compute mean squared error between two tensors.
    
    Args:
        x: First tensor (e.g., original output)
        y: Second tensor (e.g., compressed output)
        
    Returns:
        MSE loss
    """
    return F.mse_loss(x, y)


def relative_error(x: Tensor, y: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Compute relative error between two tensors.
    
    Args:
        x: Reference tensor
        y: Comparison tensor
        eps: Small value to avoid division by zero
        
    Returns:
        Relative error
    """
    return torch.norm(x - y) / (torch.norm(x) + eps)


def frobenius_distance(x: Tensor, y: Tensor) -> Tensor:
    """
    Compute Frobenius norm distance between two tensors.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Frobenius distance
    """
    return torch.norm(x - y, p='fro')


def spectral_distance(x: Tensor, y: Tensor) -> Tensor:
    """
    Compute spectral norm distance between two matrices.
    
    Args:
        x: First matrix
        y: Second matrix
        
    Returns:
        Spectral distance
    """
    if x.dim() != 2 or y.dim() != 2:
        raise ValueError("Spectral distance requires 2D tensors (matrices)")
    
    diff = x - y
    return torch.norm(diff, p=2)  # Spectral norm


class CompressionMetrics:
    """
    Comprehensive metrics for evaluating model compression quality.
    
    This class provides various metrics to assess the impact of compression
    on model performance, including output similarity and layer-wise analysis.
    """
    
    def __init__(self):
        self.metrics = {}
    
    def compute_output_metrics(
        self,
        original_outputs: Union[Tensor, List[Tensor]],
        compressed_outputs: Union[Tensor, List[Tensor]]
    ) -> Dict[str, float]:
        """
        Compute comprehensive output similarity metrics.
        
        Args:
            original_outputs: Outputs from original model
            compressed_outputs: Outputs from compressed model
            
        Returns:
            Dictionary of computed metrics
        """
        metrics = {}
        
        # Handle single tensor or list of tensors
        if isinstance(original_outputs, Tensor):
            original_outputs = [original_outputs]
        if isinstance(compressed_outputs, Tensor):
            compressed_outputs = [compressed_outputs]
        
        if len(original_outputs) != len(compressed_outputs):
            raise ValueError("Number of outputs must match")
        
        # Compute metrics for each output tensor
        for i, (orig, comp) in enumerate(zip(original_outputs, compressed_outputs)):
            if orig.shape != comp.shape:
                logger.warning(f"Output {i} shapes don't match: {orig.shape} vs {comp.shape}")
                continue
            
            prefix = f"output_{i}_" if len(original_outputs) > 1 else ""
            
            # Cosine similarity
            cos_sim = cosine_similarity(orig.flatten(), comp.flatten()).item()
            metrics[f"{prefix}cosine_similarity"] = cos_sim
            
            # MSE
            mse = mse_loss(orig, comp).item()
            metrics[f"{prefix}mse"] = mse
            
            # Relative error
            rel_err = relative_error(orig, comp).item()
            metrics[f"{prefix}relative_error"] = rel_err
            
            # Frobenius distance
            frob_dist = frobenius_distance(orig, comp).item()
            metrics[f"{prefix}frobenius_distance"] = frob_dist
            
            # Pearson correlation (for flattened tensors)
            orig_flat = orig.flatten().cpu().numpy()
            comp_flat = comp.flatten().cpu().numpy()
            
            if len(orig_flat) > 1:
                corr = np.corrcoef(orig_flat, comp_flat)[0, 1]
                if not np.isnan(corr):
                    metrics[f"{prefix}pearson_correlation"] = corr
            
            # Signal-to-noise ratio
            signal_power = torch.mean(orig ** 2)
            noise_power = torch.mean((orig - comp) ** 2)
            
            if noise_power > 0:
                snr = 10 * torch.log10(signal_power / noise_power)
                metrics[f"{prefix}snr_db"] = snr.item()
        
        return metrics
    
    def compute_layer_metrics(
        self,
        original_model: nn.Module,
        compressed_model: nn.Module
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute layer-wise compression metrics.
        
        Args:
            original_model: Original model
            compressed_model: Compressed model
            
        Returns:
            Dictionary mapping layer names to their metrics
        """
        layer_metrics = {}
        
        # Get corresponding layers
        orig_layers = dict(original_model.named_modules())
        comp_layers = dict(compressed_model.named_modules())
        
        for name, orig_layer in orig_layers.items():
            if name not in comp_layers:
                continue
            
            comp_layer = comp_layers[name]
            
            # Only analyze layers with parameters
            if not list(orig_layer.parameters()) or not list(comp_layer.parameters()):
                continue
            
            metrics = {}
            
            # Parameter count comparison
            orig_params = sum(p.numel() for p in orig_layer.parameters())
            comp_params = sum(p.numel() for p in comp_layer.parameters())
            
            metrics['original_parameters'] = orig_params
            metrics['compressed_parameters'] = comp_params
            metrics['compression_ratio'] = orig_params / comp_params if comp_params > 0 else float('inf')
            metrics['parameter_reduction'] = 1 - (comp_params / orig_params) if orig_params > 0 else 0
            
            # Weight similarity (for compatible layers)
            if hasattr(orig_layer, 'weight') and hasattr(comp_layer, 'weight'):
                try:
                    # For compressed layers, reconstruct weight if possible
                    if hasattr(comp_layer, 'reconstruct_weight'):
                        comp_weight = comp_layer.reconstruct_weight()
                    else:
                        comp_weight = comp_layer.weight
                    
                    orig_weight = orig_layer.weight
                    
                    if orig_weight.shape == comp_weight.shape:
                        # Direct comparison
                        cos_sim = cosine_similarity(
                            orig_weight.flatten(), 
                            comp_weight.flatten()
                        ).item()
                        metrics['weight_cosine_similarity'] = cos_sim
                        
                        rel_err = relative_error(orig_weight, comp_weight).item()
                        metrics['weight_relative_error'] = rel_err
                    
                except Exception as e:
                        logger.debug(f"Could not compare weights for layer {name}: {e}")
            
            layer_metrics[name] = metrics
        
        return layer_metrics
    
    def compute_model_metrics(
        self,
        original_model: nn.Module,
        compressed_model: nn.Module,
        test_inputs: Optional[List[Tensor]] = None
    ) -> Dict[str, Any]:
        """
        Compute comprehensive model-level compression metrics.
        
        Args:
            original_model: Original model
            compressed_model: Compressed model
            test_inputs: Optional test inputs for output comparison
            
        Returns:
            Dictionary with all computed metrics
        """
        metrics = {
            'model_summary': {},
            'layer_metrics': {},
            'output_metrics': {}
        }
        
        # Model-level summary
        orig_params = sum(p.numel() for p in original_model.parameters())
        comp_params = sum(p.numel() for p in compressed_model.parameters())
        
        metrics['model_summary'] = {
            'original_parameters': orig_params,
            'compressed_parameters': comp_params,
            'overall_compression_ratio': orig_params / comp_params if comp_params > 0 else float('inf'),
            'overall_parameter_reduction': 1 - (comp_params / orig_params) if orig_params > 0 else 0
        }
        
        # Layer-wise metrics
        metrics['layer_metrics'] = self.compute_layer_metrics(original_model, compressed_model)
        
        # Output comparison metrics
        if test_inputs is not None:
            original_model.eval()
            compressed_model.eval()
            
            all_output_metrics = []
            
            with torch.no_grad():
                for test_input in test_inputs:
                    try:
                        orig_output = original_model(test_input)
                        comp_output = compressed_model(test_input)
                        
                        output_metrics = self.compute_output_metrics(orig_output, comp_output)
                        all_output_metrics.append(output_metrics)
                        
                    except Exception as e:
                        logger.warning(f"Failed to compute output metrics: {e}")
            
            # Average metrics across all test inputs
            if all_output_metrics:
                avg_metrics = {}
                for key in all_output_metrics[0].keys():
                    values = [m[key] for m in all_output_metrics if key in m]
                    if values:
                        avg_metrics[f"avg_{key}"] = sum(values) / len(values)
                
                metrics['output_metrics'] = avg_metrics
        
        return metrics


def evaluate_compression_quality(
    original_model: nn.Module,
    compressed_model: nn.Module,
    test_loader: Optional[torch.utils.data.DataLoader] = None,
    num_samples: int = 100,
    device: Optional[Union[str, torch.device]] = None
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of compression quality.
    
    Args:
        original_model: Original model
        compressed_model: Compressed model
        test_loader: Data loader for evaluation
        num_samples: Number of samples to evaluate
        device: Device for computation
        
    Returns:
        Comprehensive quality evaluation results
    """
    if device is not None:
        original_model = original_model.to(device)
        compressed_model = compressed_model.to(device)
    
    metrics_computer = CompressionMetrics()
    
    # Collect test inputs
    test_inputs = []
    if test_loader is not None:
        for i, batch in enumerate(test_loader):
            if i >= num_samples:
                break
            
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]  # Assume first element is input
            else:
                inputs = batch
            
            if device is not None:
                inputs = inputs.to(device)
            
            test_inputs.append(inputs)
    
    # Compute comprehensive metrics
    results = metrics_computer.compute_model_metrics(
        original_model,
        compressed_model, 
        test_inputs if test_inputs else None
    )
    
    # Add summary statistics
    if results['output_metrics']:
        # Quality score (weighted combination of metrics)
        cos_sim = results['output_metrics'].get('avg_cosine_similarity', 0)
        rel_err = results['output_metrics'].get('avg_relative_error', 1)
        
        # Simple quality score (0-1, higher is better)
        quality_score = cos_sim * (1 - min(rel_err, 1))
        results['quality_score'] = max(0, quality_score)
        
        # Quality grade
        if quality_score > 0.95:
            quality_grade = 'A'
        elif quality_score > 0.90:
            quality_grade = 'B'
        elif quality_score > 0.80:
            quality_grade = 'C'
        elif quality_score > 0.70:
            quality_grade = 'D'
        else:
            quality_grade = 'F'
        
        results['quality_grade'] = quality_grade
    
    return results


def create_compression_report(
    evaluation_results: Dict[str, Any],
    model_name: str = "Model"
) -> str:
    """
    Create a human-readable compression quality report.
    
    Args:
        evaluation_results: Results from evaluate_compression_quality
        model_name: Name of the model for the report
        
    Returns:
        Formatted report string
    """
    report = f"""
TensorSlim Compression Quality Report
===================================
Model: {model_name}
Date: {torch.datetime.now().strftime('%Y-%m-%d %H:%M:%S') if hasattr(torch, 'datetime') else 'N/A'}

COMPRESSION SUMMARY
------------------
Original Parameters: {evaluation_results['model_summary']['original_parameters']:,}
Compressed Parameters: {evaluation_results['model_summary']['compressed_parameters']:,}
Compression Ratio: {evaluation_results['model_summary']['overall_compression_ratio']:.2f}x
Parameter Reduction: {evaluation_results['model_summary']['overall_parameter_reduction']*100:.1f}%

QUALITY ASSESSMENT
-----------------
"""
    
    if 'quality_score' in evaluation_results:
        report += f"Overall Quality Score: {evaluation_results['quality_score']:.3f} (Grade: {evaluation_results['quality_grade']})\n"
    
    if evaluation_results['output_metrics']:
        output_metrics = evaluation_results['output_metrics']
        report += f"""
Output Similarity Metrics:
  Cosine Similarity: {output_metrics.get('avg_cosine_similarity', 'N/A'):.4f}
  Relative Error: {output_metrics.get('avg_relative_error', 'N/A'):.4f}
  Pearson Correlation: {output_metrics.get('avg_pearson_correlation', 'N/A'):.4f}
"""
        
        if 'avg_snr_db' in output_metrics:
            report += f"  Signal-to-Noise Ratio: {output_metrics['avg_snr_db']:.2f} dB\n"
    
    # Layer-wise summary
    layer_metrics = evaluation_results['layer_metrics']
    if layer_metrics:
        compressed_layers = [name for name, metrics in layer_metrics.items() 
                           if metrics['compression_ratio'] > 1.1]
        
        report += f"""
LAYER ANALYSIS
--------------
Total Layers Analyzed: {len(layer_metrics)}
Layers Compressed: {len(compressed_layers)}
"""
        
        if compressed_layers:
            avg_compression = sum(layer_metrics[name]['compression_ratio'] 
                                for name in compressed_layers) / len(compressed_layers)
            report += f"Average Layer Compression: {avg_compression:.2f}x\n"
    
    report += "\nRECOMMENDations\n"
    report += "---------------\n"
    
    quality_score = evaluation_results.get('quality_score', 0)
    if quality_score > 0.95:
        report += "✓ Excellent compression quality. Model is ready for deployment.\n"
    elif quality_score > 0.90:
        report += "✓ Good compression quality. Consider additional validation before deployment.\n"
    elif quality_score > 0.80:
        report += "⚠ Moderate compression quality. Consider reducing compression ratio.\n"
    else:
        report += "⚠ Low compression quality. Recommend reducing compression ratio or using different strategy.\n"
    
    compression_ratio = evaluation_results['model_summary']['overall_compression_ratio']
    if compression_ratio > 10:
        report += "⚠ Very high compression ratio. Monitor quality carefully.\n"
    elif compression_ratio > 5:
        report += "ⓘ High compression ratio achieved. Validate on representative datasets.\n"
    
    return report.strip()