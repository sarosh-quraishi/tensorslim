"""
Transformer-specific compression optimizations.

This module provides specialized compression techniques for transformer architectures,
taking advantage of their structure for better compression ratios and quality.
"""

from typing import Dict, List, Optional, Union, Tuple, Any
import torch
import torch.nn as nn
from torch import Tensor
import logging
from tqdm import tqdm

from ..core import TensorSlim, RandomizedSVD, SlimLinear
from ..core.layers import convert_layer_to_slim

logger = logging.getLogger(__name__)


class TransformerSlim:
    """
    Specialized compression for transformer architectures.
    
    This class provides optimized compression strategies for transformers,
    including attention-aware compression and layer-specific optimization.
    
    Args:
        ffn_rank: Rank for feed-forward network layers
        output_rank: Rank for output projection layers
        preserve_embeddings: Whether to preserve embedding layers
        preserve_layernorm: Whether to preserve layer normalization
        preserve_attention: Whether to preserve attention layers (recommended)
        quality_threshold: Minimum quality to maintain
        device: Device for computation
    """
    
    def __init__(
        self,
        ffn_rank: Union[int, float] = 128, 
        output_rank: Union[int, float] = 256,
        preserve_embeddings: bool = True,
        preserve_layernorm: bool = True,
        preserve_attention: bool = True,
        quality_threshold: float = 0.95,
        device: Optional[Union[str, torch.device]] = None,
        progress_bar: bool = True
    ):
        self.ffn_rank = ffn_rank
        self.output_rank = output_rank
        self.preserve_embeddings = preserve_embeddings
        self.preserve_layernorm = preserve_layernorm
        self.preserve_attention = preserve_attention
        self.quality_threshold = quality_threshold
        self.device = device
        self.progress_bar = progress_bar
        
        # Compression statistics
        self.compression_stats = {}
    
    def compress(
        self,
        model: nn.Module,
        inplace: bool = False
    ) -> nn.Module:
        """
        Compress transformer model with architecture-aware optimization.
        
        Args:
            model: Transformer model to compress
            inplace: Modify model in-place
            
        Returns:
            Compressed transformer model
        """
        if not inplace:
            import copy
            model = copy.deepcopy(model)
            
        if self.device is not None:
            model = model.to(self.device)
            
        # Analyze transformer structure
        transformer_info = self._analyze_transformer_structure(model)
        logger.info(f"Detected transformer with {len(transformer_info['layers'])} layers")
        
        # Apply layer-specific compression
        self.compression_stats = {}
        
        if self.progress_bar:
            pbar = tqdm(
                transformer_info['layers'],
                desc="Compressing transformer layers",
                unit="layer"
            )
        else:
            pbar = transformer_info['layers']
            
        for layer_idx, layer_info in enumerate(pbar):
            self._compress_transformer_layer(
                model, 
                layer_info, 
                layer_idx,
                transformer_info['total_layers']
            )
            
            if self.progress_bar and hasattr(pbar, 'set_postfix'):
                pbar.set_postfix({
                    'layer': f"{layer_idx+1}/{transformer_info['total_layers']}",
                    'type': layer_info.get('type', 'unknown')
                })
        
        if self.progress_bar and hasattr(pbar, 'close'):
            pbar.close()
            
        # Log compression summary
        self._log_compression_summary()
        
        return model
    
    def _analyze_transformer_structure(
        self, 
        model: nn.Module
    ) -> Dict[str, Any]:
        """
        Analyze transformer structure to identify key components.
        
        Returns:
            Dictionary with transformer structure information
        """
        structure = {
            'layers': [],
            'embeddings': [],
            'total_layers': 0,
            'attention_layers': 0,
            'ffn_layers': 0
        }
        
        # Common transformer layer patterns
        attention_patterns = [
            'attention', 'attn', 'self_attn', 'multihead', 
            'query', 'key', 'value', 'q_proj', 'k_proj', 'v_proj'
        ]
        
        ffn_patterns = [
            'ffn', 'feed_forward', 'intermediate', 'mlp',
            'fc1', 'fc2', 'linear1', 'linear2', 'dense'
        ]
        
        embedding_patterns = [
            'embedding', 'embeddings', 'word_embeddings', 
            'position_embeddings', 'token_embeddings'
        ]
        
        output_patterns = [
            'output', 'out_proj', 'dense', 'classifier', 'lm_head'
        ]
        
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
                
            layer_info = {
                'name': name,
                'module': module,
                'type': 'unknown'
            }
            
            name_lower = name.lower()
            
            # Categorize layer type
            if any(pattern in name_lower for pattern in attention_patterns):
                layer_info['type'] = 'attention'
                structure['attention_layers'] += 1
            elif any(pattern in name_lower for pattern in ffn_patterns):
                layer_info['type'] = 'ffn'  
                structure['ffn_layers'] += 1
            elif any(pattern in name_lower for pattern in embedding_patterns):
                layer_info['type'] = 'embedding'
            elif any(pattern in name_lower for pattern in output_patterns):
                layer_info['type'] = 'output'
            else:
                # Try to infer from layer dimensions
                layer_info['type'] = self._infer_layer_type(module)
            
            structure['layers'].append(layer_info)
            structure['total_layers'] += 1
            
        return structure
    
    def _infer_layer_type(self, layer: nn.Linear) -> str:
        """Infer layer type from dimensions and characteristics."""
        in_features = layer.in_features
        out_features = layer.out_features
        
        # Common patterns in transformer dimensions
        if in_features == out_features:
            # Square matrices often in attention or residual connections
            return 'attention'
        elif out_features > in_features * 2:
            # Expansion typical in FFN first layer
            return 'ffn'
        elif in_features > out_features * 2:
            # Contraction typical in FFN second layer or output
            return 'ffn'
        else:
            return 'linear'
    
    def _compress_transformer_layer(
        self,
        model: nn.Module,
        layer_info: Dict[str, Any],
        layer_idx: int,
        total_layers: int
    ) -> None:
        """Compress a single transformer layer with type-specific optimization."""
        layer_name = layer_info['name']
        layer_module = layer_info['module']
        layer_type = layer_info['type']
        
        # Skip preservation cases
        if layer_type == 'embedding' and self.preserve_embeddings:
            logger.debug(f"Preserving embedding layer: {layer_name}")
            return
            
        if 'layernorm' in layer_name.lower() and self.preserve_layernorm:
            logger.debug(f"Preserving layer norm: {layer_name}")
            return
            
        # Skip attention layers as recommended in README
        if layer_type == 'attention' and self.preserve_attention:
            logger.debug(f"Skipping attention layer (preserving for quality): {layer_name}")
            return
        
        # Determine compression rank based on layer type
        rank = self._get_layer_rank(layer_type, layer_module, layer_idx, total_layers)
        
        if rank <= 0:
            logger.debug(f"Skipping layer {layer_name} (rank too small)")
            return
            
        # Apply compression
        try:
            compressed_layer = convert_layer_to_slim(
                layer_module,
                rank=rank,
                n_power_iterations=1,
                n_oversamples=10
            )
            
            if compressed_layer is not None:
                self._replace_layer(model, layer_name, compressed_layer)
                
                # Store compression stats
                self.compression_stats[layer_name] = {
                    'type': layer_type,
                    'rank': rank,
                    'compression_ratio': compressed_layer.compression_ratio(),
                    'original_params': layer_module.weight.numel() + (
                        layer_module.bias.numel() if layer_module.bias is not None else 0
                    ),
                    'compressed_params': sum(p.numel() for p in compressed_layer.parameters())
                }
                
                logger.debug(
                    f"Compressed {layer_type} layer {layer_name}: "
                    f"{compressed_layer.compression_ratio():.2f}x compression"
                )
                
        except Exception as e:
            logger.warning(f"Failed to compress layer {layer_name}: {e}")
    
    def _get_layer_rank(
        self,
        layer_type: str,
        layer_module: nn.Linear,
        layer_idx: int,
        total_layers: int
    ) -> int:
        """Determine compression rank for specific layer type."""
        if layer_type == 'ffn':
            base_rank = self.ffn_rank
        elif layer_type == 'output':
            base_rank = self.output_rank
        else:
            # Default to FFN rank for unknown layers
            base_rank = self.ffn_rank
            
        # Convert float ratios to absolute ranks
        if isinstance(base_rank, float):
            min_dim = min(layer_module.in_features, layer_module.out_features)
            base_rank = max(1, int(min_dim * base_rank))
        
        # Layer position adjustments can be added here if needed
        # Currently focused on FFN layers only
        
        # Ensure rank doesn't exceed layer dimensions
        max_rank = min(layer_module.in_features, layer_module.out_features)
        return min(base_rank, max_rank)
    
    def _replace_layer(
        self,
        model: nn.Module,
        layer_name: str,
        new_layer: nn.Module
    ) -> None:
        """Replace a layer in the model."""
        parts = layer_name.split('.')
        parent = model
        
        for part in parts[:-1]:
            parent = getattr(parent, part)
            
        setattr(parent, parts[-1], new_layer)
    
    def _log_compression_summary(self) -> None:
        """Log compression statistics summary."""
        if not self.compression_stats:
            return
            
        total_original = sum(s['original_params'] for s in self.compression_stats.values())
        total_compressed = sum(s['compressed_params'] for s in self.compression_stats.values())
        overall_ratio = total_original / total_compressed if total_compressed > 0 else 0
        
        # Statistics by layer type
        type_stats = {}
        for stats in self.compression_stats.values():
            layer_type = stats['type']
            if layer_type not in type_stats:
                type_stats[layer_type] = {'count': 0, 'total_ratio': 0}
            type_stats[layer_type]['count'] += 1
            type_stats[layer_type]['total_ratio'] += stats['compression_ratio']
        
        logger.info("Transformer Compression Summary:")
        logger.info(f"  Total layers compressed: {len(self.compression_stats)}")
        logger.info(f"  Overall compression ratio: {overall_ratio:.2f}x")
        logger.info(f"  Parameter reduction: {(1 - total_compressed/total_original)*100:.1f}%")
        
        for layer_type, stats in type_stats.items():
            avg_ratio = stats['total_ratio'] / stats['count']
            logger.info(f"  {layer_type.capitalize()} layers: {stats['count']} compressed, "
                       f"avg {avg_ratio:.2f}x compression")


class AttentionSlim:
    """
    Specialized compression for multi-head attention layers.
    
    This class implements attention-aware compression that can preserve
    attention patterns while reducing parameters.
    """
    
    def __init__(
        self,
        num_heads: int,
        head_rank: Union[int, float] = 0.5,
        preserve_heads: Optional[List[int]] = None,
        head_importance_threshold: float = 0.1
    ):
        self.num_heads = num_heads
        self.head_rank = head_rank
        self.preserve_heads = preserve_heads or []
        self.head_importance_threshold = head_importance_threshold
    
    def compress_attention_weights(
        self,
        query_weight: Tensor,
        key_weight: Tensor, 
        value_weight: Tensor,
        output_weight: Tensor
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Compress attention weights with head-aware optimization.
        
        Args:
            query_weight: Query projection weight matrix
            key_weight: Key projection weight matrix
            value_weight: Value projection weight matrix
            output_weight: Output projection weight matrix
            
        Returns:
            Tuple of compressed weight dictionaries for Q, K, V, O
        """
        d_model = query_weight.size(1)
        d_k = d_model // self.num_heads
        
        # Analyze head importance if not specified
        if not self.preserve_heads:
            head_importance = self._analyze_head_importance(
                query_weight, key_weight, value_weight
            )
            self.preserve_heads = [
                i for i, imp in enumerate(head_importance)
                if imp > self.head_importance_threshold
            ]
        
        # Compress each projection with head awareness
        compressed_q = self._compress_projection_by_heads(
            query_weight, self.num_heads, self.preserve_heads
        )
        compressed_k = self._compress_projection_by_heads(
            key_weight, self.num_heads, self.preserve_heads
        )
        compressed_v = self._compress_projection_by_heads(
            value_weight, self.num_heads, self.preserve_heads
        )
        compressed_o = self._compress_output_projection(
            output_weight, self.num_heads, self.preserve_heads
        )
        
        return compressed_q, compressed_k, compressed_v, compressed_o
    
    def _analyze_head_importance(
        self,
        query_weight: Tensor,
        key_weight: Tensor,
        value_weight: Tensor
    ) -> List[float]:
        """
        Analyze the importance of attention heads based on weight magnitudes.
        
        Returns:
            List of importance scores for each head
        """
        d_model = query_weight.size(1)
        d_k = d_model // self.num_heads
        
        head_importance = []
        
        for head_idx in range(self.num_heads):
            start_idx = head_idx * d_k
            end_idx = (head_idx + 1) * d_k
            
            # Extract head weights
            q_head = query_weight[start_idx:end_idx, :]
            k_head = key_weight[start_idx:end_idx, :]
            v_head = value_weight[start_idx:end_idx, :]
            
            # Calculate importance as average Frobenius norm
            q_norm = torch.norm(q_head, p='fro')
            k_norm = torch.norm(k_head, p='fro')
            v_norm = torch.norm(v_head, p='fro')
            
            avg_norm = (q_norm + k_norm + v_norm) / 3
            head_importance.append(avg_norm.item())
        
        # Normalize importance scores
        max_importance = max(head_importance)
        if max_importance > 0:
            head_importance = [imp / max_importance for imp in head_importance]
        
        return head_importance
    
    def _compress_projection_by_heads(
        self,
        weight: Tensor,
        num_heads: int,
        preserve_heads: List[int]
    ) -> Dict[str, Any]:
        """Compress projection matrix with head-specific ranks."""
        from ..core.randomized_svd import randomized_svd
        
        d_model = weight.size(1)
        d_k = weight.size(0) // num_heads
        
        compressed_components = []
        
        for head_idx in range(num_heads):
            start_idx = head_idx * d_k
            end_idx = (head_idx + 1) * d_k
            
            head_weight = weight[start_idx:end_idx, :]
            
            # Determine rank for this head
            if head_idx in preserve_heads:
                # Use higher rank for important heads
                if isinstance(self.head_rank, float):
                    rank = max(1, int(min(head_weight.shape) * self.head_rank * 1.5))
                else:
                    rank = min(self.head_rank * 2, min(head_weight.shape))
            else:
                # Use lower rank for less important heads
                if isinstance(self.head_rank, float):
                    rank = max(1, int(min(head_weight.shape) * self.head_rank * 0.7))
                else:
                    rank = min(int(self.head_rank * 0.7), min(head_weight.shape))
            
            # Compress head
            U, s, Vt = randomized_svd(head_weight, rank)
            compressed_components.append({
                'U': U,
                's': s, 
                'Vt': Vt,
                'head_idx': head_idx,
                'rank': rank
            })
        
        return {
            'type': 'head_compressed',
            'components': compressed_components,
            'num_heads': num_heads,
            'original_shape': weight.shape
        }
    
    def _compress_output_projection(
        self,
        weight: Tensor,
        num_heads: int,
        preserve_heads: List[int]
    ) -> Dict[str, Any]:
        """Compress output projection with head awareness."""
        from ..core.randomized_svd import randomized_svd
        
        # For output projection, we can use standard SVD
        # but with rank adjusted based on preserved heads
        preservation_ratio = len(preserve_heads) / num_heads
        
        if isinstance(self.head_rank, float):
            adjusted_rank = max(1, int(min(weight.shape) * self.head_rank * (0.5 + preservation_ratio)))
        else:
            adjusted_rank = min(int(self.head_rank * (0.5 + preservation_ratio)), min(weight.shape))
        
        U, s, Vt = randomized_svd(weight, adjusted_rank)
        
        return {
            'type': 'standard_compressed',
            'U': U,
            's': s,
            'Vt': Vt,
            'rank': adjusted_rank,
            'original_shape': weight.shape
        }