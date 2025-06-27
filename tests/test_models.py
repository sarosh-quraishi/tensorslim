"""
Test suite for TensorSlim model-specific functionality.

This module tests transformer compression and model analysis functionality.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, 'src')

from tensorslim.models import (
    TransformerSlim,
    AttentionSlim,
    BaseModelCompressor,
    LayerAnalyzer,
    CompressionStrategy,
    ModelProfiler
)


class SimpleTransformer(nn.Module):
    """Simple transformer model for testing."""
    
    def __init__(self, d_model=256, num_heads=8, ff_dim=512):
        super().__init__()
        self.d_model = d_model
        
        # Attention layers
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.attention_output = nn.Linear(d_model, d_model)
        
        # Feed-forward layers
        self.ffn_1 = nn.Linear(d_model, ff_dim)
        self.ffn_2 = nn.Linear(ff_dim, d_model)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Output layer
        self.output = nn.Linear(d_model, 10)
    
    def forward(self, x):
        # Simplified transformer forward pass
        attn = self.attention_output(self.value(x))
        x = self.norm1(x + attn)
        
        ff = self.ffn_2(torch.relu(self.ffn_1(x)))
        x = self.norm2(x + ff)
        
        return self.output(x)


class TestTransformerSlim:
    """Test cases for TransformerSlim."""
    
    def create_test_transformer(self):
        """Create a test transformer model."""
        return SimpleTransformer(d_model=128, ff_dim=256)
    
    def test_transformer_slim_init(self):
        """Test TransformerSlim initialization."""
        compressor = TransformerSlim(
            attention_rank=32,
            ffn_rank=64,
            output_rank=128,
            preserve_embeddings=True,
            preserve_layernorm=True
        )
        
        assert compressor.attention_rank == 32
        assert compressor.ffn_rank == 64
        assert compressor.output_rank == 128
        assert compressor.preserve_embeddings is True
        assert compressor.preserve_layernorm is True
    
    def test_transformer_structure_analysis(self):
        """Test transformer structure analysis."""
        model = self.create_test_transformer()
        compressor = TransformerSlim(attention_rank=16, ffn_rank=32)
        
        structure = compressor._analyze_transformer_structure(model)
        
        assert 'layers' in structure
        assert 'total_layers' in structure
        assert 'attention_layers' in structure
        assert 'ffn_layers' in structure
        
        # Should detect attention and FFN layers
        assert structure['attention_layers'] > 0
        assert structure['ffn_layers'] > 0
    
    def test_layer_type_inference(self):
        """Test layer type inference."""
        compressor = TransformerSlim(attention_rank=16, ffn_rank=32)
        
        # Test attention layer
        attention_layer = nn.Linear(256, 256)
        layer_type = compressor._infer_layer_type(attention_layer)
        assert layer_type in ['attention', 'linear']
        
        # Test FFN layer (expansion)
        ffn_layer = nn.Linear(256, 1024)
        layer_type = compressor._infer_layer_type(ffn_layer)
        assert layer_type == 'ffn'
    
    def test_transformer_compression(self):
        """Test transformer model compression."""
        model = self.create_test_transformer()
        original_params = sum(p.numel() for p in model.parameters())
        
        compressor = TransformerSlim(
            attention_rank=16,
            ffn_rank=32,
            preserve_layernorm=True,
            progress_bar=False
        )
        
        compressed_model = compressor.compress(model, inplace=False)
        compressed_params = sum(p.numel() for p in compressed_model.parameters())
        
        # Should reduce parameters
        assert compressed_params < original_params
        
        # Should have compression statistics
        assert len(compressor.compression_stats) > 0
    
    def test_layer_preservation(self):
        """Test layer preservation functionality."""
        model = self.create_test_transformer()
        
        compressor = TransformerSlim(
            attention_rank=16,
            ffn_rank=32,
            preserve_layernorm=True,
            progress_bar=False
        )
        
        original_norms = [
            (name, module) for name, module in model.named_modules()
            if isinstance(module, nn.LayerNorm)
        ]
        
        compressed_model = compressor.compress(model, inplace=False)
        
        compressed_norms = [
            (name, module) for name, module in compressed_model.named_modules()
            if isinstance(module, nn.LayerNorm)
        ]
        
        # LayerNorm layers should be preserved
        assert len(original_norms) == len(compressed_norms)
    
    def test_rank_calculation(self):
        """Test compression rank calculation."""
        compressor = TransformerSlim(
            attention_rank=0.5,  # Ratio
            ffn_rank=64,         # Absolute
            output_rank=0.75     # Ratio
        )
        
        # Test attention layer (ratio)
        attention_layer = nn.Linear(256, 256)
        rank = compressor._get_layer_rank('attention', attention_layer, 0, 1)
        assert rank == 128  # 256 * 0.5
        
        # Test FFN layer (absolute)
        ffn_layer = nn.Linear(256, 512)
        rank = compressor._get_layer_rank('ffn', ffn_layer, 0, 1)
        assert rank == 64
        
        # Test output layer (ratio)
        output_layer = nn.Linear(128, 10)
        rank = compressor._get_layer_rank('output', output_layer, 0, 1)
        assert rank == 7  # min(128, 10) * 0.75 = 7


class TestAttentionSlim:
    """Test cases for AttentionSlim."""
    
    def test_attention_slim_init(self):
        """Test AttentionSlim initialization."""
        compressor = AttentionSlim(
            num_heads=8,
            head_rank=0.5,
            preserve_heads=[0, 1],
            head_importance_threshold=0.1
        )
        
        assert compressor.num_heads == 8
        assert compressor.head_rank == 0.5
        assert compressor.preserve_heads == [0, 1]
        assert compressor.head_importance_threshold == 0.1
    
    def test_head_importance_analysis(self):
        """Test attention head importance analysis."""
        torch.manual_seed(42)
        
        d_model = 256
        num_heads = 8
        d_k = d_model // num_heads
        
        # Create attention weight matrices
        query_weight = torch.randn(d_model, d_model)
        key_weight = torch.randn(d_model, d_model)
        value_weight = torch.randn(d_model, d_model)
        
        compressor = AttentionSlim(num_heads=num_heads, head_rank=0.5)
        importance = compressor._analyze_head_importance(
            query_weight, key_weight, value_weight
        )
        
        assert len(importance) == num_heads
        assert all(0 <= imp <= 1 for imp in importance)
    
    def test_attention_compression(self):
        """Test attention weight compression."""
        torch.manual_seed(42)
        
        d_model = 128
        num_heads = 4
        
        # Create attention weights
        query_weight = torch.randn(d_model, d_model)
        key_weight = torch.randn(d_model, d_model)
        value_weight = torch.randn(d_model, d_model)
        output_weight = torch.randn(d_model, d_model)
        
        compressor = AttentionSlim(num_heads=num_heads, head_rank=0.5)
        
        compressed_q, compressed_k, compressed_v, compressed_o = compressor.compress_attention_weights(
            query_weight, key_weight, value_weight, output_weight
        )
        
        # Should return compressed components
        assert compressed_q['type'] == 'head_compressed'
        assert compressed_k['type'] == 'head_compressed'
        assert compressed_v['type'] == 'head_compressed'
        assert compressed_o['type'] == 'standard_compressed'
        
        # Should have correct number of head components
        assert len(compressed_q['components']) == num_heads


class TestLayerAnalyzer:
    """Test cases for LayerAnalyzer."""
    
    def test_linear_layer_analysis(self):
        """Test analysis of linear layers."""
        layer = nn.Linear(256, 128, bias=True)
        info = LayerAnalyzer.get_layer_info(layer)
        
        assert info['type'] == 'Linear'
        assert info['in_features'] == 256
        assert info['out_features'] == 128
        assert info['has_bias'] is True
        assert info['compressible'] is True
        assert info['compression_potential'] > 0
    
    def test_conv2d_layer_analysis(self):
        """Test analysis of Conv2d layers."""
        layer = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False)
        info = LayerAnalyzer.get_layer_info(layer)
        
        assert info['type'] == 'Conv2d'
        assert info['in_channels'] == 16
        assert info['out_channels'] == 32
        assert info['kernel_size'] == (3, 3)
        assert info['has_bias'] is False
        assert info['compressible'] is True
    
    def test_embedding_layer_analysis(self):
        """Test analysis of embedding layers."""
        layer = nn.Embedding(1000, 256, padding_idx=0)
        info = LayerAnalyzer.get_layer_info(layer)
        
        assert info['type'] == 'Embedding'
        assert info['num_embeddings'] == 1000
        assert info['embedding_dim'] == 256
        assert info['padding_idx'] == 0
        assert info['compressible'] is True
    
    def test_unsupported_layer_analysis(self):
        """Test analysis of unsupported layers."""
        layer = nn.ReLU()
        info = LayerAnalyzer.get_layer_info(layer)
        
        assert info['type'] == 'ReLU'
        assert info['compressible'] is False
        assert info['compression_potential'] == 0.0
    
    def test_layer_grouping(self):
        """Test layer grouping functionality."""
        model = SimpleTransformer(d_model=128)
        groups = LayerAnalyzer.identify_layer_groups(model)
        
        assert 'attention' in groups
        assert 'feedforward' in groups
        assert 'normalization' in groups
        assert 'output' in groups
        
        # Should categorize layers correctly
        assert len(groups['attention']) > 0
        assert len(groups['feedforward']) > 0
        assert len(groups['normalization']) > 0


class TestCompressionStrategy:
    """Test cases for CompressionStrategy."""
    
    def test_strategy_initialization(self):
        """Test CompressionStrategy initialization."""
        strategy = CompressionStrategy(
            base_compression_ratio=0.5,
            layer_specific_ratios={'attention': 0.6, 'ffn': 0.4},
            quality_threshold=0.95,
            preserve_patterns=['norm', 'bias']
        )
        
        assert strategy.base_compression_ratio == 0.5
        assert strategy.layer_specific_ratios['attention'] == 0.6
        assert strategy.quality_threshold == 0.95
        assert 'norm' in strategy.preserve_patterns
    
    def test_layer_compression_ratio(self):
        """Test layer-specific compression ratio calculation."""
        strategy = CompressionStrategy(
            base_compression_ratio=0.5,
            layer_specific_ratios={'attention': 0.7}
        )
        
        # Test attention layer override
        layer_info = {'compression_potential': 0.8}
        ratio = strategy.get_layer_compression_ratio('model.attention.query', layer_info)
        assert ratio == 0.7
        
        # Test preserved layer
        ratio = strategy.get_layer_compression_ratio('model.norm1', layer_info)
        assert ratio is None  # Should be preserved
        
        # Test regular layer
        layer_info = {'compression_potential': 0.6}
        ratio = strategy.get_layer_compression_ratio('model.linear', layer_info)
        expected = 0.5 * (0.5 + 0.5 * 0.6)  # base * (0.5 + 0.5 * potential)
        assert abs(ratio - expected) < 1e-6
    
    def test_compression_plan_creation(self):
        """Test compression plan creation."""
        model = SimpleTransformer(d_model=64)
        strategy = CompressionStrategy(base_compression_ratio=0.5)
        
        plan = strategy.create_compression_plan(model)
        
        # Should have entries for compressible layers
        assert len(plan) > 0
        
        for layer_name, config in plan.items():
            assert 'compression_ratio' in config
            assert 'layer_info' in config
            assert 'method' in config
            assert 'quality_threshold' in config


class TestModelProfiler:
    """Test cases for ModelProfiler."""
    
    def test_model_profiling(self):
        """Test model profiling functionality."""
        model = SimpleTransformer(d_model=128)
        profile = ModelProfiler.profile_model(model)
        
        assert 'total_parameters' in profile
        assert 'trainable_parameters' in profile
        assert 'model_size_mb' in profile
        assert 'layer_count' in profile
        assert 'linear_layers' in profile
        
        # Basic sanity checks
        assert profile['total_parameters'] > 0
        assert profile['model_size_mb'] > 0
        assert profile['linear_layers'] > 0
    
    def test_performance_profiling(self):
        """Test performance profiling with sample input."""
        model = SimpleTransformer(d_model=64)
        sample_input = torch.randn(1, 32, 64)
        
        profile = ModelProfiler.profile_model(model, sample_input)
        
        assert 'avg_inference_time_ms' in profile
        assert 'min_inference_time_ms' in profile
        assert 'max_inference_time_ms' in profile
        
        # Performance metrics should be positive
        assert profile['avg_inference_time_ms'] > 0
        assert profile['min_inference_time_ms'] > 0
    
    def test_model_size_calculation(self):
        """Test model size calculation."""
        model = nn.Linear(100, 50)  # 100*50 + 50 = 5050 parameters
        
        size_mb = ModelProfiler._calculate_model_size(model)
        
        # Should be reasonable size
        assert size_mb > 0
        assert size_mb < 1  # Should be less than 1MB for small model


class TestBaseModelCompressor:
    """Test cases for BaseModelCompressor."""
    
    def test_base_compressor_interface(self):
        """Test BaseModelCompressor interface."""
        # BaseModelCompressor is abstract, so we can't instantiate it directly
        # But we can test that it defines the required interface
        
        from tensorslim.models.base import BaseModelCompressor
        
        # Check that abstract methods are defined
        assert hasattr(BaseModelCompressor, 'compress')
        assert hasattr(BaseModelCompressor, 'analyze_model')
        
        # Check that it has the expected initialization parameters
        class TestCompressor(BaseModelCompressor):
            def compress(self, model, inplace=False):
                return model
            
            def analyze_model(self, model):
                return {}
        
        compressor = TestCompressor(
            compression_ratio=0.5,
            quality_threshold=0.95
        )
        
        assert compressor.compression_ratio == 0.5
        assert compressor.quality_threshold == 0.95
    
    def test_compression_benefit_estimation(self):
        """Test compression benefit estimation."""
        class TestCompressor(BaseModelCompressor):
            def compress(self, model, inplace=False):
                return model
            
            def analyze_model(self, model):
                total_params = sum(p.numel() for p in model.parameters())
                return {
                    'total_parameters': total_params,
                    'compressible_parameters': int(total_params * 0.8),  # 80% compressible
                    'model_size_mb': total_params * 4 / (1024 * 1024)  # 4 bytes per param
                }
        
        model = nn.Linear(1000, 500)  # 500k + 500 parameters
        compressor = TestCompressor(compression_ratio=0.5)
        
        benefits = compressor.estimate_compression_benefit(model)
        
        assert 'parameter_reduction' in benefits
        assert 'memory_savings_mb' in benefits
        assert 'estimated_compression_ratio' in benefits
        
        # Should show significant benefits
        assert benefits['parameter_reduction'] > 0
        assert benefits['memory_savings_mb'] > 0


# Fixtures for parameterized tests
@pytest.fixture(params=[32, 64, 128])
def model_dimension(request):
    """Parameterized model dimensions."""
    return request.param


@pytest.fixture(params=[0.25, 0.5, 0.75])
def compression_ratio(request):
    """Parameterized compression ratios."""
    return request.param


class TestParameterizedModelCompression:
    """Parameterized tests for model compression."""
    
    def test_various_model_sizes(self, model_dimension):
        """Test compression with various model sizes."""
        model = SimpleTransformer(d_model=model_dimension)
        
        compressor = TransformerSlim(
            attention_rank=model_dimension // 4,
            ffn_rank=model_dimension // 2,
            progress_bar=False
        )
        
        compressed = compressor.compress(model, inplace=False)
        
        original_params = sum(p.numel() for p in model.parameters())
        compressed_params = sum(p.numel() for p in compressed.parameters())
        
        assert compressed_params < original_params
    
    def test_various_compression_ratios(self, compression_ratio):
        """Test compression with various ratios."""
        model = SimpleTransformer(d_model=128)
        
        compressor = TransformerSlim(
            attention_rank=compression_ratio,
            ffn_rank=compression_ratio,
            progress_bar=False
        )
        
        compressed = compressor.compress(model, inplace=False)
        
        original_params = sum(p.numel() for p in model.parameters())
        compressed_params = sum(p.numel() for p in compressed.parameters())
        
        # More aggressive compression should yield smaller models
        assert compressed_params <= original_params


if __name__ == "__main__":
    pytest.main([__file__])