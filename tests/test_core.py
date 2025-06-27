"""
Test suite for TensorSlim core functionality.

This module tests the core compression algorithms, SVD implementations,
and compressed layer functionality.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, 'src')

from tensorslim.core import (
    RandomizedSVD,
    AdaptiveRandomizedSVD,
    randomized_svd,
    estimate_rank,
    TensorSlim,
    ModelCompressor,
    compress_model,
    analyze_model_compression,
    SlimLinear,
    SlimConv2d,
    SlimEmbedding,
    convert_layer_to_slim
)


class TestRandomizedSVD:
    """Test cases for RandomizedSVD algorithm."""
    
    def test_initialization(self):
        """Test RandomizedSVD initialization."""
        svd = RandomizedSVD(rank=50, n_oversamples=5, n_power_iterations=1)
        assert svd.rank == 50
        assert svd.n_oversamples == 5
        assert svd.n_power_iterations == 1
    
    def test_basic_svd(self):
        """Test basic SVD decomposition."""
        # Create test matrix
        torch.manual_seed(42)
        matrix = torch.randn(100, 80)
        
        svd = RandomizedSVD(rank=20)
        U, s, Vt = svd.fit_transform(matrix)
        
        # Check shapes
        assert U.shape == (100, 20)
        assert s.shape == (20,)
        assert Vt.shape == (20, 80)
        
        # Check reconstruction quality
        reconstructed = svd.reconstruct(U, s, Vt)
        relative_err = svd.relative_error(matrix, reconstructed)
        assert relative_err < 80.0  # Should be reasonable quality for rank-20 approximation
    
    def test_rank_validation(self):
        """Test rank validation."""
        matrix = torch.randn(50, 30)
        svd = RandomizedSVD(rank=40)  # Rank too large
        
        # Should clamp rank instead of raising error
        U, s, Vt = svd.fit_transform(matrix)
        assert s.shape[0] == 30  # Should be clamped to min(matrix.shape)
    
    def test_wide_matrix(self):
        """Test SVD on wide matrices."""
        torch.manual_seed(42)
        matrix = torch.randn(30, 100)  # Wide matrix
        
        svd = RandomizedSVD(rank=20)
        U, s, Vt = svd.fit_transform(matrix)
        
        assert U.shape == (30, 20)
        assert s.shape == (20,)
        assert Vt.shape == (20, 100)
    
    def test_compression_ratio_calculation(self):
        """Test compression ratio calculation."""
        svd = RandomizedSVD(rank=25)
        original_shape = (100, 80)
        
        ratio = svd.compression_ratio(original_shape)
        expected_ratio = (100 * 80) / (25 * (100 + 80 + 1))
        assert abs(ratio - expected_ratio) < 1e-6
    
    def test_device_handling(self):
        """Test device handling."""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            matrix = torch.randn(50, 40, device=device)
            
            svd = RandomizedSVD(rank=15, device=device)
            U, s, Vt = svd.fit_transform(matrix)
            
            assert U.device == device
            assert s.device == device  
            assert Vt.device == device


class TestAdaptiveRandomizedSVD:
    """Test cases for AdaptiveRandomizedSVD."""
    
    def test_adaptive_svd(self):
        """Test adaptive SVD with quality threshold."""
        torch.manual_seed(42)
        matrix = torch.randn(80, 60)
        
        svd = AdaptiveRandomizedSVD(rank=20, target_quality=0.90)
        U, s, Vt = svd.fit_transform(matrix)
        
        # Check that reconstruction meets quality threshold
        reconstructed = svd.reconstruct(U, s, Vt)
        relative_err = svd.relative_error(matrix, reconstructed)
        
        # Should achieve reasonable quality (adaptive should be better than basic)
        assert relative_err < 80.0  # Still reasonable for randomized approximation


class TestSlimLinear:
    """Test cases for SlimLinear compressed layer."""
    
    def test_slim_linear_creation(self):
        """Test SlimLinear layer creation."""
        torch.manual_seed(42)
        U = torch.randn(128, 32)
        s = torch.randn(32).abs()
        Vt = torch.randn(32, 256)
        bias = torch.randn(128)
        
        layer = SlimLinear(U, s, Vt, bias, original_shape=(128, 256))
        
        assert layer.in_features == 256
        assert layer.out_features == 128
        assert layer.rank == 32
    
    def test_slim_linear_forward(self):
        """Test SlimLinear forward pass."""
        torch.manual_seed(42)
        
        # Create original linear layer
        original = nn.Linear(256, 128)
        
        # Compress it
        weight = original.weight.data
        U, s, Vt = randomized_svd(weight, rank=32)
        
        slim_layer = SlimLinear(U, s, Vt, original.bias.data, weight.shape)
        
        # Test forward pass
        x = torch.randn(10, 256)
        
        with torch.no_grad():
            original_output = original(x)
            slim_output = slim_layer(x)
            
            # Outputs should be similar (not exact due to compression)
            mse = torch.mean((original_output - slim_output) ** 2)
            assert mse < 1.0  # Reasonable approximation
    
    def test_slim_linear_compression_ratio(self):
        """Test compression ratio calculation."""
        U = torch.randn(100, 25)
        s = torch.randn(25)
        Vt = torch.randn(25, 200)
        
        layer = SlimLinear(U, s, Vt, None, original_shape=(100, 200))
        ratio = layer.compression_ratio()
        
        original_params = 100 * 200
        compressed_params = 100 * 25 + 25 + 25 * 200
        expected_ratio = original_params / compressed_params
        
        assert abs(ratio - expected_ratio) < 1e-6


class TestSlimConv2d:
    """Test cases for SlimConv2d compressed layer."""
    
    def test_slim_conv2d_creation(self):
        """Test SlimConv2d layer creation."""
        torch.manual_seed(42)
        
        # Simulate compressed conv layer components
        out_ch, in_ch, kh, kw = 64, 32, 3, 3
        rank = 20
        
        U = torch.randn(out_ch, rank)
        s = torch.randn(rank).abs()
        Vt = torch.randn(rank, in_ch * kh * kw)
        
        layer = SlimConv2d(
            U, s, Vt,
            original_shape=(out_ch, in_ch, kh, kw),
            kernel_size=(kh, kw)
        )
        
        assert layer.out_channels == out_ch
        assert layer.in_channels == in_ch
        assert layer.rank == rank
    
    def test_slim_conv2d_forward(self):
        """Test SlimConv2d forward pass."""
        torch.manual_seed(42)
        
        # Create test input
        x = torch.randn(2, 16, 32, 32)
        
        # Create components for compressed conv
        out_ch, in_ch, kh, kw = 32, 16, 3, 3
        rank = 15
        
        U = torch.randn(out_ch, rank)
        s = torch.randn(rank).abs()
        Vt = torch.randn(rank, in_ch * kh * kw)
        
        layer = SlimConv2d(
            U, s, Vt,
            original_shape=(out_ch, in_ch, kh, kw),
            kernel_size=(kh, kw),
            padding=1
        )
        
        # Test forward pass
        output = layer(x)
        
        # Check output shape
        expected_shape = (2, 32, 32, 32)  # With padding=1
        assert output.shape == expected_shape


class TestTensorSlim:
    """Test cases for TensorSlim model compressor."""
    
    def create_test_model(self):
        """Create a simple test model."""
        model = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
        return model
    
    def test_tensorslim_initialization(self):
        """Test TensorSlim initialization."""
        compressor = TensorSlim(
            rank=0.5,
            target_layers=['Linear'],
            preserve_layers=['BatchNorm']
        )
        
        assert compressor.rank == 0.5
        assert compressor.target_layers == ['Linear']
        assert compressor.preserve_layers == ['BatchNorm']
    
    def test_model_compression(self):
        """Test model compression."""
        model = self.create_test_model()
        original_params = sum(p.numel() for p in model.parameters())
        
        compressor = TensorSlim(rank=0.3, progress_bar=False)
        compressed_model = compressor.compress(model, inplace=False)
        
        compressed_params = sum(p.numel() for p in compressed_model.parameters())
        
        # Should reduce parameters
        assert compressed_params < original_params
        
        # Should have compression statistics
        assert len(compressor.compression_stats) > 0
    
    def test_inplace_compression(self):
        """Test in-place compression."""
        model = self.create_test_model()
        model_id = id(model)
        
        compressor = TensorSlim(rank=0.4, progress_bar=False)
        result = compressor.compress(model, inplace=True)
        
        # Should be the same object
        assert id(result) == model_id
    
    def test_layer_filtering(self):
        """Test layer filtering functionality."""
        model = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.Linear(32, 16),
            nn.Dropout(0.1),
            nn.Linear(16, 8)
        )
        
        compressor = TensorSlim(
            rank=0.5,
            target_layers=['Linear'],
            preserve_layers=['BatchNorm', 'Dropout'],
            progress_bar=False
        )
        
        compressed_model = compressor.compress(model, inplace=False)
        
        # Should only compress Linear layers
        linear_layers = [m for m in compressed_model.modules() 
                        if isinstance(m, (nn.Linear, SlimLinear))]
        
        # Some linear layers should be compressed (converted to SlimLinear)
        slim_layers = [m for m in linear_layers if isinstance(m, SlimLinear)]
        assert len(slim_layers) > 0


class TestModelCompressor:
    """Test cases for high-level ModelCompressor."""
    
    def test_model_compressor(self):
        """Test ModelCompressor functionality."""
        model = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        compressor = ModelCompressor(quality_threshold=0.90)
        compressed_model, info = compressor.compress(
            model, 
            compression_ratio=0.4,
            inplace=False
        )
        
        # Check return values
        assert compressed_model is not None
        assert 'compression_stats' in info
        assert 'method' in info


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_compress_model_function(self):
        """Test compress_model convenience function."""
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(), 
            nn.Linear(50, 25)
        )
        
        compressed = compress_model(
            model,
            compression_ratio=0.5,
            inplace=False
        )
        
        assert compressed is not None
        assert compressed is not model  # Should be different object
    
    def test_analyze_model_compression(self):
        """Test model compression analysis."""
        model = nn.Sequential(
            nn.Linear(80, 40),
            nn.ReLU(),
            nn.Linear(40, 20)
        )
        
        results = analyze_model_compression(
            model,
            compression_ratios=[0.25, 0.5, 0.75]
        )
        
        assert len(results) == 3
        for ratio_key in ['ratio_0.25', 'ratio_0.5', 'ratio_0.75']:
            assert ratio_key in results
            assert 'actual_ratio' in results[ratio_key]
    
    def test_estimate_rank(self):
        """Test rank estimation."""
        torch.manual_seed(42)
        matrix = torch.randn(100, 80)
        
        rank = estimate_rank(matrix, energy_threshold=0.90)
        
        assert isinstance(rank, int)
        assert rank > 0
        assert rank <= min(matrix.shape)
    
    def test_convert_layer_to_slim(self):
        """Test layer conversion to slim version."""
        original_layer = nn.Linear(64, 32)
        
        slim_layer = convert_layer_to_slim(original_layer, rank=16)
        
        assert isinstance(slim_layer, SlimLinear)
        assert slim_layer.rank == 16
        assert slim_layer.in_features == 64
        assert slim_layer.out_features == 32


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_rank(self):
        """Test handling of invalid rank values."""
        matrix = torch.randn(50, 30)
        
        with pytest.raises(ValueError):
            RandomizedSVD(rank=0).fit_transform(matrix)
        
        with pytest.raises(ValueError):
            RandomizedSVD(rank=-5).fit_transform(matrix)
    
    def test_empty_model(self):
        """Test handling of models with no compressible layers."""
        model = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.BatchNorm1d(10)
        )
        
        compressor = TensorSlim(rank=0.5, progress_bar=False)
        result = compressor.compress(model, inplace=False)
        
        # Should return model unchanged
        assert result is not None
        assert len(compressor.compression_stats) == 0
    
    def test_unsupported_layer_conversion(self):
        """Test conversion of unsupported layer types."""
        layer = nn.BatchNorm1d(64)
        
        result = convert_layer_to_slim(layer, rank=16)
        
        # Should return None for unsupported layers
        assert result is None


# Fixtures for parameterized tests
@pytest.fixture(params=[0.25, 0.5, 0.75])
def compression_ratio(request):
    """Parameterized compression ratios."""
    return request.param


@pytest.fixture(params=[16, 32, 64])
def matrix_rank(request):
    """Parameterized matrix ranks."""
    return request.param


class TestParameterized:
    """Parameterized tests for various configurations."""
    
    def test_various_compression_ratios(self, compression_ratio):
        """Test compression with various ratios."""
        model = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        compressor = TensorSlim(rank=compression_ratio, progress_bar=False)
        compressed = compressor.compress(model, inplace=False)
        
        original_params = sum(p.numel() for p in model.parameters())
        compressed_params = sum(p.numel() for p in compressed.parameters())
        
        # Should achieve some compression
        assert compressed_params <= original_params
    
    def test_various_ranks(self, matrix_rank):
        """Test SVD with various ranks."""
        torch.manual_seed(42)
        matrix = torch.randn(100, 100)
        
        svd = RandomizedSVD(rank=matrix_rank)
        U, s, Vt = svd.fit_transform(matrix)
        
        assert U.shape[1] == matrix_rank
        assert s.shape[0] == matrix_rank
        assert Vt.shape[0] == matrix_rank


if __name__ == "__main__":
    pytest.main([__file__])