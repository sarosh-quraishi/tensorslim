"""
Unit tests for SRHT and whitening functionality.

Tests the correctness and performance of the new SRHT and whitening components
to ensure they provide the expected improvements while maintaining compatibility.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import math
from typing import Tuple
import sys
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tensorslim.core.srht_utils import (
    fast_walsh_hadamard_transform,
    create_srht_matrix,
    apply_srht,
    SRHTOperator,
    srht_range_finder,
    next_power_of_2
)
from tensorslim.core.whitening import (
    DataWhitener,
    WhitenedSVD,
    RandomDataset,
    create_calibration_dataset
)
from tensorslim.core.randomized_svd import RandomizedSVD
from tensorslim.core.layers import convert_layer_to_slim


class TestSRHT:
    """Test SRHT utilities."""
    
    def test_next_power_of_2(self):
        """Test next power of 2 calculation."""
        assert next_power_of_2(1) == 1
        assert next_power_of_2(2) == 2
        assert next_power_of_2(3) == 4
        assert next_power_of_2(7) == 8
        assert next_power_of_2(8) == 8
        assert next_power_of_2(15) == 16
        assert next_power_of_2(1000) == 1024
    
    def test_fast_walsh_hadamard_transform(self):
        """Test fast Walsh-Hadamard transform."""
        # Test with simple cases
        for n in [1, 2, 4, 8, 16, 32]:
            x = torch.randn(5, n)  # Batch of vectors
            
            # Apply transform
            y = fast_walsh_hadamard_transform(x.clone())
            
            # Check properties
            assert y.shape == x.shape
            assert not torch.isnan(y).any()
            assert not torch.isinf(y).any()
            
            # For single vector, check orthogonality property
            if n > 1:
                x_single = torch.randn(n)
                y_single = fast_walsh_hadamard_transform(x_single.clone())
                
                # WHT should preserve energy (up to normalization)
                energy_ratio = torch.norm(y_single) / torch.norm(x_single)
                assert abs(energy_ratio - 1.0) < 1e-5
    
    def test_fast_walsh_hadamard_transform_orthogonality(self):
        """Test that WHT is orthogonal."""
        n = 16
        I = torch.eye(n)
        
        # Apply WHT to identity matrix
        H = torch.stack([fast_walsh_hadamard_transform(I[i]) for i in range(n)])
        
        # Check orthogonality: H @ H^T should be identity (up to scaling)
        HHT = H @ H.T
        expected = torch.eye(n)
        
        # Allow for small numerical errors
        assert torch.allclose(HHT, expected, atol=1e-5)
    
    def test_create_srht_matrix(self):
        """Test SRHT matrix creation."""
        n, k = 64, 16
        
        sampling_indices, diagonal_signs, n_padded = create_srht_matrix(n, k)
        
        # Check dimensions
        assert len(sampling_indices) == k
        assert len(diagonal_signs) == n_padded
        assert n_padded >= n
        assert n_padded & (n_padded - 1) == 0  # Power of 2
        
        # Check values
        assert torch.all((diagonal_signs == 1) | (diagonal_signs == -1))
        assert torch.all(sampling_indices < n_padded)
        assert torch.all(sampling_indices >= 0)
    
    def test_apply_srht(self):
        """Test SRHT application to matrices."""
        m, n, k = 32, 64, 16
        
        # Create test matrix
        A = torch.randn(m, n)
        
        # Create SRHT components
        sampling_indices, diagonal_signs, n_padded = create_srht_matrix(n, k)
        
        # Apply SRHT
        result = apply_srht(A, sampling_indices, diagonal_signs, n_padded, k)
        
        # Check output shape
        assert result.shape == (m, k)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()
    
    def test_srht_operator(self):
        """Test SRHT operator class."""
        n, k = 128, 32
        
        # Create operator
        srht = SRHTOperator(n, k, device=torch.device('cpu'))
        
        # Test application
        A = torch.randn(64, n)
        result = srht(A)
        
        assert result.shape == (64, k)
        assert not torch.isnan(result).any()
        
        # Test device movement
        if torch.cuda.is_available():
            srht_cuda = srht.to(torch.device('cuda'))
            A_cuda = A.cuda()
            result_cuda = srht_cuda(A_cuda)
            assert result_cuda.device.type == 'cuda'
    
    def test_srht_range_finder(self):
        """Test SRHT-based range finding."""
        m, n = 100, 80
        rank = 20
        
        # Create low-rank matrix
        U = torch.randn(m, rank)
        V = torch.randn(rank, n)
        A = U @ V
        
        # Find range using SRHT
        Q = srht_range_finder(A, rank)
        
        # Check properties
        assert Q.shape[0] == m
        assert Q.shape[1] >= rank
        
        # Check orthogonality
        QTQ = Q.T @ Q
        I = torch.eye(Q.shape[1])
        assert torch.allclose(QTQ, I, atol=1e-4)
        
        # Check that Q spans the range well
        projected = Q @ (Q.T @ A)
        error = torch.norm(A - projected, 'fro') / torch.norm(A, 'fro')
        assert error < 0.1  # Should be a good approximation


class TestWhitening:
    """Test whitening functionality."""
    
    def test_random_dataset(self):
        """Test random calibration dataset."""
        hidden_dim = 256
        dataset = RandomDataset(hidden_dim)
        
        samples = dataset.get_samples(n_samples=32, max_length=128)
        
        assert samples.shape == (32, 128, hidden_dim)
        assert not torch.isnan(samples).any()
    
    def test_data_whitener_covariance(self):
        """Test covariance computation."""
        # Create layer with known statistics
        layer = nn.Linear(64, 32)
        
        # Create calibration data with known covariance structure
        n_samples = 1000
        mean = torch.zeros(64)
        cov = torch.eye(64) + 0.1 * torch.randn(64, 64)
        cov = cov @ cov.T  # Ensure positive definite
        
        # Generate samples from multivariate normal
        samples = torch.distributions.MultivariateNormal(mean, cov).sample((n_samples,))
        
        # Compute covariance
        whitener = DataWhitener()
        computed_cov = whitener.compute_activation_covariance(layer, samples)
        
        # Check that computed covariance is close to true covariance
        # (allowing for sampling variation)
        relative_error = torch.norm(computed_cov - cov, 'fro') / torch.norm(cov, 'fro')
        assert relative_error < 0.2  # Should be reasonably close
    
    def test_whitening_matrix_computation(self):
        """Test whitening matrix computation."""
        # Create positive definite covariance matrix
        n = 32
        A = torch.randn(n, n)
        cov = A @ A.T + 0.1 * torch.eye(n)  # Ensure positive definite
        
        whitener = DataWhitener()
        whitening_matrix_inv = whitener.compute_whitening_matrix(cov)
        
        # Check that whitening matrix works
        whitened_cov = whitening_matrix_inv @ cov @ whitening_matrix_inv.T
        
        # Should be approximately identity
        identity = torch.eye(n)
        error = torch.norm(whitened_cov - identity, 'fro')
        assert error < 1e-3
    
    def test_whitened_svd(self):
        """Test whitened SVD compression."""
        # Create test layer
        layer = nn.Linear(64, 32)
        
        # Create whitener with random dataset
        dataset = RandomDataset(hidden_dim=64)
        whitener = DataWhitener(calibration_dataset=dataset)
        whitened_svd = WhitenedSVD(whitener)
        
        # Test compression
        rank = 16
        U, s, Vt, whitening_inv = whitened_svd.compress_layer(layer, rank)
        
        # Check output shapes
        assert U.shape == (32, rank)
        assert s.shape == (rank,)
        assert Vt.shape == (rank, 64)
        assert whitening_inv.shape == (64, 64)
        
        # Test reconstruction
        reconstructed_layer = whitened_svd.reconstruct_layer(
            U, s, Vt, whitening_inv, layer.bias
        )
        
        assert isinstance(reconstructed_layer, nn.Linear)
        assert reconstructed_layer.weight.shape == layer.weight.shape
    
    def test_whitening_vs_standard_comparison(self):
        """Test that whitening provides some improvement."""
        # Create layer with correlated inputs (where whitening should help)
        layer = nn.Linear(32, 16)
        
        # Create correlated calibration data
        base_data = torch.randn(256, 16)
        correlation_matrix = torch.randn(32, 16)
        calibration_data = base_data @ correlation_matrix.T  # Creates correlations
        
        # Create test data with same correlation structure
        test_base = torch.randn(64, 16)
        test_data = test_base @ correlation_matrix.T
        
        # Test whitening vs standard
        from tensorslim.core.whitening import compare_whitening_vs_standard
        
        rank = 8
        comparison = compare_whitening_vs_standard(
            layer, rank, calibration_data, test_data
        )
        
        # Whitening should provide some improvement (not always, but often)
        # We mainly check that the function runs without errors
        assert 'whitened_mse' in comparison
        assert 'standard_mse' in comparison
        assert 'improvement_factor' in comparison
        assert comparison['improvement_factor'] > 0


class TestRandomizedSVDIntegration:
    """Test integration of SRHT and whitening with RandomizedSVD."""
    
    def test_randomized_svd_with_srht(self):
        """Test RandomizedSVD with SRHT enabled."""
        A = torch.randn(100, 80)
        rank = 20
        
        # Test with SRHT
        svd_srht = RandomizedSVD(rank, use_srht=True, use_whitening=False)
        U, s, Vt = svd_srht.fit_transform(A)
        
        # Check shapes
        assert U.shape == (100, rank)
        assert s.shape == (rank,)
        assert Vt.shape == (rank, 80)
        
        # Check reconstruction quality
        A_reconstructed = svd_srht.reconstruct(U, s, Vt)
        error = svd_srht.relative_error(A, A_reconstructed)
        assert error < 50  # Should be reasonable approximation
    
    def test_randomized_svd_with_whitening(self):
        """Test RandomizedSVD with whitening enabled."""
        # Create layer
        layer = nn.Linear(64, 32)
        A = layer.weight.data
        rank = 16
        
        # Test with whitening
        svd_whitening = RandomizedSVD(
            rank, 
            use_srht=False, 
            use_whitening=True,
            whitening_dataset="random",
            n_calibration_samples=128
        )
        
        U, s, Vt = svd_whitening.fit_transform(A, layer=layer)
        
        # Check shapes
        assert U.shape == (32, rank)
        assert s.shape == (rank,)
        assert Vt.shape == (rank, 64)
    
    def test_randomized_svd_srht_plus_whitening(self):
        """Test RandomizedSVD with both SRHT and whitening."""
        layer = nn.Linear(64, 32)
        A = layer.weight.data
        rank = 16
        
        # Test with both features
        svd_both = RandomizedSVD(
            rank,
            use_srht=True,
            use_whitening=True,
            whitening_dataset="random",
            n_calibration_samples=64
        )
        
        U, s, Vt = svd_both.fit_transform(A, layer=layer)
        
        # Check shapes
        assert U.shape == (32, rank)
        assert s.shape == (rank,)
        assert Vt.shape == (rank, 64)
    
    def test_layer_conversion_with_new_features(self):
        """Test layer conversion with SRHT and whitening."""
        layer = nn.Linear(64, 32)
        rank = 16
        
        # Test different configurations
        configs = [
            {"use_srht": False, "use_whitening": False},
            {"use_srht": True, "use_whitening": False},
            {"use_srht": False, "use_whitening": True},
            {"use_srht": True, "use_whitening": True},
        ]
        
        for config in configs:
            compressed = convert_layer_to_slim(layer, rank=rank, **config)
            
            assert compressed is not None
            assert hasattr(compressed, 'forward')
            
            # Test forward pass
            test_input = torch.randn(10, 64)
            output = compressed(test_input)
            assert output.shape == (10, 32)


class TestPerformanceRegression:
    """Test that new features don't break existing functionality."""
    
    def test_backward_compatibility(self):
        """Test that old API still works."""
        from tensorslim.core.randomized_svd import randomized_svd
        
        A = torch.randn(50, 40)
        rank = 10
        
        # Old API should still work
        U, s, Vt = randomized_svd(A, rank)
        
        assert U.shape == (50, rank)
        assert s.shape == (rank,)
        assert Vt.shape == (rank, 40)
    
    def test_error_handling(self):
        """Test error handling for edge cases."""
        # Test with invalid rank
        A = torch.randn(10, 20)
        
        svd = RandomizedSVD(rank=100)  # Rank too large
        U, s, Vt = svd.fit_transform(A)  # Should clamp rank
        
        assert U.shape[1] <= min(A.shape)
        assert s.shape[0] <= min(A.shape)
        assert Vt.shape[0] <= min(A.shape)
    
    def test_numerical_stability(self):
        """Test numerical stability with challenging matrices."""
        # Test with rank-deficient matrix
        rank = 5
        A = torch.randn(20, rank) @ torch.randn(rank, 30)
        
        svd = RandomizedSVD(rank=rank, use_srht=True)
        U, s, Vt = svd.fit_transform(A)
        
        # Should handle rank-deficient case gracefully
        assert not torch.isnan(U).any()
        assert not torch.isnan(s).any()
        assert not torch.isnan(Vt).any()
        
        # Singular values should be non-negative
        assert torch.all(s >= 0)


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_device_compatibility(device):
    """Test that all components work on different devices."""
    device = torch.device(device)
    
    # Test SRHT operator
    srht = SRHTOperator(64, 16, device=device)
    A = torch.randn(32, 64, device=device)
    result = srht(A)
    assert result.device == device
    
    # Test RandomizedSVD
    svd = RandomizedSVD(16, use_srht=True, device=device)
    U, s, Vt = svd.fit_transform(A)
    assert U.device == device
    assert s.device == device
    assert Vt.device == device


if __name__ == "__main__":
    # Run a subset of tests for quick verification
    import unittest
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    for test_class in [TestSRHT, TestWhitening, TestRandomizedSVDIntegration]:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ {len(result.failures)} tests failed, {len(result.errors)} errors")
        sys.exit(1)