"""
Fast randomized SVD implementation for neural network compression.

Based on "Finding structure with randomness: Probabilistic algorithms for
constructing approximate matrix decompositions" by Halko, Martinsson, and Tropp (2011).
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np


class RandomizedSVD:
    """
    Fast randomized SVD for large matrices using the Halko-Martinsson-Tropp algorithm.
    
    This implementation is optimized for neural network weight matrices and provides
    10-100x speedup over standard SVD while maintaining high accuracy.
    
    Args:
        rank: Target rank for the low-rank approximation
        n_oversamples: Number of additional samples for stability (default: 10)
        n_power_iterations: Number of power iterations for accuracy (default: 2)
        random_state: Random seed for reproducibility
        device: Device to perform computations on
    """
    
    def __init__(
        self,
        rank: int,
        n_oversamples: int = 10,
        n_power_iterations: int = 2,
        random_state: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None
    ):
        self.rank = rank
        self.n_oversamples = n_oversamples
        self.n_power_iterations = n_power_iterations
        self.random_state = random_state
        self.device = device
        
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)
    
    def __call__(
        self, 
        matrix: Tensor, 
        rank: Optional[int] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute randomized SVD of input matrix.
        
        Args:
            matrix: Input matrix to decompose (M x N)
            rank: Override the default rank for this decomposition
            
        Returns:
            Tuple of (U, s, Vt) where:
            - U: Left singular vectors (M x rank)
            - s: Singular values (rank,)
            - Vt: Right singular vectors transposed (rank x N)
        """
        return self.fit_transform(matrix, rank)
    
    def fit_transform(
        self, 
        matrix: Tensor, 
        rank: Optional[int] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute randomized SVD decomposition.
        
        Args:
            matrix: Input matrix to decompose
            rank: Target rank (uses instance rank if None)
            
        Returns:
            Low-rank SVD decomposition (U, s, Vt)
        """
        if rank is None:
            rank = self.rank
            
        # Ensure matrix is on correct device
        if self.device is not None:
            matrix = matrix.to(self.device)
        
        # Validate inputs
        if rank <= 0:
            raise ValueError(f"Rank must be positive, got {rank}")
        if rank > min(matrix.shape):
            # Clamp rank to maximum possible instead of raising error
            rank = min(matrix.shape)
            print(f"Warning: Rank clamped to {rank} for matrix shape {matrix.shape}")
            
        # Choose algorithm based on matrix orientation
        if matrix.shape[0] >= matrix.shape[1]:
            return self._randomized_svd_tall(matrix, rank)
        else:
            return self._randomized_svd_wide(matrix, rank)
    
    def _randomized_svd_tall(
        self, 
        matrix: Tensor, 
        rank: int
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Randomized SVD for tall matrices (M >= N)."""
        m, n = matrix.shape
        l = min(rank + self.n_oversamples, n)
        
        # Stage A: Randomized range finding
        Q = self._randomized_range_finder(matrix, l)
        
        # Stage B: Compute SVD of projected matrix
        B = Q.T @ matrix  # (l x n)
        U_tilde, s, Vt = torch.linalg.svd(B, full_matrices=False)
        
        # Reconstruct U
        U = Q @ U_tilde
        
        # Return top-rank components
        return U[:, :rank], s[:rank], Vt[:rank, :]
    
    def _randomized_svd_wide(
        self, 
        matrix: Tensor, 
        rank: int
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Randomized SVD for wide matrices (M < N)."""
        # Transpose and use tall algorithm
        U_t, s, Vt_t = self._randomized_svd_tall(matrix.T, rank)
        
        # Transpose results back
        return Vt_t.T, s, U_t.T
    
    def _randomized_range_finder(
        self, 
        matrix: Tensor, 
        size: int
    ) -> Tensor:
        """
        Find an orthonormal basis for the range of matrix using randomized sampling.
        
        Args:
            matrix: Input matrix (M x N)
            size: Number of basis vectors to find
            
        Returns:
            Orthonormal matrix Q (M x size) spanning approximate range of matrix
        """
        m, n = matrix.shape
        
        # Generate random test matrix
        Omega = torch.randn(n, size, device=matrix.device, dtype=matrix.dtype)
        
        # Basic randomized range finding
        Y = matrix @ Omega
        
        # Power iterations for improved accuracy
        for _ in range(self.n_power_iterations):
            # Orthogonalize Y between power iterations for numerical stability
            Y = matrix @ (matrix.T @ Y)
            if Y.shape[1] > 1:
                Y, _ = torch.linalg.qr(Y, mode='reduced')
        
        # Orthogonalize using QR decomposition
        Q, _ = torch.linalg.qr(Y, mode='reduced')
        
        return Q
    
    def reconstruct(
        self, 
        U: Tensor, 
        s: Tensor, 
        Vt: Tensor
    ) -> Tensor:
        """
        Reconstruct matrix from SVD components.
        
        Args:
            U: Left singular vectors
            s: Singular values
            Vt: Right singular vectors (transposed)
            
        Returns:
            Reconstructed matrix
        """
        return U @ torch.diag(s) @ Vt
    
    def compression_ratio(
        self, 
        original_shape: Tuple[int, int], 
        rank: Optional[int] = None
    ) -> float:
        """
        Calculate compression ratio for given matrix shape and rank.
        
        Args:
            original_shape: Shape of original matrix (M, N)
            rank: Compression rank (uses instance rank if None)
            
        Returns:
            Compression ratio (original_size / compressed_size)
        """
        if rank is None:
            rank = self.rank
            
        m, n = original_shape
        original_size = m * n
        compressed_size = rank * (m + n + 1)  # U + s + Vt
        
        return original_size / compressed_size
    
    def relative_error(
        self, 
        original: Tensor, 
        reconstructed: Tensor
    ) -> float:
        """
        Calculate relative Frobenius norm error between matrices.
        
        Args:
            original: Original matrix
            reconstructed: Reconstructed matrix
            
        Returns:
            Relative error as a percentage
        """
        error_norm = torch.norm(original - reconstructed, p='fro')
        original_norm = torch.norm(original, p='fro')
        
        if original_norm == 0:
            return 0.0 if error_norm == 0 else float('inf')
            
        return (error_norm / original_norm).item() * 100
    
    def activation_quality(
        self,
        original_weight: Tensor,
        compressed_weight: Tensor,
        test_inputs: Tensor,
        layer_type: str = "linear"
    ) -> float:
        """
        Calculate quality based on activation similarity rather than weight reconstruction.
        
        Args:
            original_weight: Original weight matrix
            compressed_weight: Compressed weight matrix
            test_inputs: Test inputs for activation comparison
            layer_type: Type of layer ("linear", "conv2d", etc.)
            
        Returns:
            Quality score between 0 and 1 (1 = perfect preservation)
        """
        with torch.no_grad():
            if layer_type == "linear":
                # For linear layers: y = x @ W.T
                original_output = F.linear(test_inputs, original_weight)
                compressed_output = F.linear(test_inputs, compressed_weight)
            elif layer_type == "conv2d":
                # For conv layers: assume weight is in conv format
                original_output = F.conv2d(test_inputs, original_weight)
                compressed_output = F.conv2d(test_inputs, compressed_weight)
            else:
                # Fallback to matrix multiplication
                original_output = test_inputs @ original_weight.T
                compressed_output = test_inputs @ compressed_weight.T
            
            # Flatten outputs for comparison
            orig_flat = original_output.flatten()
            comp_flat = compressed_output.flatten()
            
            # Use cosine similarity as quality metric
            cosine_sim = F.cosine_similarity(orig_flat, comp_flat, dim=0)
            
            # Convert to quality score (handle NaN case)
            if torch.isnan(cosine_sim):
                return 0.0
            
            # Ensure positive quality score
            quality = (cosine_sim + 1.0) / 2.0  # Map [-1, 1] to [0, 1]
            return quality.item()


class AdaptiveRandomizedSVD(RandomizedSVD):
    """
    Adaptive randomized SVD that automatically selects optimal parameters.
    
    This variant automatically chooses the number of oversamples and power iterations
    based on the matrix properties and target quality.
    """
    
    def __init__(
        self,
        rank: int,
        target_quality: float = 0.95,
        max_oversamples: int = 20,
        max_power_iterations: int = 5,
        random_state: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None
    ):
        self.target_quality = target_quality
        self.max_oversamples = max_oversamples  
        self.max_power_iterations = max_power_iterations
        
        super().__init__(
            rank=rank,
            n_oversamples=10,  # Initial value, will be adapted
            n_power_iterations=2,  # Initial value, will be adapted
            random_state=random_state,
            device=device
        )
    
    def fit_transform(
        self, 
        matrix: Tensor, 
        rank: Optional[int] = None,
        test_inputs: Optional[Tensor] = None,
        layer_type: str = "linear"
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Adaptive SVD that optimizes parameters for target quality using activation-based measurement.
        
        Args:
            matrix: Weight matrix to compress
            rank: Target rank (uses instance rank if None)
            test_inputs: Test inputs for activation quality measurement
            layer_type: Type of layer for proper activation computation
        """
        if rank is None:
            rank = self.rank
            
        # Start with conservative parameters
        best_result = None
        best_quality = 0.0
        
        for n_over in range(5, self.max_oversamples + 1, 5):
            for n_power in range(1, self.max_power_iterations + 1):
                self.n_oversamples = n_over
                self.n_power_iterations = n_power
                
                # Compute SVD with current parameters
                U, s, Vt = super().fit_transform(matrix, rank)
                
                # Calculate quality based on activation similarity if test inputs provided
                if test_inputs is not None:
                    # Reconstruct compressed weight matrix
                    compressed_weight = self.reconstruct(U, s, Vt)
                    quality = self.activation_quality(
                        matrix, compressed_weight, test_inputs, layer_type
                    )
                else:
                    # Fallback to spectral quality (energy preservation)
                    # This is faster and still meaningful for neural networks
                    total_energy = torch.sum(s ** 2) if len(s) > rank else torch.norm(matrix, p='fro') ** 2
                    captured_energy = torch.sum(s[:rank] ** 2)
                    quality = (captured_energy / total_energy).item()
                
                if quality >= self.target_quality:
                    return U, s, Vt
                    
                if quality > best_quality:
                    best_quality = quality
                    best_result = (U, s, Vt)
        
        # Return best result if target not achieved
        if best_result is not None:
            return best_result
        else:
            # Fallback to standard parameters
            self.n_oversamples = 10
            self.n_power_iterations = 2
            return super().fit_transform(matrix, rank)


def randomized_svd(
    matrix: Tensor,
    rank: int,
    n_oversamples: int = 10,
    n_power_iterations: int = 2,
    random_state: Optional[int] = None
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Convenience function for randomized SVD.
    
    Args:
        matrix: Input matrix to decompose
        rank: Target rank for decomposition
        n_oversamples: Number of additional samples for stability
        n_power_iterations: Number of power iterations for accuracy
        random_state: Random seed for reproducibility
        
    Returns:
        SVD decomposition (U, s, Vt)
    """
    svd = RandomizedSVD(
        rank=rank,
        n_oversamples=n_oversamples,
        n_power_iterations=n_power_iterations,
        random_state=random_state
    )
    
    return svd.fit_transform(matrix)


def estimate_rank(
    matrix: Tensor, 
    energy_threshold: float = 0.95
) -> int:
    """
    Estimate optimal rank to preserve given energy threshold.
    
    Args:
        matrix: Input matrix
        energy_threshold: Fraction of energy to preserve (0-1)
        
    Returns:
        Estimated optimal rank
    """
    # Use a small randomized SVD to estimate spectrum
    max_rank = min(matrix.shape) // 4
    max_rank = max(10, max_rank)  # At least 10 components
    
    U, s, Vt = randomized_svd(matrix, max_rank)
    
    # Calculate cumulative energy
    s_squared = s ** 2
    total_energy = s_squared.sum()
    cumulative_energy = torch.cumsum(s_squared, dim=0)
    energy_ratios = cumulative_energy / total_energy
    
    # Find rank that preserves target energy
    rank_idx = torch.where(energy_ratios >= energy_threshold)[0]
    
    if len(rank_idx) == 0:
        return max_rank
    else:
        return rank_idx[0].item() + 1