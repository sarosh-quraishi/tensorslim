"""
Subsampled Randomized Hadamard Transform (SRHT) utilities.

This module provides efficient implementations of SRHT for use in randomized SVD,
offering O(n log n) matrix-vector multiplication compared to O(n²) for Gaussian matrices.

The SRHT matrix has the form: Ω = √(n/k) · S · H · D where:
- S: random row sampling matrix (k × n)
- H: Hadamard matrix (n × n) 
- D: random diagonal matrix with ±1 entries (n × n)

References:
- Halko, Martinsson, Tropp (2011): "Finding structure with randomness"
- Ailon, Chazelle (2009): "The Fast Johnson-Lindenstrauss Transform"
"""

import torch
import torch.nn as nn
import math
import warnings
from typing import Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


def next_power_of_2(n: int) -> int:
    """Find the next power of 2 greater than or equal to n."""
    return 1 << (n - 1).bit_length()


def fast_walsh_hadamard_transform(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Fast Walsh-Hadamard Transform using the iterative algorithm.
    
    This implements the fast WHT in O(n log n) time complexity.
    The input tensor is modified in-place along the specified dimension.
    
    Args:
        x: Input tensor, last dimension must be a power of 2
        dim: Dimension along which to apply the transform
        
    Returns:
        Transformed tensor (same shape as input)
        
    Note:
        This function assumes the specified dimension has length that is a power of 2.
    """
    if dim != -1 and dim != x.dim() - 1:
        # Move the target dimension to the end for easier processing
        x = x.movedim(dim, -1)
        moved = True
    else:
        moved = False
    
    n = x.shape[-1]
    if n == 1:
        return x
        
    # Ensure n is a power of 2
    if n & (n - 1) != 0:
        raise ValueError(f"Transform dimension must be a power of 2, got {n}")
    
    # Iterative fast WHT algorithm
    step = 1
    while step < n:
        for i in range(0, n, step * 2):
            for j in range(step):
                u = x[..., i + j]
                v = x[..., i + j + step]
                x[..., i + j] = u + v
                x[..., i + j + step] = u - v
        step *= 2
    
    # Normalize by sqrt(n) to make it an orthogonal transform
    x = x / math.sqrt(n)
    
    if moved:
        x = x.movedim(-1, dim)
    
    return x


def create_srht_matrix(
    n: int, 
    k: int, 
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Create components for SRHT matrix Ω = √(n_padded/k) · S · H · D.
    
    Instead of forming the full matrix, we return the components for efficient
    matrix-vector multiplication.
    
    Args:
        n: Original dimension size
        k: Target dimension size (k <= n)
        device: Device to create tensors on
        dtype: Data type for tensors
        
    Returns:
        Tuple of (sampling_indices, diagonal_signs, padded_size)
        - sampling_indices: Random row indices for sampling matrix S (k,)
        - diagonal_signs: Random ±1 diagonal entries for matrix D (n_padded,)
        - padded_size: Size after padding to next power of 2
    """
    if k > n:
        raise ValueError(f"Target dimension k={k} cannot be larger than input dimension n={n}")
    
    # Pad to next power of 2 for efficient Hadamard transform
    n_padded = next_power_of_2(n)
    
    if device is None:
        device = torch.device('cpu')
    
    # Create random sampling indices (S matrix)
    sampling_indices = torch.randperm(n_padded, dtype=torch.long, device=device)[:k]
    
    # Create random diagonal signs (D matrix)
    diagonal_signs = torch.randint(0, 2, (n_padded,), dtype=dtype, device=device) * 2 - 1
    
    return sampling_indices, diagonal_signs, n_padded


def apply_srht(
    matrix: torch.Tensor,
    sampling_indices: torch.Tensor,
    diagonal_signs: torch.Tensor,
    n_padded: int,
    k: int
) -> torch.Tensor:
    """
    Apply SRHT transform to a matrix: result = √(n_padded/k) · S · H · D · matrix.
    
    This computes the transform efficiently without forming the full SRHT matrix.
    
    Args:
        matrix: Input matrix of shape (..., n, m)
        sampling_indices: Row sampling indices from create_srht_matrix
        diagonal_signs: Diagonal signs from create_srht_matrix  
        n_padded: Padded dimension size
        k: Target dimension size
        
    Returns:
        Transformed matrix of shape (..., k, m)
    """
    *batch_dims, n, m = matrix.shape
    device = matrix.device
    dtype = matrix.dtype
    
    # Step 1: Apply diagonal matrix D (element-wise multiplication)
    if n < n_padded:
        # Pad with zeros to n_padded
        padded_matrix = torch.zeros(*batch_dims, n_padded, m, dtype=dtype, device=device)
        padded_matrix[..., :n, :] = matrix
    else:
        padded_matrix = matrix
    
    # Apply diagonal signs
    padded_matrix = padded_matrix * diagonal_signs.unsqueeze(-1)
    
    # Step 2: Apply Hadamard matrix H (fast Walsh-Hadamard transform)
    # Apply WHT along the second-to-last dimension
    transformed_matrix = fast_walsh_hadamard_transform(padded_matrix, dim=-2)
    
    # Step 3: Apply sampling matrix S (select rows)
    sampled_matrix = transformed_matrix[..., sampling_indices, :]
    
    # Step 4: Apply scaling factor √(n_padded/k)
    scaling_factor = math.sqrt(n_padded / k)
    result = sampled_matrix * scaling_factor
    
    return result


class SRHTOperator:
    """
    SRHT operator that can be applied to matrices efficiently.
    
    This class encapsulates the SRHT components and provides methods
    for applying the transform to matrices.
    """
    
    def __init__(
        self,
        n: int,
        k: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        seed: Optional[int] = None
    ):
        """
        Initialize SRHT operator.
        
        Args:
            n: Input dimension size
            k: Output dimension size
            device: Device for computation
            dtype: Data type for computation
            seed: Random seed for reproducibility
        """
        if seed is not None:
            torch.manual_seed(seed)
            
        self.n = n
        self.k = k
        self.device = device or torch.device('cpu')
        self.dtype = dtype
        
        # Create SRHT components
        self.sampling_indices, self.diagonal_signs, self.n_padded = create_srht_matrix(
            n, k, device, dtype
        )
        
        logger.debug(f"Created SRHT operator: {n} -> {k} (padded to {self.n_padded})")
    
    def apply(self, matrix: torch.Tensor) -> torch.Tensor:
        """
        Apply SRHT transform to matrix.
        
        Args:
            matrix: Input matrix of shape (..., n, m)
            
        Returns:
            Transformed matrix of shape (..., k, m)
        """
        return apply_srht(
            matrix, 
            self.sampling_indices, 
            self.diagonal_signs, 
            self.n_padded, 
            self.k
        )
    
    def __call__(self, matrix: torch.Tensor) -> torch.Tensor:
        """Make the operator callable."""
        return self.apply(matrix)
    
    def to(self, device: torch.device) -> 'SRHTOperator':
        """Move operator to a different device."""
        self.device = device
        self.sampling_indices = self.sampling_indices.to(device)
        self.diagonal_signs = self.diagonal_signs.to(device)
        return self
    
    def memory_usage(self) -> int:
        """Return memory usage in bytes."""
        return (
            self.sampling_indices.numel() * self.sampling_indices.element_size() +
            self.diagonal_signs.numel() * self.diagonal_signs.element_size()
        )


def srht_range_finder(
    matrix: torch.Tensor,
    target_rank: int,
    oversampling: int = 10,
    n_iterations: int = 2
) -> torch.Tensor:
    """
    Find range of matrix using SRHT for randomized SVD.
    
    This replaces the Gaussian random matrix with SRHT for better
    computational efficiency.
    
    Args:
        matrix: Input matrix A of shape (m, n)
        target_rank: Target rank for range finding
        oversampling: Extra samples for stability
        n_iterations: Number of power iterations
        
    Returns:
        Q matrix of shape (m, target_rank + oversampling)
    """
    m, n = matrix.shape
    k = target_rank + oversampling
    
    # Create SRHT operator
    srht = SRHTOperator(n, k, device=matrix.device, dtype=matrix.dtype)
    
    # Apply SRHT to matrix: Y = A * Ω
    Y = matrix @ srht.apply(torch.eye(n, device=matrix.device, dtype=matrix.dtype))
    
    # Power iterations for better accuracy
    for _ in range(n_iterations):
        # Y = A * (A^T * Y)
        Y = matrix @ (matrix.T @ Y)
    
    # QR decomposition to get orthonormal basis
    Q, _ = torch.linalg.qr(Y, mode='reduced')
    
    return Q


def compare_srht_vs_gaussian(
    matrix_size: Tuple[int, int],
    target_rank: int,
    device: Optional[torch.device] = None,
    n_trials: int = 5
) -> dict:
    """
    Compare SRHT vs Gaussian random matrices for range finding.
    
    Args:
        matrix_size: (m, n) dimensions of test matrix
        target_rank: Target rank for comparison
        device: Device for computation
        n_trials: Number of trials for timing
        
    Returns:
        Dictionary with comparison results
    """
    if device is None:
        device = torch.device('cpu')
    
    m, n = matrix_size
    
    # Create test matrix
    torch.manual_seed(42)
    A = torch.randn(m, n, device=device)
    
    # Time SRHT method
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
    end_time = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
    
    if device.type == 'cuda':
        start_time.record()
    
    for _ in range(n_trials):
        Q_srht = srht_range_finder(A, target_rank)
    
    if device.type == 'cuda':
        end_time.record()
        torch.cuda.synchronize()
        srht_time = start_time.elapsed_time(end_time) / n_trials
    else:
        import time
        start = time.time()
        for _ in range(n_trials):
            Q_srht = srht_range_finder(A, target_rank)
        srht_time = (time.time() - start) * 1000 / n_trials  # Convert to ms
    
    # Time Gaussian method
    if device.type == 'cuda':
        start_time.record()
    else:
        start = time.time()
    
    for _ in range(n_trials):
        # Standard Gaussian range finder
        Omega = torch.randn(n, target_rank + 10, device=device)
        Y = A @ Omega
        Q_gaussian, _ = torch.linalg.qr(Y, mode='reduced')
    
    if device.type == 'cuda':
        end_time.record()
        torch.cuda.synchronize()
        gaussian_time = start_time.elapsed_time(end_time) / n_trials
    else:
        gaussian_time = (time.time() - start) * 1000 / n_trials
    
    # Compare approximation quality
    # Project matrix onto both ranges and compare reconstruction error
    A_proj_srht = Q_srht @ (Q_srht.T @ A)
    A_proj_gaussian = Q_gaussian @ (Q_gaussian.T @ A)
    
    error_srht = torch.norm(A - A_proj_srht, 'fro') / torch.norm(A, 'fro')
    error_gaussian = torch.norm(A - A_proj_gaussian, 'fro') / torch.norm(A, 'fro')
    
    # Memory usage comparison
    srht_op = SRHTOperator(n, target_rank + 10, device=device)
    srht_memory = srht_op.memory_usage()
    gaussian_memory = (target_rank + 10) * n * 4  # 4 bytes per float32
    
    return {
        'srht_time_ms': srht_time,
        'gaussian_time_ms': gaussian_time,
        'speedup': gaussian_time / srht_time,
        'srht_error': error_srht.item(),
        'gaussian_error': error_gaussian.item(),
        'error_ratio': error_srht.item() / error_gaussian.item(),
        'srht_memory_bytes': srht_memory,
        'gaussian_memory_bytes': gaussian_memory,
        'memory_ratio': gaussian_memory / srht_memory
    }