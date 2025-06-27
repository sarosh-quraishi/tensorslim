"""
Truncation-aware data whitening for improved SVD compression.

This module implements the whitening technique from the SVD-LLM paper, which
uses calibration data to compute a whitening transformation that makes 
truncation decisions more accurate by aligning singular values with actual
compression loss.

References:
- Wang et al. (2025): "SVD-LLM: Truncation-aware Singular Value Decomposition 
  for Large Language Model Compression"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Dict, List, Tuple, Iterator
import logging
from tqdm import tqdm
import warnings
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class CalibrationDataset(ABC):
    """Abstract base class for calibration datasets."""
    
    @abstractmethod
    def get_samples(self, n_samples: int, max_length: int = 512) -> torch.Tensor:
        """
        Get calibration samples.
        
        Args:
            n_samples: Number of samples to return
            max_length: Maximum sequence length
            
        Returns:
            Tensor of shape (n_samples, seq_len, hidden_dim) or (n_samples, hidden_dim)
        """
        pass


class WikiTextDataset(CalibrationDataset):
    """WikiText-2 dataset for calibration."""
    
    def __init__(
        self, 
        tokenizer_name: str = "gpt2",
        device: Optional[torch.device] = None
    ):
        """
        Initialize WikiText dataset.
        
        Args:
            tokenizer_name: Name of tokenizer to use
            device: Device for tensors
        """
        self.device = device or torch.device('cpu')
        self.tokenizer_name = tokenizer_name
        self._data = None
        
    def _load_data(self):
        """Load and cache the dataset."""
        if self._data is not None:
            return
            
        try:
            from datasets import load_dataset
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "Please install transformers and datasets: "
                "pip install transformers datasets"
            )
        
        # Load dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Tokenize texts
        texts = [item['text'] for item in dataset if len(item['text'].strip()) > 0]
        
        logger.info(f"Loaded {len(texts)} text samples from WikiText-2")
        self._data = texts
        self._tokenizer = tokenizer
        
    def get_samples(self, n_samples: int, max_length: int = 512) -> torch.Tensor:
        """Get tokenized samples from WikiText."""
        self._load_data()
        
        # Sample texts
        import random
        sampled_texts = random.sample(self._data, min(n_samples, len(self._data)))
        
        # Tokenize
        encoded = self._tokenizer(
            sampled_texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        return encoded['input_ids'].to(self.device)


class RandomDataset(CalibrationDataset):
    """Random Gaussian data for calibration (fallback)."""
    
    def __init__(
        self, 
        hidden_dim: int,
        device: Optional[torch.device] = None
    ):
        """
        Initialize random dataset.
        
        Args:
            hidden_dim: Hidden dimension size
            device: Device for tensors
        """
        self.hidden_dim = hidden_dim
        self.device = device or torch.device('cpu')
        
    def get_samples(self, n_samples: int, max_length: int = 512) -> torch.Tensor:
        """Generate random Gaussian samples."""
        return torch.randn(
            n_samples, max_length, self.hidden_dim, 
            device=self.device
        )


class DataWhitener:
    """
    Truncation-aware data whitening for SVD compression.
    
    This class computes a whitening transformation using calibration data
    that makes singular values more directly correlated with compression loss.
    """
    
    def __init__(
        self,
        calibration_dataset: Optional[CalibrationDataset] = None,
        n_calibration_samples: int = 256,
        max_sequence_length: int = 512,
        regularization: float = 1e-6,
        device: Optional[torch.device] = None
    ):
        """
        Initialize data whitener.
        
        Args:
            calibration_dataset: Dataset for calibration
            n_calibration_samples: Number of calibration samples
            max_sequence_length: Maximum sequence length for samples
            regularization: Regularization for Cholesky decomposition
            device: Device for computation
        """
        self.calibration_dataset = calibration_dataset
        self.n_calibration_samples = n_calibration_samples
        self.max_sequence_length = max_sequence_length
        self.regularization = regularization
        self.device = device or torch.device('cpu')
        
        # Cached whitening matrices
        self._whitening_cache: Dict[Tuple[int, int], torch.Tensor] = {}
        
    def compute_activation_covariance(
        self,
        layer: nn.Linear,
        calibration_data: torch.Tensor,
        batch_size: int = 32
    ) -> torch.Tensor:
        """
        Compute input activation covariance matrix for a layer.
        
        Args:
            layer: Linear layer to analyze
            calibration_data: Calibration input data
            batch_size: Batch size for processing
            
        Returns:
            Covariance matrix of shape (input_dim, input_dim)
        """
        input_dim = layer.in_features
        
        # Collect activations
        activations = []
        
        # Process data in batches
        n_samples = calibration_data.shape[0]
        
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                batch = calibration_data[i:i+batch_size]
                
                # Flatten if needed (for sequence data)
                if batch.dim() == 3:  # (batch, seq, dim)
                    batch = batch.view(-1, batch.shape[-1])
                elif batch.dim() == 2 and batch.shape[-1] != input_dim:
                    # This might be token IDs - skip for now
                    logger.warning("Skipping batch with incompatible dimensions")
                    continue
                
                # Only keep samples that match input dimension
                if batch.shape[-1] == input_dim:
                    activations.append(batch)
        
        if not activations:
            logger.warning("No compatible activations found, using identity covariance")
            return torch.eye(input_dim, device=self.device)
            
        # Concatenate all activations
        all_activations = torch.cat(activations, dim=0)
        
        logger.debug(f"Computing covariance from {all_activations.shape[0]} activation samples")
        
        # Compute empirical covariance
        # Center the data
        mean_activation = all_activations.mean(dim=0, keepdim=True)
        centered_activations = all_activations - mean_activation
        
        # Covariance matrix
        n_samples = centered_activations.shape[0]
        covariance = (centered_activations.T @ centered_activations) / (n_samples - 1)
        
        # Add regularization for numerical stability
        covariance += self.regularization * torch.eye(input_dim, device=self.device)
        
        return covariance
    
    def compute_whitening_matrix(
        self,
        covariance: torch.Tensor,
        cache_key: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """
        Compute whitening matrix from covariance using Cholesky decomposition.
        
        Args:
            covariance: Covariance matrix
            cache_key: Optional cache key for storing result
            
        Returns:
            Whitening matrix S such that X_whitened = S^(-1) @ X
        """
        if cache_key is not None and cache_key in self._whitening_cache:
            return self._whitening_cache[cache_key]
        
        try:
            # Cholesky decomposition: Σ = L @ L^T
            L = torch.linalg.cholesky(covariance)
            
            # Whitening matrix is the inverse of L
            # S^(-1) = L^(-1), so that S^(-1) @ Σ @ S^(-T) = I
            whitening_matrix = torch.linalg.inv(L)
            
        except RuntimeError as e:
            logger.warning(f"Cholesky decomposition failed: {e}, using eigendecomposition")
            
            # Fallback to eigendecomposition
            eigenvals, eigenvecs = torch.linalg.eigh(covariance)
            
            # Ensure positive eigenvalues
            eigenvals = torch.clamp(eigenvals, min=self.regularization)
            
            # Whitening matrix: S^(-1) = U @ diag(λ^(-1/2)) @ U^T
            sqrt_inv_eigenvals = torch.diag(1.0 / torch.sqrt(eigenvals))
            whitening_matrix = eigenvecs @ sqrt_inv_eigenvals @ eigenvecs.T
        
        if cache_key is not None:
            self._whitening_cache[cache_key] = whitening_matrix
            
        return whitening_matrix
    
    def whiten_layer(
        self,
        layer: nn.Linear,
        calibration_data: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute whitening transformation for a specific layer.
        
        Args:
            layer: Linear layer to whiten
            calibration_data: Optional calibration data. If None, uses dataset.
            
        Returns:
            Tuple of (whitened_weight, whitening_matrix)
            - whitened_weight: W @ S for SVD compression
            - whitening_matrix: S^(-1) for transforming inputs
        """
        input_dim = layer.in_features
        output_dim = layer.out_features
        
        # Get calibration data
        if calibration_data is None:
            if self.calibration_dataset is None:
                logger.warning("No calibration data available, using identity whitening")
                identity = torch.eye(input_dim, device=self.device)
                return layer.weight.data, identity
            
            calibration_data = self.calibration_dataset.get_samples(
                self.n_calibration_samples,
                self.max_sequence_length
            )
        
        # Compute activation covariance
        cache_key = (input_dim, calibration_data.shape[0])
        
        covariance = self.compute_activation_covariance(layer, calibration_data)
        
        # Compute whitening matrix
        whitening_matrix_inv = self.compute_whitening_matrix(covariance, cache_key)
        
        # The whitening matrix we return is S^(-1) for transforming inputs
        # The weight matrix becomes W @ S (where S = inv(whitening_matrix_inv))
        whitening_matrix = torch.linalg.inv(whitening_matrix_inv)
        whitened_weight = layer.weight.data @ whitening_matrix
        
        logger.debug(f"Whitened layer {input_dim}→{output_dim}")
        
        return whitened_weight, whitening_matrix_inv
    
    def clear_cache(self):
        """Clear the whitening matrix cache."""
        self._whitening_cache.clear()


class WhitenedSVD:
    """
    SVD compression with data whitening.
    
    This class combines data whitening with SVD compression to achieve
    better truncation decisions.
    """
    
    def __init__(
        self,
        whitener: DataWhitener,
        use_whitening: bool = True
    ):
        """
        Initialize whitened SVD.
        
        Args:
            whitener: Data whitener instance
            use_whitening: Whether to apply whitening (for ablation studies)
        """
        self.whitener = whitener
        self.use_whitening = use_whitening
    
    def compress_layer(
        self,
        layer: nn.Linear,
        rank: int,
        calibration_data: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Compress a layer using whitened SVD.
        
        Args:
            layer: Linear layer to compress
            rank: Target rank for compression
            calibration_data: Optional calibration data
            
        Returns:
            Tuple of (U, s, Vt, whitening_matrix_inv)
            - U, s, Vt: SVD components of whitened weight
            - whitening_matrix_inv: Inverse whitening matrix (None if whitening disabled)
        """
        if self.use_whitening:
            # Apply whitening
            whitened_weight, whitening_matrix_inv = self.whitener.whiten_layer(
                layer, calibration_data
            )
            
            # SVD of whitened weight
            U, s, Vt = torch.linalg.svd(whitened_weight, full_matrices=False)
            
            # Truncate to target rank
            U = U[:, :rank]
            s = s[:rank]
            Vt = Vt[:rank, :]
            
            return U, s, Vt, whitening_matrix_inv
        else:
            # Standard SVD without whitening
            U, s, Vt = torch.linalg.svd(layer.weight.data, full_matrices=False)
            
            # Truncate to target rank
            U = U[:, :rank]
            s = s[:rank]
            Vt = Vt[:rank, :]
            
            return U, s, Vt, None
    
    def reconstruct_layer(
        self,
        U: torch.Tensor,
        s: torch.Tensor,
        Vt: torch.Tensor,
        whitening_matrix_inv: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None
    ) -> nn.Linear:
        """
        Reconstruct a compressed layer from SVD components.
        
        Args:
            U, s, Vt: SVD components
            whitening_matrix_inv: Inverse whitening matrix (if whitening was used)
            bias: Original bias term
            
        Returns:
            Reconstructed linear layer
        """
        # Reconstruct weight matrix
        if whitening_matrix_inv is not None:
            # For whitened SVD: W_reconstructed = U @ diag(s) @ Vt @ S^(-1)
            # where S^(-1) is the whitening_matrix_inv
            reconstructed_weight = U @ torch.diag(s) @ Vt @ whitening_matrix_inv
        else:
            # Standard reconstruction
            reconstructed_weight = U @ torch.diag(s) @ Vt
        
        # Create new layer
        out_features, in_features = reconstructed_weight.shape
        new_layer = nn.Linear(in_features, out_features, bias=bias is not None)
        new_layer.weight.data = reconstructed_weight
        
        if bias is not None:
            new_layer.bias.data = bias
        
        return new_layer


def create_calibration_dataset(
    dataset_name: str = "wikitext2",
    hidden_dim: Optional[int] = None,
    device: Optional[torch.device] = None
) -> CalibrationDataset:
    """
    Create a calibration dataset by name.
    
    Args:
        dataset_name: Name of dataset ("wikitext2", "random")
        hidden_dim: Hidden dimension (required for random dataset)
        device: Device for tensors
        
    Returns:
        CalibrationDataset instance
    """
    if dataset_name.lower() == "wikitext2":
        try:
            return WikiTextDataset(device=device)
        except ImportError:
            logger.warning("WikiText dataset unavailable, falling back to random data")
            if hidden_dim is None:
                raise ValueError("hidden_dim required for random dataset fallback")
            return RandomDataset(hidden_dim, device)
    
    elif dataset_name.lower() == "random":
        if hidden_dim is None:
            raise ValueError("hidden_dim required for random dataset")
        return RandomDataset(hidden_dim, device)
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def compare_whitening_vs_standard(
    layer: nn.Linear,
    target_rank: int,
    calibration_data: torch.Tensor,
    test_data: torch.Tensor
) -> Dict[str, float]:
    """
    Compare whitened SVD vs standard SVD on a layer.
    
    Args:
        layer: Layer to compress
        target_rank: Target compression rank
        calibration_data: Data for whitening computation
        test_data: Data for evaluation
        
    Returns:
        Dictionary with comparison metrics
    """
    device = layer.weight.device
    
    # Create whitener
    whitener = DataWhitener(device=device)
    whitened_svd = WhitenedSVD(whitener)
    
    # Test whitened SVD
    U_w, s_w, Vt_w, whitening_inv = whitened_svd.compress_layer(
        layer, target_rank, calibration_data
    )
    layer_w = whitened_svd.reconstruct_layer(U_w, s_w, Vt_w, whitening_inv, layer.bias)
    
    # Test standard SVD
    U_s, s_s, Vt_s, _ = whitened_svd.compress_layer(layer, target_rank, None)
    layer_s = whitened_svd.reconstruct_layer(U_s, s_s, Vt_s, None, layer.bias)
    
    # Evaluate on test data
    with torch.no_grad():
        # Original outputs
        original_output = layer(test_data)
        
        # Compressed outputs
        whitened_output = layer_w(test_data)
        standard_output = layer_s(test_data)
        
        # Compute errors
        whitened_error = F.mse_loss(whitened_output, original_output).item()
        standard_error = F.mse_loss(standard_output, original_output).item()
        
        # Compute relative errors
        original_norm = torch.norm(original_output).item()
        whitened_rel_error = torch.norm(whitened_output - original_output).item() / original_norm
        standard_rel_error = torch.norm(standard_output - original_output).item() / original_norm
    
    return {
        'whitened_mse': whitened_error,
        'standard_mse': standard_error,
        'whitened_relative_error': whitened_rel_error,
        'standard_relative_error': standard_rel_error,
        'improvement_factor': standard_error / whitened_error,
        'whitened_singular_values': s_w.cpu().numpy().tolist(),
        'standard_singular_values': s_s.cpu().numpy().tolist()
    }