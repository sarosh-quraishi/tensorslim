"""
Memory-efficient compressed layer implementations.

This module provides drop-in replacements for standard PyTorch layers
that use SVD compression for efficient inference.
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math


class SlimLinear(nn.Module):
    """
    Memory-efficient compressed linear layer using SVD decomposition.
    
    This layer replaces nn.Linear with a compressed version that uses
    three smaller matrices (U, s, Vt) instead of one large weight matrix.
    
    The forward pass computes: x @ (U @ diag(s) @ Vt).T + bias
    Which is equivalent to the original layer but with fewer parameters.
    
    Args:
        U: Left singular vectors (output_dim x rank)
        s: Singular values (rank,)
        Vt: Right singular vectors transposed (rank x input_dim)
        bias: Bias vector (optional)
        original_shape: Original weight matrix shape for reference
    """
    
    def __init__(
        self,
        U: Tensor,
        s: Tensor,
        Vt: Tensor,
        bias: Optional[Tensor] = None,
        original_shape: Optional[Tuple[int, int]] = None
    ):
        super().__init__()
        
        # Store SVD components as buffers (not trainable parameters)
        self.register_buffer('U', U.contiguous())
        self.register_buffer('s', s.contiguous())
        self.register_buffer('Vt', Vt.contiguous())
        
        if bias is not None:
            self.register_parameter('bias', nn.Parameter(bias))
        else:
            self.register_parameter('bias', None)
            
        # Store metadata
        self.original_shape = original_shape
        self.rank = s.size(0)
        self.in_features = Vt.size(1)
        self.out_features = U.size(0)
        
    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass through compressed linear layer.
        
        Computes: input @ (V @ diag(s) @ U^T) + bias
        Optimized as: ((input @ V) * s) @ U^T + bias
        """
        # Handle batch dimensions properly
        original_shape = input.shape
        batch_dims = original_shape[:-1]
        input_flat = input.view(-1, self.in_features)
        
        # Efficient computation: (input @ V) * s @ U^T
        # This avoids materializing the full weight matrix
        x = torch.matmul(input_flat, self.Vt.T)  # (batch, rank)
        x = x * self.s.unsqueeze(0)  # Broadcast multiply with singular values
        x = torch.matmul(x, self.U.T)  # (batch, out_features)
        
        # Reshape back to original batch dimensions
        output_shape = batch_dims + (self.out_features,)
        x = x.view(output_shape)
        
        if self.bias is not None:
            x = x + self.bias
            
        return x
    
    def reconstruct_weight(self) -> Tensor:
        """
        Reconstruct the full weight matrix from SVD components.
        
        Returns:
            Reconstructed weight matrix of shape (out_features, in_features)
        """
        # Reconstruct: U @ diag(s) @ Vt
        # Note: The original weight was (out_features, in_features)
        # U is (out_features, rank), s is (rank,), Vt is (rank, in_features)
        return self.U @ torch.diag(self.s) @ self.Vt
    
    def compression_ratio(self) -> float:
        """Calculate the compression ratio compared to original linear layer."""
        if self.original_shape is None:
            return 1.0
            
        original_params = self.original_shape[0] * self.original_shape[1]
        if self.bias is not None:
            original_params += self.original_shape[0]
            
        compressed_params = self.U.numel() + self.s.numel() + self.Vt.numel()
        if self.bias is not None:
            compressed_params += self.bias.numel()
            
        return original_params / compressed_params
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, ' \
               f'rank={self.rank}, compression_ratio={self.compression_ratio():.2f}x'


class SlimConv2d(nn.Module):
    """
    Memory-efficient compressed 2D convolutional layer using SVD decomposition.
    
    This layer replaces nn.Conv2d with a compressed version that decomposes
    the weight tensor and performs efficient convolution.
    
    Args:
        U: Left singular vectors for reshaped weight matrix
        s: Singular values 
        Vt: Right singular vectors transposed for reshaped weight matrix
        bias: Bias vector (optional)
        original_shape: Original weight tensor shape (out_ch, in_ch, kh, kw)
        kernel_size: Convolution kernel size
        stride: Convolution stride
        padding: Convolution padding
        dilation: Convolution dilation
        groups: Number of groups for grouped convolution
        padding_mode: Padding mode ('zeros', 'reflect', 'replicate', 'circular')
    """
    
    def __init__(
        self,
        U: Tensor,
        s: Tensor, 
        Vt: Tensor,
        bias: Optional[Tensor] = None,
        original_shape: Optional[Tuple[int, int, int, int]] = None,
        kernel_size: Tuple[int, int] = (3, 3),
        stride: Tuple[int, int] = (1, 1),
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Tuple[int, int] = (1, 1),
        groups: int = 1,
        padding_mode: str = 'zeros'
    ):
        super().__init__()
        
        # Store SVD components as buffers (not trainable parameters)
        self.register_buffer('U', U.contiguous())
        self.register_buffer('s', s.contiguous())
        self.register_buffer('Vt', Vt.contiguous())
        
        if bias is not None:
            self.register_parameter('bias', nn.Parameter(bias))
        else:
            self.register_parameter('bias', None)
        
        # Store convolution parameters
        self.original_shape = original_shape
        self.kernel_size = kernel_size
        self.stride = stride  
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        
        # Derived properties
        self.rank = s.size(0)
        if original_shape is not None:
            self.out_channels, self.in_channels, self.kernel_h, self.kernel_w = original_shape
        else:
            # Infer from SVD shapes
            self.out_channels = U.size(0)
            total_in_features = Vt.size(1)
            self.kernel_h, self.kernel_w = kernel_size
            self.in_channels = total_in_features // (self.kernel_h * self.kernel_w)
    
    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass through compressed convolutional layer.
        
        Reconstructs the convolution kernel on-the-fly and applies convolution.
        For very low ranks, this can be more efficient than standard convolution.
        """
        # Reconstruct weight tensor efficiently
        # U @ diag(s) @ Vt -> (out_channels, in_channels * kh * kw)
        weight_2d = self.U @ torch.diag(self.s) @ self.Vt
        
        # Reshape back to convolution format
        weight = weight_2d.view(
            self.out_channels, 
            self.in_channels, 
            self.kernel_h, 
            self.kernel_w
        )
        
        # Apply standard convolution
        return F.conv2d(
            input=input,
            weight=weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )
    
    def reconstruct_weight(self) -> Tensor:
        """
        Reconstruct the full weight tensor from SVD components.
        
        Returns:
            Reconstructed weight tensor of original shape
        """
        weight_2d = self.U @ torch.diag(self.s) @ self.Vt
        return weight_2d.view(self.original_shape)
    
    def compression_ratio(self) -> float:
        """Calculate the compression ratio compared to original conv layer."""
        if self.original_shape is None:
            return 1.0
            
        original_params = math.prod(self.original_shape)
        if self.bias is not None:
            original_params += self.original_shape[0]  # out_channels
            
        compressed_params = self.U.numel() + self.s.numel() + self.Vt.numel()
        if self.bias is not None:
            compressed_params += self.bias.numel()
            
        return original_params / compressed_params
    
    def extra_repr(self) -> str:
        return f'in_channels={self.in_channels}, out_channels={self.out_channels}, ' \
               f'kernel_size={self.kernel_size}, rank={self.rank}, ' \
               f'compression_ratio={self.compression_ratio():.2f}x'


class SlimSeparableConv2d(nn.Module):
    """
    Efficient separable convolution using SVD compression.
    
    This layer combines depthwise and pointwise convolutions with SVD compression
    for maximum efficiency in both parameters and computation.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        bias: bool = True,
        rank_ratio: float = 0.5
    ):
        super().__init__()
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        
        # Depthwise convolution (not compressed, already efficient)
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,  # Key: groups = in_channels for depthwise
            bias=False
        )
        
        # Compressed pointwise convolution (1x1)
        pointwise_rank = max(1, int(min(in_channels, out_channels) * rank_ratio))
        
        # Create SVD components for pointwise conv
        U = torch.randn(out_channels, pointwise_rank) * math.sqrt(2.0 / (in_channels + out_channels))
        s = torch.ones(pointwise_rank)  
        Vt = torch.randn(pointwise_rank, in_channels) * math.sqrt(2.0 / (in_channels + out_channels))
        
        self.pointwise = SlimLinear(
            U=U,
            s=s,
            Vt=Vt,
            bias=torch.zeros(out_channels) if bias else None,
            original_shape=(out_channels, in_channels)
        )
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through separable convolution."""
        # Depthwise convolution
        x = self.depthwise(x)
        
        # Pointwise convolution (via compressed linear layer)
        # Need to reshape from conv format to linear format
        batch_size, channels, height, width = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        x = x.view(-1, channels)  # (B*H*W, C)
        
        # Apply compressed pointwise conv
        x = self.pointwise(x)  # (B*H*W, out_channels)
        
        # Reshape back to conv format
        x = x.view(batch_size, height, width, self.out_channels)
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, out_channels, H, W)
        
        return x


class SlimEmbedding(nn.Module):
    """
    Compressed embedding layer using SVD decomposition.
    
    Useful for very large embedding tables where compression can significantly
    reduce memory usage with minimal quality loss.
    """
    
    def __init__(
        self,
        U: Tensor,
        s: Tensor,
        Vt: Tensor,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        original_shape: Optional[Tuple[int, int]] = None
    ):
        super().__init__()
        
        # Store SVD components as buffers (not trainable parameters)
        self.register_buffer('U', U.contiguous())
        self.register_buffer('s', s.contiguous())
        self.register_buffer('Vt', Vt.contiguous())
        
        # Embedding parameters
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self.original_shape = original_shape
        
        # Derived properties
        self.num_embeddings = U.size(0)
        self.embedding_dim = Vt.size(1)
        self.rank = s.size(0)
        
    def forward(self, input: Tensor) -> Tensor:
        """Forward pass through compressed embedding layer."""
        # Reconstruct embedding weight matrix
        weight = self.U @ torch.diag(self.s) @ self.Vt
        
        return F.embedding(
            input=input,
            weight=weight,
            padding_idx=self.padding_idx,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse
        )
    
    def compression_ratio(self) -> float:
        """Calculate compression ratio compared to original embedding."""
        if self.original_shape is None:
            return 1.0
            
        original_params = math.prod(self.original_shape)
        compressed_params = self.U.numel() + self.s.numel() + self.Vt.numel()
        
        return original_params / compressed_params


def convert_layer_to_slim(
    layer: nn.Module,
    rank: Union[int, float] = 0.5,
    use_srht: bool = True,
    use_whitening: bool = False,
    **svd_kwargs
) -> Optional[nn.Module]:
    """
    Convert a standard PyTorch layer to its compressed equivalent.
    
    Args:
        layer: PyTorch layer to compress
        rank: Compression rank (int) or ratio (float)
        use_srht: Whether to use SRHT instead of Gaussian matrices
        use_whitening: Whether to use data whitening
        **svd_kwargs: Additional arguments for SVD computation
        
    Returns:
        Compressed layer or None if layer type not supported
    """
    from .randomized_svd import randomized_svd
    
    if isinstance(layer, nn.Linear):
        weight = layer.weight.data
        bias = layer.bias.data if layer.bias is not None else None
        
        # Determine rank
        if isinstance(rank, float):
            target_rank = max(1, int(min(weight.shape) * rank))
        else:
            target_rank = min(rank, min(weight.shape))
        
        # Compute SVD with new features
        U, s, Vt = randomized_svd(
            weight, 
            target_rank, 
            use_srht=use_srht,
            use_whitening=use_whitening,
            layer=layer if use_whitening else None,
            **svd_kwargs
        )
        
        return SlimLinear(U, s, Vt, bias, weight.shape)
        
    elif isinstance(layer, nn.Conv2d):
        weight = layer.weight.data
        bias = layer.bias.data if layer.bias is not None else None
        
        # Reshape for SVD
        original_shape = weight.shape
        out_ch, in_ch, kh, kw = original_shape
        weight_2d = weight.view(out_ch, in_ch * kh * kw)
        
        # Determine rank
        if isinstance(rank, float):
            target_rank = max(1, int(min(weight_2d.shape) * rank))
        else:
            target_rank = min(rank, min(weight_2d.shape))
        
        # Compute SVD (note: whitening not typically used for conv layers)
        U, s, Vt = randomized_svd(
            weight_2d, 
            target_rank, 
            use_srht=use_srht,
            use_whitening=False,  # Whitening less common for conv
            **svd_kwargs
        )
        
        return SlimConv2d(
            U=U, s=s, Vt=Vt, bias=bias,
            original_shape=original_shape,
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=layer.groups,
            padding_mode=layer.padding_mode
        )
        
    elif isinstance(layer, nn.Embedding):
        weight = layer.weight.data
        
        # Determine rank
        if isinstance(rank, float):
            target_rank = max(1, int(min(weight.shape) * rank))
        else:
            target_rank = min(rank, min(weight.shape))
        
        # Compute SVD (note: whitening less common for embeddings)
        U, s, Vt = randomized_svd(
            weight, 
            target_rank, 
            use_srht=use_srht,
            use_whitening=False,  # Whitening less common for embeddings
            **svd_kwargs
        )
        
        return SlimEmbedding(
            U=U, s=s, Vt=Vt,
            padding_idx=layer.padding_idx,
            max_norm=layer.max_norm,
            norm_type=layer.norm_type,
            scale_grad_by_freq=layer.scale_grad_by_freq,
            sparse=layer.sparse,
            original_shape=weight.shape
        )
    
    return None