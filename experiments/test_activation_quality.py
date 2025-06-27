#!/usr/bin/env python3
"""
Test script to compare activation-based quality vs matrix reconstruction quality.
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, 'src')

from tensorslim.core.randomized_svd import RandomizedSVD, AdaptiveRandomizedSVD

def create_test_layer_and_data():
    """Create a test linear layer and some test data."""
    torch.manual_seed(42)
    
    # Create a simple linear layer
    layer = nn.Linear(512, 256, bias=False)
    weight = layer.weight.data  # Shape: (256, 512)
    
    # Create test inputs
    batch_size = 32
    test_inputs = torch.randn(batch_size, 512)
    
    return weight, test_inputs

def test_quality_metrics():
    """Compare old vs new quality measurement approaches."""
    print("🧪 Testing Activation-Based Quality vs Matrix Reconstruction Quality")
    print("=" * 70)
    
    weight, test_inputs = create_test_layer_and_data()
    print(f"Weight matrix shape: {weight.shape}")
    print(f"Test input shape: {test_inputs.shape}")
    
    # Test different compression ranks
    ranks_to_test = [32, 64, 128, 192]
    
    print(f"\n{'Rank':<6} {'Matrix Recon':<14} {'Activation Sim':<15} {'Difference':<12}")
    print("-" * 50)
    
    svd = RandomizedSVD(rank=64)  # Will be overridden in loop
    
    for rank in ranks_to_test:
        svd.rank = rank
        
        # Perform SVD compression
        U, s, Vt = svd.fit_transform(weight, rank)
        compressed_weight = svd.reconstruct(U, s, Vt)
        
        # Method 1: Matrix reconstruction quality (old way)
        matrix_error = svd.relative_error(weight, compressed_weight)
        matrix_quality = 1.0 - matrix_error / 100
        
        # Method 2: Activation-based quality (new way)
        activation_quality = svd.activation_quality(
            weight, compressed_weight, test_inputs, "linear"
        )
        
        difference = activation_quality - matrix_quality
        
        print(f"{rank:<6} {matrix_quality:<14.4f} {activation_quality:<15.4f} {difference:<12.4f}")

def test_adaptive_svd():
    """Test AdaptiveRandomizedSVD with both quality metrics."""
    print(f"\n🎯 Testing AdaptiveRandomizedSVD with Activation Quality")
    print("=" * 60)
    
    weight, test_inputs = create_test_layer_and_data()
    
    target_qualities = [0.90, 0.95, 0.98]
    
    print(f"{'Target':<8} {'Old Method':<12} {'New Method':<12} {'Old Rank':<10} {'New Rank':<10}")
    print("-" * 60)
    
    for target_quality in target_qualities:
        # Test with old method (matrix reconstruction) - fallback when no test_inputs
        svd_old = AdaptiveRandomizedSVD(
            rank=64, 
            target_quality=target_quality,
            max_oversamples=15,
            max_power_iterations=3
        )
        
        U_old, s_old, Vt_old = svd_old.fit_transform(weight)
        old_rank = len(s_old)
        compressed_old = svd_old.reconstruct(U_old, s_old, Vt_old)
        old_quality = 1.0 - svd_old.relative_error(weight, compressed_old) / 100
        
        # Test with new method (activation-based)
        svd_new = AdaptiveRandomizedSVD(
            rank=64, 
            target_quality=target_quality,
            max_oversamples=15,
            max_power_iterations=3
        )
        
        U_new, s_new, Vt_new = svd_new.fit_transform(
            weight, test_inputs=test_inputs, layer_type="linear"
        )
        new_rank = len(s_new)
        compressed_new = svd_new.reconstruct(U_new, s_new, Vt_new)
        new_quality = svd_new.activation_quality(
            weight, compressed_new, test_inputs, "linear"
        )
        
        print(f"{target_quality:<8.2f} {old_quality:<12.4f} {new_quality:<12.4f} {old_rank:<10} {new_rank:<10}")

def benchmark_model_compression():
    """Test on a small BERT-like model."""
    print(f"\n🏗️ Testing on Mini-BERT Model")
    print("=" * 40)
    
    # Create a mini transformer-like model
    torch.manual_seed(42)
    
    # Attention weights (query, key, value projections)
    attention_dim = 768
    hidden_dim = 3072
    
    # Create typical transformer layer weights
    weights = {
        'attention.query': torch.randn(attention_dim, attention_dim),
        'attention.key': torch.randn(attention_dim, attention_dim), 
        'attention.value': torch.randn(attention_dim, attention_dim),
        'ffn.up': torch.randn(hidden_dim, attention_dim),
        'ffn.down': torch.randn(attention_dim, hidden_dim),
    }
    
    # Test inputs for each layer type
    batch_size = 16
    seq_len = 128
    test_inputs = {
        'attention': torch.randn(batch_size * seq_len, attention_dim),
        'ffn.up': torch.randn(batch_size * seq_len, attention_dim),
        'ffn.down': torch.randn(batch_size * seq_len, hidden_dim),
    }
    
    compression_ratio = 0.5  # 50% compression
    
    print(f"{'Layer':<15} {'Original Size':<14} {'Compressed Size':<16} {'Act Quality':<12} {'Compression':<12}")
    print("-" * 80)
    
    for layer_name, weight in weights.items():
        layer_type = 'attention' if 'attention' in layer_name else 'ffn.up' if 'up' in layer_name else 'ffn.down'
        test_input = test_inputs.get(layer_type, test_inputs['attention'])
        
        # Calculate target rank
        min_dim = min(weight.shape)
        target_rank = int(min_dim * compression_ratio)
        
        # Compress using activation quality
        svd = AdaptiveRandomizedSVD(
            rank=target_rank,
            target_quality=0.92,
            max_oversamples=10,
            max_power_iterations=2
        )
        
        U, s, Vt = svd.fit_transform(
            weight, test_inputs=test_input, layer_type="linear"
        )
        
        compressed_weight = svd.reconstruct(U, s, Vt)
        quality = svd.activation_quality(
            weight, compressed_weight, test_input, "linear"
        )
        
        original_size = weight.numel()
        compressed_size = len(s) * (weight.shape[0] + weight.shape[1] + 1)
        actual_compression = original_size / compressed_size
        
        print(f"{layer_name:<15} {original_size:<14} {compressed_size:<16} {quality:<12.4f} {actual_compression:<12.2f}x")

if __name__ == "__main__":
    test_quality_metrics()
    test_adaptive_svd()
    benchmark_model_compression()
    
    print(f"\n✅ Testing completed!")