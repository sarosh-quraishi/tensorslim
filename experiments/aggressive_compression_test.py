#!/usr/bin/env python3
"""
Test more aggressive compression ratios to get compelling benchmark numbers.
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, 'src')

from tensorslim.core.randomized_svd import RandomizedSVD

def test_aggressive_compression():
    """Test much more aggressive compression ratios."""
    print("🚀 Testing Aggressive Compression Ratios")
    print("=" * 60)
    
    # Create larger test matrices (more realistic model sizes)
    torch.manual_seed(42)
    
    # Focus on FFN layers only (attention layers preserved)
    layers = {
        'BERT FFN Up (768→3072)': (3072, 768),
        'BERT FFN Down (3072→768)': (768, 3072), 
        'GPT-2 FFN Up (1024→4096)': (4096, 1024),
        'GPT-2 FFN Down (4096→1024)': (1024, 4096),
    }
    
    # Test much more aggressive compression ratios
    compression_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]  # 10x, 5x, 3.3x, 2.5x, 2x
    
    print(f"{'Layer':<25} {'Compression':<12} {'Ratio':<8} {'Act Quality':<12} {'Status':<10}")
    print("-" * 75)
    
    for layer_name, (out_dim, in_dim) in layers.items():
        weight = torch.randn(out_dim, in_dim)
        test_inputs = torch.randn(32, in_dim)
        
        for comp_ratio in compression_ratios:
            # Calculate target rank for this compression ratio
            total_params = out_dim * in_dim
            target_compressed_params = int(total_params * comp_ratio)
            
            # For SVD: compressed_size = rank * (out_dim + in_dim + 1)
            target_rank = target_compressed_params // (out_dim + in_dim + 1)
            target_rank = max(1, min(target_rank, min(out_dim, in_dim) - 1))
            
            svd = RandomizedSVD(rank=target_rank)
            U, s, Vt = svd.fit_transform(weight, target_rank)
            compressed_weight = svd.reconstruct(U, s, Vt)
            
            # Calculate actual compression ratio
            compressed_params = target_rank * (out_dim + in_dim + 1)
            actual_ratio = total_params / compressed_params
            
            # Calculate activation quality
            quality = svd.activation_quality(weight, compressed_weight, test_inputs, "linear")
            
            # Determine status
            if quality >= 0.95:
                status = "Excellent"
            elif quality >= 0.90:
                status = "Good"
            elif quality >= 0.85:
                status = "Fair"
            else:
                status = "Poor"
            
            print(f"{layer_name:<25} {comp_ratio:<12.1%} {actual_ratio:<8.1f}x {quality:<12.3f} {status:<10}")
            
            # Stop if quality gets too low for this layer
            if quality < 0.80:
                break
        
        print()

def test_adaptive_aggressive():
    """Test AdaptiveRandomizedSVD with more aggressive targets."""
    print("🎯 Testing Adaptive SVD with Aggressive Targets")
    print("=" * 55)
    
    torch.manual_seed(42)
    
    # Large realistic layer
    weight = torch.randn(4096, 1024)  # GPT-like FFN layer
    test_inputs = torch.randn(64, 1024)
    
    # Try different quality thresholds
    quality_targets = [0.95, 0.90, 0.85, 0.80]
    
    print(f"{'Quality Target':<15} {'Achieved':<10} {'Ratio':<8} {'Params Saved':<12}")
    print("-" * 50)
    
    for target in quality_targets:
        # Start with very aggressive rank and let adaptive algorithm find best
        from tensorslim.core.randomized_svd import AdaptiveRandomizedSVD
        
        svd = AdaptiveRandomizedSVD(
            rank=50,  # Start very aggressive
            target_quality=target,
            max_oversamples=20,
            max_power_iterations=4
        )
        
        U, s, Vt = svd.fit_transform(
            weight, test_inputs=test_inputs, layer_type="linear"
        )
        
        compressed_weight = svd.reconstruct(U, s, Vt)
        actual_quality = svd.activation_quality(weight, compressed_weight, test_inputs, "linear")
        
        # Calculate compression
        original_params = weight.numel()
        compressed_params = len(s) * (weight.shape[0] + weight.shape[1] + 1)
        ratio = original_params / compressed_params
        params_saved = (1 - compressed_params / original_params) * 100
        
        print(f"{target:<15.2f} {actual_quality:<10.3f} {ratio:<8.1f}x {params_saved:<12.1f}%")

if __name__ == "__main__":
    test_aggressive_compression()
    test_adaptive_aggressive()