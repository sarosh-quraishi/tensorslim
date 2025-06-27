#!/usr/bin/env python3
"""
Test to understand why quality loss is so high and find better compression ratios.
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, 'src')

from tensorslim import compress_model
from tensorslim.core.compression import TensorSlim

def test_quality_vs_compression():
    """Test different compression approaches to minimize quality loss."""
    print("🔬 Quality vs Compression Analysis")
    print("=" * 50)
    
    # Create a simple test model
    model = nn.Sequential(
        nn.Linear(768, 768),  # BERT-like
        nn.ReLU(),
        nn.Linear(768, 768),
        nn.ReLU(),
        nn.Linear(768, 30522)  # Vocab size
    )
    
    orig_params = sum(p.numel() for p in model.parameters())
    print(f"Test model parameters: {orig_params:,}")
    
    # Test very conservative compression ratios
    conservative_ratios = [0.95, 0.90, 0.85, 0.80, 0.75]
    
    for ratio in conservative_ratios:
        print(f"\n--- Compression Ratio: {ratio} ---")
        
        try:
            compressed = compress_model(model, compression_ratio=ratio)
            comp_params = sum(p.numel() for p in compressed.parameters())
            actual_compression = orig_params / comp_params
            size_reduction = (1 - comp_params / orig_params) * 100
            
            # Quality test
            test_input = torch.randn(8, 768)
            model.eval()
            compressed.eval()
            
            with torch.no_grad():
                orig_out = model(test_input)
                comp_out = compressed(test_input)
                
                cos_sim = torch.nn.functional.cosine_similarity(
                    orig_out.flatten(), comp_out.flatten(), dim=0
                ).item()
                
                mse = torch.nn.functional.mse_loss(orig_out, comp_out).item()
                relative_error = (torch.norm(orig_out - comp_out) / torch.norm(orig_out)).item()
                
                quality_retention = cos_sim * 100
                quality_loss = (1 - cos_sim) * 100
            
            print(f"  Compressed params: {comp_params:,}")
            print(f"  Actual compression: {actual_compression:.2f}x")
            print(f"  Size reduction: {size_reduction:.1f}%")
            print(f"  Quality retention: {quality_retention:.2f}%")
            print(f"  Quality loss: {quality_loss:.2f}%")
            print(f"  MSE: {mse:.6f}")
            print(f"  Relative error: {relative_error:.6f}")
            
            if quality_loss < 2.0:
                print(f"  ✅ Achieves <2% quality loss!")
            elif quality_loss < 5.0:
                print(f"  ⚠️ Acceptable quality loss (<5%)")
            else:
                print(f"  ❌ High quality loss (>5%)")
        
        except Exception as e:
            print(f"  ❌ Failed: {e}")

def test_rank_based_compression():
    """Test fixed rank compression instead of ratio-based."""
    print(f"\n🎯 Testing Fixed Rank Compression")
    print("=" * 50)
    
    model = nn.Linear(768, 768)  # Single layer test
    orig_params = model.weight.numel() + (model.bias.numel() if model.bias is not None else 0)
    print(f"Original params: {orig_params:,}")
    print(f"Weight shape: {model.weight.shape}")
    
    # Test different fixed ranks
    ranks = [600, 500, 400, 300, 200, 100]
    
    for rank in ranks:
        print(f"\n--- Fixed Rank: {rank} ---")
        
        try:
            # Use TensorSlim with fixed rank
            compressor = TensorSlim(rank=rank)  # Use integer rank directly
            compressed = compressor.compress(model, inplace=False)
            
            # Get SlimLinear layer
            slim_layer = compressed[0] if isinstance(compressed, nn.Sequential) else compressed
            
            comp_params = sum(p.numel() for p in compressed.parameters())
            actual_compression = orig_params / comp_params
            size_reduction = (1 - comp_params / orig_params) * 100
            
            # Quality test
            test_input = torch.randn(32, 768)
            model.eval()
            compressed.eval()
            
            with torch.no_grad():
                orig_out = model(test_input)
                comp_out = compressed(test_input)
                
                cos_sim = torch.nn.functional.cosine_similarity(
                    orig_out.flatten(), comp_out.flatten(), dim=0
                ).item()
                
                relative_error = (torch.norm(orig_out - comp_out) / torch.norm(orig_out)).item()
                quality_loss = (1 - cos_sim) * 100
            
            print(f"  Rank: {rank}/{min(model.weight.shape)} ({rank/min(model.weight.shape)*100:.1f}%)")
            print(f"  Compressed params: {comp_params:,}")
            print(f"  Actual compression: {actual_compression:.2f}x")
            print(f"  Size reduction: {size_reduction:.1f}%")
            print(f"  Quality loss: {quality_loss:.2f}%")
            print(f"  Relative error: {relative_error:.4f}")
            
            if quality_loss < 2.0:
                print(f"  ✅ Achieves <2% quality loss with {actual_compression:.1f}x compression!")
                
        except Exception as e:
            print(f"  ❌ Failed: {e}")

def find_optimal_rank_for_quality():
    """Binary search to find optimal rank for <2% quality loss."""
    print(f"\n🎯 Finding Optimal Rank for <2% Quality Loss")
    print("=" * 50)
    
    model = nn.Linear(768, 768)
    orig_params = model.weight.numel() + (model.bias.numel() if model.bias is not None else 0)
    
    # Binary search for optimal rank
    low_rank = 50
    high_rank = min(model.weight.shape)  # 768
    target_quality_loss = 2.0
    
    best_rank = None
    best_compression = None
    best_quality_loss = float('inf')
    
    print(f"Searching for rank between {low_rank} and {high_rank}...")
    
    for iteration in range(10):  # Max 10 iterations
        mid_rank = (low_rank + high_rank) // 2
        print(f"\nIteration {iteration + 1}: Testing rank {mid_rank}")
        
        try:
            compressor = TensorSlim(rank=mid_rank)
            compressed = compressor.compress(model, inplace=False)
            
            comp_params = sum(p.numel() for p in compressed.parameters())
            actual_compression = orig_params / comp_params
            
            # Quality test
            test_input = torch.randn(32, 768)
            model.eval()
            compressed.eval()
            
            with torch.no_grad():
                orig_out = model(test_input)
                comp_out = compressed(test_input)
                
                cos_sim = torch.nn.functional.cosine_similarity(
                    orig_out.flatten(), comp_out.flatten(), dim=0
                ).item()
                
                quality_loss = (1 - cos_sim) * 100
            
            print(f"  Quality loss: {quality_loss:.2f}%, Compression: {actual_compression:.2f}x")
            
            if quality_loss < target_quality_loss:
                # Quality is good, try lower rank for more compression
                best_rank = mid_rank
                best_compression = actual_compression
                best_quality_loss = quality_loss
                high_rank = mid_rank - 1
                print(f"  ✅ Good quality! Trying lower rank...")
            else:
                # Quality too poor, need higher rank
                low_rank = mid_rank + 1
                print(f"  ❌ Poor quality, trying higher rank...")
            
            if low_rank > high_rank:
                break
                
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            break
    
    if best_rank:
        print(f"\n🏆 OPTIMAL RESULT:")
        print(f"  Best rank: {best_rank}")
        print(f"  Compression: {best_compression:.2f}x")
        print(f"  Quality loss: {best_quality_loss:.2f}%")
        print(f"  Rank ratio: {best_rank/768:.2f} ({best_rank/768*100:.1f}%)")
    else:
        print(f"\n❌ Could not find rank achieving <2% quality loss")

if __name__ == "__main__":
    test_quality_vs_compression()
    test_rank_based_compression()
    find_optimal_rank_for_quality()