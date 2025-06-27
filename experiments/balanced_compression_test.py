#!/usr/bin/env python3
"""
Find the balanced compression ratios that give good compression with high quality.
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, 'src')

from tensorslim.core.randomized_svd import RandomizedSVD

def find_optimal_compression():
    """Find optimal compression ratios for different layer types."""
    print("🎯 Finding Optimal Compression Ratios")
    print("=" * 50)
    
    torch.manual_seed(42)
    
    # Test realistic transformer layers
    layers = {
        'BERT Attention (768×768)': (768, 768),
        'BERT FFN Up (3072×768)': (3072, 768),
        'BERT FFN Down (768×3072)': (768, 3072),
        'GPT-2 Attention (1024×1024)': (1024, 1024),
        'GPT-2 FFN Up (4096×1024)': (4096, 1024),
    }
    
    print(f"{'Layer':<25} {'Rank':<6} {'Ratio':<8} {'Quality':<8} {'Status':<10}")
    print("-" * 65)
    
    results = []
    
    for layer_name, (out_dim, in_dim) in layers.items():
        weight = torch.randn(out_dim, in_dim)
        test_inputs = torch.randn(32, in_dim)
        
        # Test different ranks to find good compression with quality >90%
        max_rank = min(out_dim, in_dim)
        
        for rank in [max_rank//8, max_rank//6, max_rank//4, max_rank//3, max_rank//2]:
            if rank < 1:
                continue
                
            svd = RandomizedSVD(rank=rank, n_oversamples=15, n_power_iterations=3)
            U, s, Vt = svd.fit_transform(weight, rank)
            compressed_weight = svd.reconstruct(U, s, Vt)
            
            # Calculate compression ratio
            original_params = out_dim * in_dim
            compressed_params = rank * (out_dim + in_dim + 1)
            ratio = original_params / compressed_params
            
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
            
            print(f"{layer_name:<25} {rank:<6} {ratio:<8.1f}x {quality:<8.3f} {status:<10}")
            
            if quality >= 0.90:
                results.append({
                    'layer': layer_name,
                    'rank': rank,
                    'ratio': ratio,
                    'quality': quality,
                    'status': status
                })
        
        print()
    
    # Find best results for README
    print("🏆 Best Results (Quality ≥ 90%):")
    print("-" * 40)
    
    excellent_results = [r for r in results if r['quality'] >= 0.95]
    good_results = [r for r in results if 0.90 <= r['quality'] < 0.95]
    
    if excellent_results:
        avg_ratio_excellent = sum(r['ratio'] for r in excellent_results) / len(excellent_results)
        avg_quality_excellent = sum(r['quality'] for r in excellent_results) / len(excellent_results)
        print(f"Excellent (≥95% quality): {avg_ratio_excellent:.1f}x avg compression, {avg_quality_excellent:.1%} avg quality")
    
    if good_results:
        avg_ratio_good = sum(r['ratio'] for r in good_results) / len(good_results)
        avg_quality_good = sum(r['quality'] for r in good_results) / len(good_results)
        print(f"Good (≥90% quality): {avg_ratio_good:.1f}x avg compression, {avg_quality_good:.1%} avg quality")
    
    # Show best single results
    if results:
        best_compression = max(results, key=lambda x: x['ratio'])
        best_quality = max(results, key=lambda x: x['quality'])
        
        print(f"\nBest compression: {best_compression['ratio']:.1f}x at {best_compression['quality']:.1%} quality")
        print(f"Best quality: {best_quality['quality']:.1%} at {best_quality['ratio']:.1f}x compression")

def test_practical_model():
    """Test on a practical small model scenario."""
    print(f"\n🏗️ Practical Model Compression Test")
    print("=" * 40)
    
    torch.manual_seed(42)
    
    # Simulate a 6-layer transformer (like DistilBERT)
    model_layers = {
        'attention.query': (768, 768),
        'attention.key': (768, 768), 
        'attention.value': (768, 768),
        'attention.output': (768, 768),
        'ffn.intermediate': (3072, 768),
        'ffn.output': (768, 3072),
    }
    
    total_original = 0
    total_compressed = 0
    total_quality = 0
    count = 0
    
    print(f"{'Layer':<20} {'Original':<10} {'Compressed':<12} {'Ratio':<8} {'Quality':<8}")
    print("-" * 65)
    
    for layer_name, (out_dim, in_dim) in model_layers.items():
        weight = torch.randn(out_dim, in_dim)
        test_inputs = torch.randn(16, in_dim)
        
        # Use different strategies for different layer types
        if 'attention' in layer_name:
            # Attention layers - more conservative
            target_rank = min(out_dim, in_dim) // 3
        else:
            # FFN layers - more aggressive
            target_rank = min(out_dim, in_dim) // 4
        
        svd = RandomizedSVD(rank=target_rank, n_oversamples=10, n_power_iterations=2)
        U, s, Vt = svd.fit_transform(weight, target_rank)
        compressed_weight = svd.reconstruct(U, s, Vt)
        
        original_params = out_dim * in_dim
        compressed_params = target_rank * (out_dim + in_dim + 1)
        ratio = original_params / compressed_params
        
        quality = svd.activation_quality(weight, compressed_weight, test_inputs, "linear")
        
        print(f"{layer_name:<20} {original_params:<10,} {compressed_params:<12,} {ratio:<8.1f}x {quality:<8.3f}")
        
        total_original += original_params
        total_compressed += compressed_params
        total_quality += quality
        count += 1
    
    overall_ratio = total_original / total_compressed
    avg_quality = total_quality / count
    
    print("-" * 65)
    print(f"{'OVERALL':<20} {total_original:<10,} {total_compressed:<12,} {overall_ratio:<8.1f}x {avg_quality:<8.3f}")

if __name__ == "__main__":
    find_optimal_compression()
    test_practical_model()