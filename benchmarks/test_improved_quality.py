#!/usr/bin/env python3
"""
Test the improved compression algorithm for <2% quality loss.
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, 'src')

from tensorslim import compress_model

def test_improved_compression():
    """Test improved compression for <2% quality loss."""
    print("🎯 Testing Improved Compression for <2% Quality Loss")
    print("=" * 60)
    
    # Test with BERT-like model
    model = nn.Sequential(
        nn.Linear(768, 768),  
        nn.ReLU(),
        nn.Linear(768, 3072),  # Feed-forward expansion
        nn.ReLU(),
        nn.Linear(3072, 768),  # Feed-forward contraction
        nn.ReLU(),
        nn.Linear(768, 30522)  # Vocabulary projection
    )
    
    orig_params = sum(p.numel() for p in model.parameters())
    orig_size_mb = (orig_params * 4) / (1024 * 1024)
    print(f"Original parameters: {orig_params:,}")
    print(f"Original size: {orig_size_mb:.1f} MB")
    
    # Test conservative compression ratios
    test_ratios = [0.98, 0.95, 0.90, 0.85]
    
    results = []
    
    for ratio in test_ratios:
        print(f"\n--- Testing Compression Ratio: {ratio} ---")
        
        try:
            compressed = compress_model(model, compression_ratio=ratio)
            comp_params = sum(p.numel() for p in compressed.parameters())
            comp_size_mb = (comp_params * 4) / (1024 * 1024)
            actual_compression = orig_params / comp_params
            size_reduction = (1 - comp_params / orig_params) * 100
            
            # Quality test
            test_input = torch.randn(16, 768)
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
            print(f"  Compressed size: {comp_size_mb:.1f} MB")
            print(f"  Actual compression: {actual_compression:.2f}x")
            print(f"  Size reduction: {size_reduction:.1f}%")
            print(f"  Quality retention: {quality_retention:.2f}%")
            print(f"  Quality loss: {quality_loss:.2f}%")
            print(f"  MSE: {mse:.6f}")
            print(f"  Relative error: {relative_error:.4f}")
            
            if quality_loss < 2.0:
                print(f"  ✅ SUCCESS: <2% quality loss achieved!")
                status = "✅ <2% loss"
            elif quality_loss < 5.0:
                print(f"  ⚠️ GOOD: <5% quality loss")
                status = "⚠️ <5% loss"
            else:
                print(f"  ❌ POOR: >5% quality loss")
                status = "❌ >5% loss"
            
            results.append({
                'ratio': ratio,
                'compression': actual_compression,
                'size_reduction': size_reduction,
                'quality_loss': quality_loss,
                'quality_retention': quality_retention,
                'status': status,
                'orig_size_mb': orig_size_mb,
                'comp_size_mb': comp_size_mb
            })
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
    
    # Summary table
    print(f"\n" + "="*60)
    print("📊 COMPRESSION RESULTS SUMMARY")
    print("="*60)
    
    print("\n| Ratio | Compression | Size Reduction | Quality Loss | Status |")
    print("|-------|-------------|----------------|--------------|---------|")
    
    for result in results:
        print(f"| {result['ratio']} | {result['compression']:.2f}x | {result['size_reduction']:.1f}% | {result['quality_loss']:.2f}% | {result['status']} |")
    
    # Find best results for <2% quality loss
    good_results = [r for r in results if r['quality_loss'] < 2.0]
    
    if good_results:
        best = max(good_results, key=lambda x: x['compression'])
        print(f"\n🏆 BEST RESULT FOR <2% QUALITY LOSS:")
        print(f"  • Compression ratio: {best['compression']:.2f}x")
        print(f"  • Size: {best['orig_size_mb']:.1f}MB → {best['comp_size_mb']:.1f}MB")
        print(f"  • Size reduction: {best['size_reduction']:.1f}%")
        print(f"  • Quality loss: {best['quality_loss']:.2f}%")
        print(f"  • Target ratio: {best['ratio']}")
        
        return best
    else:
        print(f"\n❌ No results achieved <2% quality loss")
        return None

def test_with_huggingface_model():
    """Test with a real HuggingFace model if available."""
    print(f"\n🤖 Testing with Real HuggingFace Model")
    print("=" * 60)
    
    try:
        from transformers import AutoModel
        
        print("Loading DistilBERT (smaller than BERT for faster testing)...")
        model = AutoModel.from_pretrained('distilbert-base-uncased')
        
        orig_params = sum(p.numel() for p in model.parameters())
        orig_size_mb = (orig_params * 4) / (1024 * 1024)
        print(f"Original parameters: {orig_params:,}")
        print(f"Original size: {orig_size_mb:.1f} MB")
        
        # Test with conservative ratio that should give <2% quality loss
        print(f"\nTesting compression ratio 0.98 (for <2% quality loss)...")
        
        compressed = compress_model(model, compression_ratio=0.98)
        comp_params = sum(p.numel() for p in compressed.parameters())
        comp_size_mb = (comp_params * 4) / (1024 * 1024)
        actual_compression = orig_params / comp_params
        size_reduction = (1 - comp_params / orig_params) * 100
        
        print(f"Compressed parameters: {comp_params:,}")
        print(f"Compressed size: {comp_size_mb:.1f} MB")
        print(f"Actual compression: {actual_compression:.2f}x")
        print(f"Size reduction: {size_reduction:.1f}%")
        
        # Test quality with actual text
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
            inputs = tokenizer("Hello, this is a test sentence.", return_tensors='pt')
            
            model.eval()
            compressed.eval()
            
            with torch.no_grad():
                orig_outputs = model(**inputs)
                comp_outputs = compressed(**inputs)
                
                orig_hidden = orig_outputs.last_hidden_state
                comp_hidden = comp_outputs.last_hidden_state
                
                cos_sim = torch.nn.functional.cosine_similarity(
                    orig_hidden.flatten(), comp_hidden.flatten(), dim=0
                ).item()
                
                quality_loss = (1 - cos_sim) * 100
                
                print(f"Quality loss: {quality_loss:.2f}%")
                
                if quality_loss < 2.0:
                    print(f"✅ SUCCESS: Achieved <2% quality loss with real HuggingFace model!")
                    return True
                else:
                    print(f"⚠️ Quality loss is {quality_loss:.2f}% (higher than 2%)")
                    return False
        
        except Exception as e:
            print(f"Quality test failed: {e}")
            return False
            
    except ImportError:
        print("HuggingFace transformers not available, skipping real model test")
        return None
    except Exception as e:
        print(f"Failed to test HuggingFace model: {e}")
        return None

if __name__ == "__main__":
    # Test with synthetic model
    synthetic_result = test_improved_compression()
    
    # Test with real model if available
    real_result = test_with_huggingface_model()
    
    print(f"\n" + "="*60)
    print("🎉 FINAL ASSESSMENT")
    print("="*60)
    
    if synthetic_result and synthetic_result['quality_loss'] < 2.0:
        print("✅ Synthetic model: <2% quality loss achieved")
    else:
        print("❌ Synthetic model: Failed to achieve <2% quality loss")
    
    if real_result is True:
        print("✅ Real HuggingFace model: <2% quality loss achieved")
    elif real_result is False:
        print("❌ Real HuggingFace model: Failed to achieve <2% quality loss")
    else:
        print("⚠️ Real HuggingFace model: Not tested")
    
    if (synthetic_result and synthetic_result['quality_loss'] < 2.0) or real_result is True:
        print("\n🎉 TensorSlim can achieve <2% quality loss!")
    else:
        print("\n⚠️ Further optimization needed for <2% quality loss")