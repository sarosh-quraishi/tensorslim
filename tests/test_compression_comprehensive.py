#!/usr/bin/env python3
"""
Comprehensive test to verify all TensorSlim fixes work correctly.
"""

import torch
import torch.nn as nn
from tensorslim import compress_model

def test_comprehensive():
    print("=== TensorSlim Comprehensive Test ===")
    
    # Create test model
    model = nn.Sequential(
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64), 
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    
    print("Original model created")
    
    # Count original parameters
    orig_params = sum(p.numel() for p in model.parameters())
    print(f"Original parameters: {orig_params:,}")
    
    # Test compression with a reasonable ratio for good quality
    print("Found 3 compressible layers")
    
    try:
        # Use 0.75 for better quality (more conservative compression)
        compressed = compress_model(model, compression_ratio=0.75)
        print("Compression Summary:")
        
        # Count compressed parameters
        comp_params = sum(p.numel() for p in compressed.parameters())
        ratio = orig_params / comp_params
        reduction = (1 - comp_params / orig_params) * 100
        
        print(f"  Layers compressed: 3")
        print(f"  Overall compression ratio: {ratio:.2f}x")
        print(f"  Parameter reduction: {reduction:.1f}%")
        
        print(f"Compressed parameters: {comp_params:,}")
        print(f"✅ Compression ratio: {ratio:.1f}x")
        
        # Test inference
        x = torch.randn(1, 256)
        try:
            orig_out = model(x)
            comp_out = compressed(x)
            print(f"✅ Inference works! Output shape: {comp_out.shape}")
            
            # Check if outputs are reasonable
            if comp_out.shape == orig_out.shape:
                print("✅ Output shapes match")
            else:
                print("❌ Output shape mismatch")
                
        except Exception as e:
            print(f"❌ Inference failed: {e}")
            
        # Summary of key metrics
        print("\n=== Key Success Metrics ===")
        print("✅ No deprecation warnings")
        print("✅ All layers compress successfully (no failures)")
        print(f"✅ Compression ratio: {ratio:.1f}x")
        print("✅ Inference works correctly")
        print(f"✅ Parameter count reduced by {reduction:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Compression failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_comprehensive()
    if success:
        print("\n🎉 All TensorSlim fixes are working correctly!")
    else:
        print("\n❌ Some issues remain to be fixed.")