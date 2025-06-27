#!/usr/bin/env python3
"""
Exact test case from the original prompt.
"""

import torch
import torch.nn as nn
from tensorslim import compress_model

# Create test model
model = nn.Sequential(
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 64), 
    nn.ReLU(),
    nn.Linear(64, 10)
)

# Test compression
orig_params = sum(p.numel() for p in model.parameters())
compressed = compress_model(model, compression_ratio=0.5)
comp_params = sum(p.numel() for p in compressed.parameters())

# Verify results
assert comp_params < orig_params, "Compression failed to reduce parameters"
ratio = orig_params / comp_params
assert ratio > 2.0, f"Expected 2x+ compression, got {ratio:.1f}x"

# Test inference
x = torch.randn(1, 256)
orig_out = model(x)
comp_out = compressed(x)
assert orig_out.shape == comp_out.shape, "Output shape mismatch"

print("✅ All assertions passed!")
print(f"Original params: {orig_params:,}")
print(f"Compressed params: {comp_params:,}")
print(f"Compression ratio: {ratio:.1f}x")
print(f"Output shape: {comp_out.shape}")