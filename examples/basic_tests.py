import torch
import torch.nn as nn
import sys
from tensorslim import compress_model

# Test basic compression
model = nn.Sequential(
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(), 
    nn.Linear(64, 10)
)

print('Original model created')
orig_params = sum(p.numel() for p in model.parameters())
print(f'Original parameters: {orig_params:,}')

# Compress
compressed = compress_model(model, compression_ratio=0.5)
comp_params = sum(p.numel() for p in compressed.parameters())
ratio = orig_params / comp_params

print(f'Compressed parameters: {comp_params:,}')
print(f'✅ Compression ratio: {ratio:.1f}x')

# Quick inference test
x = torch.randn(1, 256)
orig_out = model(x)
comp_out = compressed(x)
print(f'✅ Inference works! Output shape: {comp_out.shape}')
