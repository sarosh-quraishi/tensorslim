
import torch
import sys
#sys.path.insert(0, 'src')
from tensorslim.core import randomized_svd

# Test SVD
matrix = torch.randn(100, 80)
U, s, Vt = randomized_svd(matrix, rank=20)

print(f'✅ SVD shapes: U{U.shape}, s{s.shape}, Vt{Vt.shape}')

# Test reconstruction
reconstructed = U @ torch.diag(s) @ Vt
error = torch.norm(matrix - reconstructed) / torch.norm(matrix)
print(f'✅ Reconstruction error: {error.item():.6f}')
