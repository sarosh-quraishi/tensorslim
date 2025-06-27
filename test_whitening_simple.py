#!/usr/bin/env python3
"""
Simple test of whitening effectiveness with controlled real-world patterns.
"""

import sys
sys.path.insert(0, 'src')
import torch
import torch.nn as nn
from tensorslim.core.whitening import DataWhitener, WhitenedSVD
from tensorslim.core.randomized_svd import RandomizedSVD
import re

print('🧪 Simple Whitening Test with Real Text Patterns')
print('=' * 60)

# Read some text and create simple word embeddings
with open('README.md', 'r', encoding='utf-8') as f:
    text = f.read()[:5000]  # First 5000 chars to keep it manageable

# Simple tokenization
words = re.findall(r'\b\w+\b', text.lower())
vocab = list(set(words))[:100]  # Top 100 unique words
word_to_idx = {word: i for i, word in enumerate(vocab)}

print(f'Using {len(vocab)} words from README')

# Create smaller, more manageable embeddings
embed_dim = 64
torch.manual_seed(42)

# Create word embeddings with realistic correlation patterns
embeddings = torch.randn(len(vocab), embed_dim) * 0.5

# Add correlations: words that appear together get similar embeddings
window_size = 3
for i in range(len(words) - window_size):
    window_words = [w for w in words[i:i+window_size] if w in word_to_idx]
    if len(window_words) > 1:
        # Average embeddings of co-occurring words
        indices = [word_to_idx[w] for w in window_words]
        mean_embedding = embeddings[indices].mean(dim=0)
        for idx in indices:
            embeddings[idx] = 0.9 * embeddings[idx] + 0.1 * mean_embedding

print('Created correlated word embeddings')

# Create sentence-level representations by averaging word embeddings
sentences = []
sentence_words = []
current_words = []

for word in words:
    if word in word_to_idx:
        current_words.append(word)
        if len(current_words) >= 8:  # 8 words per "sentence"
            sentence_words.append(current_words[:8])
            current_words = []

# Convert to embeddings
sentence_embeddings = []
for sent_words in sentence_words:
    indices = [word_to_idx[w] for w in sent_words]
    sent_embed = embeddings[indices].mean(dim=0)  # Average word embeddings
    sentence_embeddings.append(sent_embed)

sentence_data = torch.stack(sentence_embeddings)
print(f'Created {len(sentence_data)} sentence embeddings of dim {embed_dim}')

# Split data
split_idx = int(0.8 * len(sentence_data))
calibration_data = sentence_data[:split_idx]
test_data = sentence_data[split_idx:]

print(f'Calibration: {len(calibration_data)}, Test: {len(test_data)}')

# Create layer to test
layer = nn.Linear(embed_dim, 32)
rank = 16

print(f'\\nTesting compression: {embed_dim} → 32, rank {rank}')

# Test 1: Standard SVD
print('\\n1. STANDARD SVD COMPRESSION')
svd_std = RandomizedSVD(rank, use_whitening=False)
U_std, s_std, Vt_std = svd_std.fit_transform(layer.weight.data)
compressed_std = svd_std.reconstruct(U_std, s_std, Vt_std)

# Test 2: Manual whitening (avoiding the calibration data issues)
print('\\n2. MANUAL WHITENING APPROACH')

try:
    # Compute covariance manually 
    print('Computing covariance of calibration data...')
    
    # Center the data
    mean_cal = calibration_data.mean(dim=0, keepdim=True)
    centered_cal = calibration_data - mean_cal
    
    # Compute covariance with regularization
    n_samples = centered_cal.shape[0]
    cov = (centered_cal.T @ centered_cal) / (n_samples - 1)
    
    # Add regularization for numerical stability
    regularization = 1e-3
    cov += regularization * torch.eye(embed_dim)
    
    print(f'Covariance shape: {cov.shape}')
    
    # Check condition number
    eigenvals, eigenvecs = torch.linalg.eigh(cov)
    eigenvals = torch.clamp(eigenvals, min=regularization)  # Ensure positive
    condition_num = eigenvals[-1] / eigenvals[0]
    
    print(f'Condition number: {condition_num:.1f}')
    print(f'Min eigenvalue: {eigenvals[0]:.6f}')
    print(f'Max eigenvalue: {eigenvals[-1]:.6f}')
    
    # Compute whitening matrix using eigendecomposition (more stable)
    sqrt_inv_eigenvals = torch.diag(1.0 / torch.sqrt(eigenvals))
    whitening_matrix_inv = eigenvecs @ sqrt_inv_eigenvals @ eigenvecs.T
    
    # Apply whitening to weight matrix
    whitening_matrix = eigenvecs @ torch.diag(torch.sqrt(eigenvals)) @ eigenvecs.T
    whitened_weight = layer.weight.data @ whitening_matrix
    
    # SVD on whitened weight
    U_w, s_w, Vt_w = svd_std.fit_transform(whitened_weight)
    
    # Reconstruct with whitening
    compressed_whitened = svd_std.reconstruct(U_w, s_w, Vt_w) @ whitening_matrix_inv
    
    print('✅ Manual whitening successful')
    whitening_success = True
    
except Exception as e:
    print(f'❌ Manual whitening failed: {e}')
    compressed_whitened = compressed_std
    whitening_success = False

# Evaluate quality
print('\\n3. QUALITY COMPARISON')

with torch.no_grad():
    # Original outputs
    original_out = layer(test_data)
    
    # Standard compression
    std_out = torch.nn.functional.linear(test_data, compressed_std, layer.bias)
    std_error = torch.norm(original_out - std_out) / torch.norm(original_out)
    
    # Activation-based quality (cosine similarity)
    std_quality = torch.nn.functional.cosine_similarity(
        original_out.flatten(), std_out.flatten(), dim=0
    )
    
    print(f'Standard SVD:')
    print(f'  Relative error:        {std_error:.4f}')
    print(f'  Activation quality:    {std_quality:.4f}')
    
    if whitening_success:
        # Whitened compression
        whitened_out = torch.nn.functional.linear(test_data, compressed_whitened, layer.bias)
        whitened_error = torch.norm(original_out - whitened_out) / torch.norm(original_out)
        
        whitened_quality = torch.nn.functional.cosine_similarity(
            original_out.flatten(), whitened_out.flatten(), dim=0
        )
        
        improvement_error = std_error / whitened_error
        improvement_quality = whitened_quality / std_quality
        
        print(f'\\nWhitened SVD:')
        print(f'  Relative error:        {whitened_error:.4f}')
        print(f'  Activation quality:    {whitened_quality:.4f}')
        print(f'\\nImprovement:')
        print(f'  Error reduction:       {improvement_error:.2f}x')
        print(f'  Quality improvement:   {improvement_quality:.3f}x')
        
        if improvement_error > 1.1:
            print('  ✅ Whitening reduces error significantly!')
        elif improvement_error > 1.02:
            print('  ✅ Whitening provides modest improvement')
        else:
            print('  ➡️  Whitening shows minimal difference')

print('\\n4. ANALYSIS OF TEXT CORRELATION STRUCTURE')

# Analyze the correlation structure we created
print('Analyzing sentence embedding correlations...')

correlation_matrix = torch.corrcoef(calibration_data.T)
avg_correlation = (correlation_matrix.sum() - correlation_matrix.trace()) / (embed_dim * (embed_dim - 1))

print(f'Average pairwise correlation: {avg_correlation:.4f}')
print(f'Max correlation (off-diagonal): {correlation_matrix.fill_diagonal_(0).max():.4f}')

if abs(avg_correlation) > 0.1:
    print('✅ Significant correlations present - good candidate for whitening')
elif abs(avg_correlation) > 0.05:
    print('➡️  Moderate correlations present')
else:
    print('⚠️  Low correlations - whitening may not help much')

print('\\n5. KEY INSIGHTS')
print('• Whitening effectiveness depends on input correlation structure')
print('• Real text naturally creates correlated embeddings')
print('• Numerical stability is crucial for covariance matrix operations')
print('• Regularization helps with small sample sizes')
print('• Activation-based quality measurement is more meaningful than error norms')

if whitening_success and 'improvement_error' in locals() and improvement_error > 1.05:
    print('\\n✅ CONCLUSION: Whitening provides measurable improvement with real text data!')
else:
    print('\\n➡️  CONCLUSION: Whitening provides mixed results - depends on data characteristics')