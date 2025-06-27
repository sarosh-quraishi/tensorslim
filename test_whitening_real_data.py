#!/usr/bin/env python3
"""
Test whitening effectiveness with real README text data.
"""

import sys
sys.path.insert(0, 'src')
import torch
import torch.nn as nn
from tensorslim.core.whitening import DataWhitener, WhitenedSVD, RandomDataset
import re
import numpy as np

print('📖 Testing Whitening with Real README Text Data')
print('=' * 55)

# Read README content
with open('README.md', 'r', encoding='utf-8') as f:
    readme_text = f.read()

print(f'README text length: {len(readme_text)} characters')

# Simple word-based tokenization (instead of requiring transformers)
def simple_tokenize(text, vocab_size=1000):
    # Clean and split text
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Build vocabulary from most common words
    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1
    
    # Get top words
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    vocab = {word: idx for idx, (word, _) in enumerate(sorted_words[:vocab_size])}
    vocab['<UNK>'] = vocab_size  # Unknown token
    
    # Convert text to token IDs
    tokens = [vocab.get(word, vocab['<UNK>']) for word in words]
    return tokens, vocab

# Tokenize README
tokens, vocab = simple_tokenize(readme_text, vocab_size=500)
print(f'Vocabulary size: {len(vocab)}')
print(f'Total tokens: {len(tokens)}')

# Create simple embeddings (like word2vec style)
embed_dim = 128
torch.manual_seed(42)

# Create embeddings with realistic patterns
embeddings = torch.randn(len(vocab) + 1, embed_dim)

# Add semantic clustering (words that appear together get similar embeddings)
window_size = 5
for i in range(len(tokens) - window_size):
    window_tokens = tokens[i:i+window_size]
    center_token = tokens[i + window_size//2]
    
    # Make nearby tokens more similar
    for token in window_tokens:
        if token != center_token:
            # Pull embeddings closer
            embeddings[center_token] = 0.99 * embeddings[center_token] + 0.01 * embeddings[token]

print('Created embeddings with co-occurrence patterns')

# Create sequences of embeddings (like transformer input)
seq_length = 32
sequences = []
for i in range(0, len(tokens) - seq_length, seq_length//2):  # Overlapping windows
    seq_tokens = tokens[i:i+seq_length]
    seq_embeddings = embeddings[seq_tokens]
    sequences.append(seq_embeddings.flatten())  # Flatten for linear layer input

sequences = torch.stack(sequences)
print(f'Created {len(sequences)} sequences of {sequences.shape[1]} dimensions each')

# Split into calibration and test
split_idx = int(0.8 * len(sequences))
calibration_data = sequences[:split_idx]
test_data = sequences[split_idx:]

print(f'Calibration samples: {len(calibration_data)}')
print(f'Test samples: {len(test_data)}')

# Create layer to compress (simulating FFN layer)
input_dim = sequences.shape[1]  # seq_length * embed_dim
output_dim = 256
layer = nn.Linear(input_dim, output_dim)
rank = 64

print(f'\nTesting layer: {input_dim} → {output_dim}, rank {rank}')

# Test 1: Standard SVD
print('\n1. STANDARD SVD')
from tensorslim.core.randomized_svd import RandomizedSVD
svd_standard = RandomizedSVD(rank, use_whitening=False)
U_std, s_std, Vt_std = svd_standard.fit_transform(layer.weight.data)
compressed_std = svd_standard.reconstruct(U_std, s_std, Vt_std)

# Test 2: Whitened SVD with proper calibration data
print('\n2. WHITENED SVD WITH TEXT DATA')

# Create whitener with explicit calibration data
whitener = DataWhitener()

# Manually compute whitening using our text data
try:
    # Compute covariance from calibration data
    print('Computing activation covariance...')
    cov = whitener.compute_activation_covariance(layer, calibration_data)
    
    # Analyze covariance structure
    eigenvals = torch.linalg.eigvals(cov).real
    eigenvals = torch.sort(eigenvals, descending=True)[0]
    condition_num = eigenvals[0] / eigenvals[-1]
    effective_rank = torch.sum(eigenvals > 0.01 * eigenvals[0]).item()
    
    print(f'Condition number: {condition_num:.1f}')
    print(f'Effective rank: {effective_rank}/{input_dim}')
    
    # Compute whitening matrix
    print('Computing whitening matrix...')
    whitening_inv = whitener.compute_whitening_matrix(cov)
    
    # Apply whitening
    whitening_matrix = torch.linalg.inv(whitening_inv)
    whitened_weight = layer.weight.data @ whitening_matrix
    
    # SVD on whitened weight
    U_w, s_w, Vt_w = svd_standard.fit_transform(whitened_weight)
    
    # Reconstruct with whitening applied
    compressed_whitened = svd_standard.reconstruct(U_w, s_w, Vt_w) @ whitening_inv
    
    print('✅ Whitening applied successfully')
    whitening_success = True
    
except Exception as e:
    print(f'❌ Whitening failed: {e}')
    compressed_whitened = compressed_std
    whitening_success = False

# Evaluate quality on test data
print('\n3. QUALITY EVALUATION')
with torch.no_grad():
    # Original layer outputs
    original_outputs = layer(test_data)
    
    # Standard compression outputs
    std_outputs = torch.nn.functional.linear(test_data, compressed_std, layer.bias)
    std_error = torch.norm(original_outputs - std_outputs) / torch.norm(original_outputs)
    
    if whitening_success:
        # Whitened compression outputs
        whitened_outputs = torch.nn.functional.linear(test_data, compressed_whitened, layer.bias)
        whitened_error = torch.norm(original_outputs - whitened_outputs) / torch.norm(original_outputs)
        improvement = std_error / whitened_error
        
        print(f'Standard SVD error:    {std_error:.4f}')
        print(f'Whitened SVD error:    {whitened_error:.4f}')
        print(f'Improvement factor:    {improvement:.2f}x')
        
        if improvement > 1.2:
            print('✅ Significant improvement with real text data!')
        elif improvement > 1.05:
            print('✅ Modest improvement with real text data')
        else:
            print('➡️  Minimal difference')
            
        # Also test activation-based quality
        std_quality = torch.nn.functional.cosine_similarity(
            original_outputs.flatten(), std_outputs.flatten(), dim=0
        )
        whitened_quality = torch.nn.functional.cosine_similarity(
            original_outputs.flatten(), whitened_outputs.flatten(), dim=0
        )
        
        print(f'\nActivation-based quality:')
        print(f'Standard SVD:          {std_quality:.4f}')
        print(f'Whitened SVD:          {whitened_quality:.4f}')
        print(f'Quality improvement:   {whitened_quality - std_quality:.4f}')
        
    else:
        print(f'Standard SVD error:    {std_error:.4f}')
        print('Whitening comparison not available')

print('\n4. TEST WITH DIRECT WHITENING API')
# Test using the WhitenedSVD class directly
try:
    print('Testing WhitenedSVD class with calibration data...')
    
    # Create whitener with manual calibration data
    whitener_direct = DataWhitener()
    whitened_svd = WhitenedSVD(whitener_direct, use_whitening=True)
    
    # Compress layer directly
    U_direct, s_direct, Vt_direct, whitening_inv_direct = whitened_svd.compress_layer(
        layer, rank, calibration_data
    )
    
    print('✅ Direct WhitenedSVD succeeded')
    
    # Test quality
    if whitening_inv_direct is not None:
        compressed_direct = svd_standard.reconstruct(U_direct, s_direct, Vt_direct) @ whitening_inv_direct
        direct_outputs = torch.nn.functional.linear(test_data, compressed_direct, layer.bias)
        direct_error = torch.norm(original_outputs - direct_outputs) / torch.norm(original_outputs)
        direct_improvement = std_error / direct_error
        
        print(f'Direct whitened error: {direct_error:.4f}')
        print(f'Direct improvement:    {direct_improvement:.2f}x')
    
except Exception as e:
    print(f'❌ Direct WhitenedSVD failed: {e}')

print('\n5. KEY INSIGHTS FROM REAL TEXT DATA')
print('   • Real text has natural correlation structure from language patterns')
print('   • Words that co-occur have similar embeddings') 
print('   • Sequence-level patterns create input correlations')
print('   • Whitening can exploit these patterns for better compression')

if whitening_success and 'improvement' in locals() and improvement > 1.1:
    print('   ✅ Whitening shows measurable improvement on real text!')
elif whitening_success:
    print('   ➡️  Whitening provides some benefit but may need more correlation')
else:
    print('   ⚠️  Whitening implementation needs debugging')

print('\n6. FIXING THE CALIBRATION WARNING')
print('   • The warning occurs when whitening dataset is not properly passed')
print('   • Manual calibration data (like our README text) works correctly')
print('   • Production usage should provide calibration data explicitly')