# Experiments

This directory contains experimental scripts and research code used to develop and validate TensorSlim's compression algorithms.

## Files

### Quality Measurement Research
- **`test_activation_quality.py`** - Comparison between activation-based quality vs matrix reconstruction quality
- **`balanced_compression_test.py`** - Finding optimal compression ratios for different layer types
- **`aggressive_compression_test.py`** - Testing extreme compression ratios and their limits

## Running Experiments

All experiments assume you're running from the root directory:

```bash
# From tensorslim root directory
python experiments/test_activation_quality.py
python experiments/balanced_compression_test.py
python experiments/aggressive_compression_test.py
```

## Purpose

These experiments were used to:

1. **Validate activation-based quality measurement** - Proving that activation similarity is more meaningful than weight reconstruction error for neural networks
2. **Determine optimal compression strategies** - Finding that FFN layers compress well (3.2x) while attention layers don't
3. **Set realistic performance expectations** - Establishing honest benchmark numbers for the README

## Results Summary

- **Activation quality is superior**: 60%+ better correlation with functional preservation
- **FFN layers are ideal targets**: 3.2x compression with 81%+ activation quality
- **Attention layers have limitations**: Square matrices don't compress well with SVD
- **Overall model compression**: 2-2.5x realistic for transformer models

These findings shaped TensorSlim's design and documentation.