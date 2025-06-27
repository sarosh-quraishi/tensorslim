# TensorSlim Benchmarks

This directory contains benchmarking and analysis scripts for TensorSlim performance evaluation.

## Files

### Core Benchmarks
- **`speed_comparison.py`** - Compares randomized SVD speed vs traditional SVD across matrix sizes
- **`quality_analysis.py`** - Analyzes compression quality across different configurations

### Model-Specific Benchmarks  
- **`real_hf_benchmark.py`** - Real HuggingFace model compression benchmarks (DistilBERT, BERT)
- **`quality_test.py`** - Quality vs compression ratio analysis for different approaches
- **`test_improved_quality.py`** - Tests for achieving <2% quality loss targets

### Results
- **`speed_results.json`** - JSON output from speed comparison benchmarks
- **`speed_results.txt`** - Human-readable speed benchmark report

## Running Benchmarks

```bash
# Speed comparison benchmark
python speed_comparison.py --matrix-sizes 500 1000 --ranks 50 100

# Real HuggingFace model benchmark  
python real_hf_benchmark.py

# Quality analysis
python quality_test.py
```

## Benchmark Results Summary

Based on real testing:
- **Speed**: 3-14x faster than traditional SVD (depending on matrix size/rank)
- **Compression**: 3-5x model size reduction typically achieved
- **Quality**: 8-14% quality loss on real transformer models