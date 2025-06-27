#!/usr/bin/env python3
"""
Benchmark script comparing SRHT + Whitening vs standard Gaussian SVD.

This script evaluates the performance improvements from:
1. SRHT (Subsampled Randomized Hadamard Transform) vs Gaussian random matrices
2. Data whitening vs standard SVD compression
3. Combined SRHT + Whitening vs baseline approaches

Results are saved to benchmarks/results/ for analysis.
"""

import torch
import torch.nn as nn
import time
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import logging
from tqdm import tqdm
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tensorslim.core.randomized_svd import RandomizedSVD
from tensorslim.core.srht_utils import compare_srht_vs_gaussian, SRHTOperator
from tensorslim.core.whitening import (
    DataWhitener, 
    WhitenedSVD, 
    RandomDataset,
    compare_whitening_vs_standard
)
from tensorslim.core.layers import convert_layer_to_slim

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_layers() -> List[Tuple[str, nn.Linear]]:
    """Create a variety of test layers representing different transformer components."""
    torch.manual_seed(42)
    
    layers = [
        # BERT-like layers
        ("BERT Attention Q", nn.Linear(768, 768)),
        ("BERT Attention K", nn.Linear(768, 768)),
        ("BERT Attention V", nn.Linear(768, 768)),
        ("BERT Attention Output", nn.Linear(768, 768)),
        ("BERT FFN Up", nn.Linear(768, 3072)),
        ("BERT FFN Down", nn.Linear(3072, 768)),
        
        # GPT-2 like layers
        ("GPT-2 Attention", nn.Linear(1024, 1024)),
        ("GPT-2 FFN Up", nn.Linear(1024, 4096)),
        ("GPT-2 FFN Down", nn.Linear(4096, 1024)),
        
        # Larger models (GPT-3 style)
        ("Large Attention", nn.Linear(2048, 2048)),
        ("Large FFN Up", nn.Linear(2048, 8192)),
        ("Large FFN Down", nn.Linear(8192, 2048)),
    ]
    
    return layers


def benchmark_srht_vs_gaussian(
    matrix_sizes: List[Tuple[int, int]],
    ranks: List[int],
    device: torch.device = torch.device('cpu'),
    n_trials: int = 5
) -> Dict:
    """Benchmark SRHT vs Gaussian random matrices."""
    
    logger.info("🚀 Benchmarking SRHT vs Gaussian Random Matrices")
    
    results = {
        'srht_vs_gaussian': [],
        'device': str(device),
        'n_trials': n_trials
    }
    
    for (m, n), rank in tqdm(
        [(size, rank) for size in matrix_sizes for rank in ranks],
        desc="SRHT vs Gaussian"
    ):
        if rank >= min(m, n):
            continue
            
        try:
            comparison = compare_srht_vs_gaussian(
                matrix_size=(m, n),
                target_rank=rank,
                device=device,
                n_trials=n_trials
            )
            
            comparison.update({
                'matrix_size': (m, n),
                'rank': rank,
                'matrix_elements': m * n
            })
            
            results['srht_vs_gaussian'].append(comparison)
            
            logger.info(
                f"Size {m}x{n}, Rank {rank}: "
                f"SRHT {comparison['speedup']:.1f}x faster, "
                f"Error ratio {comparison['error_ratio']:.3f}"
            )
            
        except Exception as e:
            logger.warning(f"Failed for size {m}x{n}, rank {rank}: {e}")
    
    return results


def benchmark_whitening_compression(
    test_layers: List[Tuple[str, nn.Linear]],
    ranks: List[int],
    device: torch.device = torch.device('cpu'),
    n_calibration_samples: int = 128
) -> Dict:
    """Benchmark whitening vs standard SVD compression."""
    
    logger.info("🎯 Benchmarking Whitening vs Standard SVD")
    
    results = {
        'whitening_comparison': [],
        'device': str(device),
        'n_calibration_samples': n_calibration_samples
    }
    
    for layer_name, layer in tqdm(test_layers, desc="Whitening Benchmark"):
        layer = layer.to(device)
        
        # Generate calibration and test data
        calibration_data = torch.randn(
            n_calibration_samples, layer.in_features, device=device
        )
        test_data = torch.randn(64, layer.in_features, device=device)
        
        for rank in ranks:
            if rank >= min(layer.weight.shape):
                continue
                
            try:
                # Compare whitening vs standard
                comparison = compare_whitening_vs_standard(
                    layer, rank, calibration_data, test_data
                )
                
                comparison.update({
                    'layer_name': layer_name,
                    'layer_shape': layer.weight.shape,
                    'rank': rank,
                    'compression_ratio': (layer.weight.numel() / 
                                       (rank * (layer.weight.shape[0] + layer.weight.shape[1] + 1)))
                })
                
                results['whitening_comparison'].append(comparison)
                
                logger.info(
                    f"{layer_name}, Rank {rank}: "
                    f"Whitening {comparison['improvement_factor']:.2f}x better MSE"
                )
                
            except Exception as e:
                logger.warning(f"Failed for {layer_name}, rank {rank}: {e}")
    
    return results


def benchmark_end_to_end_compression(
    test_layers: List[Tuple[str, nn.Linear]],
    rank_ratios: List[float],
    device: torch.device = torch.device('cpu'),
    n_trials: int = 3
) -> Dict:
    """Benchmark end-to-end compression with different configurations."""
    
    logger.info("⚡ Benchmarking End-to-End Compression")
    
    configurations = [
        {"name": "Baseline (Gaussian)", "use_srht": False, "use_whitening": False},
        {"name": "SRHT Only", "use_srht": True, "use_whitening": False},
        {"name": "Whitening Only", "use_srht": False, "use_whitening": True},
        {"name": "SRHT + Whitening", "use_srht": True, "use_whitening": True},
    ]
    
    results = {
        'end_to_end': [],
        'device': str(device),
        'n_trials': n_trials
    }
    
    for layer_name, layer in tqdm(test_layers, desc="End-to-End"):
        layer = layer.to(device)
        
        # Generate test data
        test_input = torch.randn(32, layer.in_features, device=device)
        
        for rank_ratio in rank_ratios:
            rank = max(1, int(min(layer.weight.shape) * rank_ratio))
            
            for config in configurations:
                try:
                    # Time compression
                    start_time = time.time()
                    
                    for _ in range(n_trials):
                        compressed_layer = convert_layer_to_slim(
                            layer,
                            rank=rank,
                            use_srht=config["use_srht"],
                            use_whitening=config["use_whitening"]
                        )
                    
                    compression_time = (time.time() - start_time) / n_trials
                    
                    # Evaluate quality
                    with torch.no_grad():
                        original_output = layer(test_input)
                        compressed_output = compressed_layer(test_input)
                        
                        mse_loss = torch.nn.functional.mse_loss(
                            compressed_output, original_output
                        ).item()
                        
                        relative_error = (
                            torch.norm(compressed_output - original_output) / 
                            torch.norm(original_output)
                        ).item()
                    
                    # Calculate compression metrics
                    original_params = layer.weight.numel()
                    if layer.bias is not None:
                        original_params += layer.bias.numel()
                    
                    compressed_params = sum(p.numel() for p in compressed_layer.parameters())
                    compression_ratio = original_params / compressed_params
                    
                    result = {
                        'layer_name': layer_name,
                        'layer_shape': layer.weight.shape,
                        'rank': rank,
                        'rank_ratio': rank_ratio,
                        'config_name': config["name"],
                        'use_srht': config["use_srht"],
                        'use_whitening': config["use_whitening"],
                        'compression_time_ms': compression_time * 1000,
                        'mse_loss': mse_loss,
                        'relative_error': relative_error,
                        'compression_ratio': compression_ratio,
                        'original_params': original_params,
                        'compressed_params': compressed_params
                    }
                    
                    results['end_to_end'].append(result)
                    
                    logger.info(
                        f"{layer_name} ({config['name']}): "
                        f"{compression_ratio:.1f}x compression, "
                        f"MSE {mse_loss:.2e}, "
                        f"Time {compression_time*1000:.1f}ms"
                    )
                    
                except Exception as e:
                    logger.warning(
                        f"Failed for {layer_name} with {config['name']}: {e}"
                    )
    
    return results


def benchmark_memory_usage(
    matrix_sizes: List[Tuple[int, int]],
    ranks: List[int]
) -> Dict:
    """Benchmark memory usage of different approaches."""
    
    logger.info("💾 Benchmarking Memory Usage")
    
    results = {
        'memory_usage': [],
    }
    
    for (m, n), rank in tqdm(
        [(size, rank) for size in matrix_sizes for rank in ranks],
        desc="Memory Usage"
    ):
        if rank >= min(m, n):
            continue
        
        # Gaussian matrix memory
        gaussian_memory = rank * n * 4  # float32
        
        # SRHT memory
        srht_op = SRHTOperator(n, rank)
        srht_memory = srht_op.memory_usage()
        
        # Compressed layer memory
        compressed_memory = rank * (m + n + 1) * 4  # U + s + Vt
        original_memory = m * n * 4
        
        result = {
            'matrix_size': (m, n),
            'rank': rank,
            'gaussian_memory_bytes': gaussian_memory,
            'srht_memory_bytes': srht_memory,
            'memory_reduction_factor': gaussian_memory / srht_memory,
            'compressed_layer_memory_bytes': compressed_memory,
            'original_layer_memory_bytes': original_memory,
            'layer_compression_factor': original_memory / compressed_memory
        }
        
        results['memory_usage'].append(result)
    
    return results


def generate_summary_report(all_results: Dict) -> Dict:
    """Generate a summary report from all benchmark results."""
    
    summary = {
        'srht_summary': {},
        'whitening_summary': {},
        'end_to_end_summary': {},
        'memory_summary': {}
    }
    
    # SRHT summary
    if 'srht_vs_gaussian' in all_results:
        srht_data = all_results['srht_vs_gaussian']
        if srht_data:
            speedups = [r['speedup'] for r in srht_data]
            error_ratios = [r['error_ratio'] for r in srht_data]
            memory_ratios = [r['memory_ratio'] for r in srht_data]
            
            summary['srht_summary'] = {
                'avg_speedup': np.mean(speedups),
                'median_speedup': np.median(speedups),
                'max_speedup': np.max(speedups),
                'avg_error_ratio': np.mean(error_ratios),
                'avg_memory_reduction': np.mean(memory_ratios),
                'n_comparisons': len(srht_data)
            }
    
    # Whitening summary
    if 'whitening_comparison' in all_results:
        whitening_data = all_results['whitening_comparison']
        if whitening_data:
            improvements = [r['improvement_factor'] for r in whitening_data]
            
            summary['whitening_summary'] = {
                'avg_improvement_factor': np.mean(improvements),
                'median_improvement_factor': np.median(improvements),
                'max_improvement_factor': np.max(improvements),
                'n_comparisons': len(whitening_data)
            }
    
    # End-to-end summary
    if 'end_to_end' in all_results:
        e2e_data = all_results['end_to_end']
        if e2e_data:
            # Group by configuration
            configs = {}
            for result in e2e_data:
                config_name = result['config_name']
                if config_name not in configs:
                    configs[config_name] = []
                configs[config_name].append(result)
            
            for config_name, config_results in configs.items():
                compression_times = [r['compression_time_ms'] for r in config_results]
                mse_losses = [r['mse_loss'] for r in config_results]
                compression_ratios = [r['compression_ratio'] for r in config_results]
                
                summary['end_to_end_summary'][config_name] = {
                    'avg_compression_time_ms': np.mean(compression_times),
                    'avg_mse_loss': np.mean(mse_losses),
                    'avg_compression_ratio': np.mean(compression_ratios),
                    'n_layers': len(config_results)
                }
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="SRHT + Whitening Benchmark")
    parser.add_argument("--device", default="cpu", help="Device to run on (cpu/cuda)")
    parser.add_argument("--output-dir", default="benchmarks/results", help="Output directory")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark")
    parser.add_argument("--skip-srht", action="store_true", help="Skip SRHT benchmark")
    parser.add_argument("--skip-whitening", action="store_true", help="Skip whitening benchmark")
    parser.add_argument("--skip-e2e", action="store_true", help="Skip end-to-end benchmark")
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = torch.device('cpu')
    
    logger.info(f"Running benchmarks on device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure benchmark parameters
    if args.quick:
        matrix_sizes = [(512, 512), (768, 3072), (1024, 4096)]
        ranks = [64, 128]
        rank_ratios = [0.25, 0.5]
        n_trials = 2
        n_calibration_samples = 64
    else:
        matrix_sizes = [(512, 512), (768, 768), (768, 3072), (1024, 1024), 
                       (1024, 4096), (2048, 2048), (2048, 8192)]
        ranks = [32, 64, 128, 256]
        rank_ratios = [0.125, 0.25, 0.5, 0.75]
        n_trials = 5
        n_calibration_samples = 256
    
    # Create test layers
    test_layers = create_test_layers()
    
    # Run benchmarks
    all_results = {
        'benchmark_config': {
            'device': str(device),
            'quick_mode': args.quick,
            'matrix_sizes': matrix_sizes,
            'ranks': ranks,
            'rank_ratios': rank_ratios,
            'n_trials': n_trials,
            'n_calibration_samples': n_calibration_samples
        }
    }
    
    try:
        # SRHT vs Gaussian benchmark
        if not args.skip_srht:
            srht_results = benchmark_srht_vs_gaussian(
                matrix_sizes, ranks, device, n_trials
            )
            all_results.update(srht_results)
        
        # Whitening benchmark
        if not args.skip_whitening:
            whitening_results = benchmark_whitening_compression(
                test_layers, ranks, device, n_calibration_samples
            )
            all_results.update(whitening_results)
        
        # End-to-end benchmark
        if not args.skip_e2e:
            e2e_results = benchmark_end_to_end_compression(
                test_layers, rank_ratios, device, n_trials
            )
            all_results.update(e2e_results)
        
        # Memory usage benchmark
        memory_results = benchmark_memory_usage(matrix_sizes, ranks)
        all_results.update(memory_results)
        
        # Generate summary
        summary = generate_summary_report(all_results)
        all_results['summary'] = summary
        
        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f"srht_whitening_benchmark_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"✅ Benchmark complete! Results saved to {results_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("📊 BENCHMARK SUMMARY")
        print("="*60)
        
        if 'srht_summary' in summary and summary['srht_summary']:
            s = summary['srht_summary']
            print(f"🚀 SRHT vs Gaussian:")
            print(f"   Average speedup: {s['avg_speedup']:.1f}x")
            print(f"   Max speedup: {s['max_speedup']:.1f}x")
            print(f"   Memory reduction: {s['avg_memory_reduction']:.1f}x")
            print(f"   Error ratio: {s['avg_error_ratio']:.3f}")
        
        if 'whitening_summary' in summary and summary['whitening_summary']:
            s = summary['whitening_summary']
            print(f"🎯 Whitening vs Standard:")
            print(f"   Average improvement: {s['avg_improvement_factor']:.2f}x")
            print(f"   Max improvement: {s['max_improvement_factor']:.2f}x")
        
        if 'end_to_end_summary' in summary:
            print(f"⚡ End-to-End Performance:")
            for config, stats in summary['end_to_end_summary'].items():
                print(f"   {config}:")
                print(f"     Avg compression time: {stats['avg_compression_time_ms']:.1f}ms")
                print(f"     Avg compression ratio: {stats['avg_compression_ratio']:.1f}x")
                print(f"     Avg MSE loss: {stats['avg_mse_loss']:.2e}")
        
        print("="*60)
        
    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise


if __name__ == "__main__":
    main()