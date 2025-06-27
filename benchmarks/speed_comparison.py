"""
Speed comparison benchmark for TensorSlim compression methods.

This benchmark compares the speed of TensorSlim compression against
standard compression methods across different model sizes and configurations.
"""

import torch
import torch.nn as nn
import time
import json
import argparse
from pathlib import Path
import numpy as np

# Import TensorSlim
import sys
sys.path.insert(0, '../src')

import tensorslim
from tensorslim import compress_model, RandomizedSVD
from tensorslim.core.randomized_svd import randomized_svd


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Speed comparison benchmark")
    
    parser.add_argument(
        "--output", 
        type=str, 
        default="speed_results.json",
        help="Output file for results"
    )
    
    parser.add_argument(
        "--matrix-sizes",
        nargs="+",
        type=int,
        default=[100, 500, 1000, 2000, 5000],
        help="Matrix sizes to test"
    )
    
    parser.add_argument(
        "--ranks",
        nargs="+", 
        type=int,
        default=[10, 25, 50, 100],
        help="Ranks to test"
    )
    
    parser.add_argument(
        "--num-trials",
        type=int,
        default=5,
        help="Number of trials per configuration"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (cpu, cuda, auto)"
    )
    
    return parser.parse_args()


def setup_device(device_arg):
    """Setup computation device."""
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)
    
    print(f"Using device: {device}")
    return device


def benchmark_standard_svd(matrix, rank, num_trials=5):
    """Benchmark standard PyTorch SVD."""
    times = []
    
    for _ in range(num_trials):
        start_time = time.time()
        
        U, S, Vt = torch.svd(matrix)
        U_truncated = U[:, :rank]
        S_truncated = S[:rank]
        Vt_truncated = Vt[:rank, :]
        
        end_time = time.time()
        times.append(end_time - start_time)
    
    return {
        'avg_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times)
    }


def benchmark_randomized_svd(matrix, rank, num_trials=5, **kwargs):
    """Benchmark TensorSlim randomized SVD."""
    times = []
    
    for _ in range(num_trials):
        start_time = time.time()
        
        U, S, Vt = randomized_svd(matrix, rank, **kwargs)
        
        end_time = time.time()
        times.append(end_time - start_time)
    
    return {
        'avg_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times)
    }


def benchmark_adaptive_svd(matrix, rank, num_trials=5):
    """Benchmark TensorSlim adaptive SVD."""
    from tensorslim.core.randomized_svd import AdaptiveRandomizedSVD
    
    times = []
    
    for _ in range(num_trials):
        start_time = time.time()
        
        svd = AdaptiveRandomizedSVD(rank=rank, target_quality=0.95)
        U, S, Vt = svd.fit_transform(matrix)
        
        end_time = time.time()
        times.append(end_time - start_time)
    
    return {
        'avg_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times)
    }


def calculate_reconstruction_error(original, U, S, Vt):
    """Calculate reconstruction error."""
    reconstructed = U @ torch.diag(S) @ Vt
    error = torch.norm(original - reconstructed, p='fro') / torch.norm(original, p='fro')
    return error.item()


def benchmark_matrix_size(size, rank, device, num_trials=5):
    """Benchmark all methods for a given matrix size and rank."""
    print(f"  Benchmarking {size}x{size} matrix, rank={rank}")
    
    # Create test matrix
    torch.manual_seed(42)  # For reproducibility
    matrix = torch.randn(size, size, device=device)
    
    results = {
        'matrix_size': size,
        'rank': rank,
        'device': str(device)
    }
    
    # Standard SVD (if not too large)
    if size <= 2000:  # Avoid OOM for large matrices
        try:
            std_results = benchmark_standard_svd(matrix.cpu(), rank, num_trials)
            results['standard_svd'] = std_results
            
            # Calculate error for standard SVD
            U, S, Vt = torch.svd(matrix.cpu())
            error = calculate_reconstruction_error(
                matrix.cpu(), U[:, :rank], S[:rank], Vt[:rank, :]
            )
            results['standard_svd']['reconstruction_error'] = error
            
        except Exception as e:
            print(f"    Standard SVD failed: {e}")
            results['standard_svd'] = {'error': str(e)}
    else:
        results['standard_svd'] = {'skipped': 'Matrix too large'}
    
    # Randomized SVD
    try:
        rand_results = benchmark_randomized_svd(matrix, rank, num_trials)
        results['randomized_svd'] = rand_results
        
        # Calculate error for randomized SVD
        U, S, Vt = randomized_svd(matrix, rank)
        error = calculate_reconstruction_error(matrix, U, S, Vt)
        results['randomized_svd']['reconstruction_error'] = error
        
    except Exception as e:
        print(f"    Randomized SVD failed: {e}")
        results['randomized_svd'] = {'error': str(e)}
    
    # Adaptive SVD
    try:
        adaptive_results = benchmark_adaptive_svd(matrix, rank, num_trials)
        results['adaptive_svd'] = adaptive_results
        
        # Calculate error for adaptive SVD
        from tensorslim.core.randomized_svd import AdaptiveRandomizedSVD
        svd = AdaptiveRandomizedSVD(rank=rank, target_quality=0.95)
        U, S, Vt = svd.fit_transform(matrix)
        error = calculate_reconstruction_error(matrix, U, S, Vt)
        results['adaptive_svd']['reconstruction_error'] = error
        
    except Exception as e:
        print(f"    Adaptive SVD failed: {e}")
        results['adaptive_svd'] = {'error': str(e)}
    
    # Calculate speedups
    if 'standard_svd' in results and 'avg_time' in results['standard_svd']:
        std_time = results['standard_svd']['avg_time']
        
        if 'randomized_svd' in results and 'avg_time' in results['randomized_svd']:
            rand_time = results['randomized_svd']['avg_time']
            results['randomized_speedup'] = std_time / rand_time
        
        if 'adaptive_svd' in results and 'avg_time' in results['adaptive_svd']:
            adapt_time = results['adaptive_svd']['avg_time']
            results['adaptive_speedup'] = std_time / adapt_time
    
    return results


def benchmark_model_compression(device, num_trials=3):
    """Benchmark model compression speed."""
    print("Benchmarking model compression...")
    
    # Create test models of different sizes
    models = {
        'small': nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        ),
        'medium': nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        ),
        'large': nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    }
    
    compression_results = {}
    
    for model_name, model in models.items():
        print(f"  Compressing {model_name} model...")
        
        model = model.to(device)
        original_params = sum(p.numel() for p in model.parameters())
        
        # Test different compression ratios
        ratios = [0.25, 0.5, 0.75]
        model_results = {
            'original_parameters': original_params,
            'compression_ratios': {}
        }
        
        for ratio in ratios:
            ratio_times = []
            
            for _ in range(num_trials):
                start_time = time.time()
                
                compressed_model = compress_model(
                    model, 
                    compression_ratio=ratio, 
                    inplace=False
                )
                
                end_time = time.time()
                ratio_times.append(end_time - start_time)
            
            compressed_params = sum(p.numel() for p in compressed_model.parameters())
            actual_ratio = original_params / compressed_params
            
            model_results['compression_ratios'][ratio] = {
                'avg_compression_time': np.mean(ratio_times),
                'std_compression_time': np.std(ratio_times),
                'compressed_parameters': compressed_params,
                'actual_compression_ratio': actual_ratio,
                'parameter_reduction': 1 - (compressed_params / original_params)
            }
        
        compression_results[model_name] = model_results
    
    return compression_results


def generate_report(results, output_file):
    """Generate a human-readable report."""
    report_file = Path(output_file).with_suffix('.txt')
    
    with open(report_file, 'w') as f:
        f.write("TensorSlim Speed Comparison Benchmark Report\n")
        f.write("=" * 50 + "\n\n")
        
        # Matrix benchmarks
        if 'matrix_benchmarks' in results:
            f.write("Matrix SVD Benchmarks\n")
            f.write("-" * 30 + "\n\n")
            
            for result in results['matrix_benchmarks']:
                size = result['matrix_size']
                rank = result['rank']
                
                f.write(f"Matrix Size: {size}x{size}, Rank: {rank}\n")
                
                # Standard SVD
                if 'standard_svd' in result and 'avg_time' in result['standard_svd']:
                    std = result['standard_svd']
                    f.write(f"  Standard SVD: {std['avg_time']:.4f}s ± {std['std_time']:.4f}s\n")
                    f.write(f"    Reconstruction Error: {std.get('reconstruction_error', 'N/A'):.6f}\n")
                
                # Randomized SVD
                if 'randomized_svd' in result and 'avg_time' in result['randomized_svd']:
                    rand = result['randomized_svd']
                    f.write(f"  Randomized SVD: {rand['avg_time']:.4f}s ± {rand['std_time']:.4f}s\n")
                    f.write(f"    Reconstruction Error: {rand.get('reconstruction_error', 'N/A'):.6f}\n")
                    if 'randomized_speedup' in result:
                        f.write(f"    Speedup: {result['randomized_speedup']:.2f}x\n")
                
                # Adaptive SVD
                if 'adaptive_svd' in result and 'avg_time' in result['adaptive_svd']:
                    adapt = result['adaptive_svd']
                    f.write(f"  Adaptive SVD: {adapt['avg_time']:.4f}s ± {adapt['std_time']:.4f}s\n")
                    f.write(f"    Reconstruction Error: {adapt.get('reconstruction_error', 'N/A'):.6f}\n")
                    if 'adaptive_speedup' in result:
                        f.write(f"    Speedup: {result['adaptive_speedup']:.2f}x\n")
                
                f.write("\n")
        
        # Model compression benchmarks
        if 'model_compression' in results:
            f.write("\nModel Compression Benchmarks\n")
            f.write("-" * 30 + "\n\n")
            
            for model_name, model_data in results['model_compression'].items():
                f.write(f"Model: {model_name.capitalize()}\n")
                f.write(f"Original Parameters: {model_data['original_parameters']:,}\n")
                
                for ratio, ratio_data in model_data['compression_ratios'].items():
                    f.write(f"\n  Compression Ratio: {ratio}\n")
                    f.write(f"    Time: {ratio_data['avg_compression_time']:.4f}s ± {ratio_data['std_compression_time']:.4f}s\n")
                    f.write(f"    Actual Ratio: {ratio_data['actual_compression_ratio']:.2f}x\n")
                    f.write(f"    Parameter Reduction: {ratio_data['parameter_reduction']:.1%}\n")
                
                f.write("\n")
        
        f.write("\nBenchmark completed successfully!\n")
    
    print(f"Report saved to: {report_file}")


def main():
    """Main benchmark function."""
    args = parse_args()
    
    print("🚀 TensorSlim Speed Comparison Benchmark")
    print("=" * 50)
    
    device = setup_device(args.device)
    
    print(f"\nConfiguration:")
    print(f"  Matrix sizes: {args.matrix_sizes}")
    print(f"  Ranks: {args.ranks}")
    print(f"  Trials per config: {args.num_trials}")
    print(f"  Device: {device}")
    
    results = {
        'config': {
            'matrix_sizes': args.matrix_sizes,
            'ranks': args.ranks,
            'num_trials': args.num_trials,
            'device': str(device)
        },
        'matrix_benchmarks': [],
        'model_compression': {}
    }
    
    # Matrix SVD benchmarks
    print(f"\n🔢 Running matrix SVD benchmarks...")
    
    total_configs = len(args.matrix_sizes) * len(args.ranks)
    current_config = 0
    
    for size in args.matrix_sizes:
        for rank in args.ranks:
            if rank >= size:  # Skip invalid configurations
                continue
                
            current_config += 1
            print(f"[{current_config}/{total_configs}] Size: {size}, Rank: {rank}")
            
            try:
                result = benchmark_matrix_size(size, rank, device, args.num_trials)
                results['matrix_benchmarks'].append(result)
            except Exception as e:
                print(f"  ❌ Failed: {e}")
                continue
    
    # Model compression benchmarks
    print(f"\n🏗️  Running model compression benchmarks...")
    
    try:
        compression_results = benchmark_model_compression(device, args.num_trials)
        results['model_compression'] = compression_results
    except Exception as e:
        print(f"❌ Model compression benchmark failed: {e}")
    
    # Save results
    print(f"\n💾 Saving results...")
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Results saved to: {output_path}")
    
    # Generate report
    generate_report(results, args.output)
    
    # Summary
    print(f"\n📊 Summary:")
    
    if results['matrix_benchmarks']:
        # Find best speedup
        speedups = [r.get('randomized_speedup', 0) for r in results['matrix_benchmarks']]
        max_speedup = max([s for s in speedups if s > 0], default=0)
        
        if max_speedup > 0:
            print(f"  Maximum speedup achieved: {max_speedup:.1f}x")
        
        # Average reconstruction error
        errors = [r['randomized_svd'].get('reconstruction_error', 0) 
                 for r in results['matrix_benchmarks'] 
                 if 'randomized_svd' in r and 'reconstruction_error' in r['randomized_svd']]
        
        if errors:
            avg_error = np.mean(errors)
            print(f"  Average reconstruction error: {avg_error:.6f}")
    
    if results['model_compression']:
        print(f"  Model compression benchmarks: {len(results['model_compression'])} models tested")
    
    print(f"\n🎉 Benchmark completed successfully!")


if __name__ == "__main__":
    main()