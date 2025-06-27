"""
Example: Comprehensive benchmark comparison of compression methods.

This example compares TensorSlim with other compression methods and
analyzes compression vs quality tradeoffs across different models.
"""

import torch
import torch.nn as nn
import time
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

# Import TensorSlim
import sys
sys.path.insert(0, '../src')

import tensorslim
from tensorslim import compress_model, analyze_model_compression
from tensorslim.utils import evaluate_compression_quality, CompressionMetrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark TensorSlim compression methods")
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./benchmark_results",
        help="Directory to save results"
    )
    
    parser.add_argument(
        "--compression-ratios",
        nargs="+",
        type=float,
        default=[0.1, 0.25, 0.5, 0.75],
        help="Compression ratios to test"
    )
    
    parser.add_argument(
        "--model-sizes",
        nargs="+",
        type=str,
        default=["small", "medium", "large"],
        help="Model sizes to test"
    )
    
    parser.add_argument(
        "--methods",
        nargs="+",
        type=str,
        default=["tensorslim", "pruning", "quantization"],
        help="Compression methods to compare"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (cpu, cuda, auto)"
    )
    
    parser.add_argument(
        "--num-trials",
        type=int,
        default=5,
        help="Number of trials per configuration"
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


def create_test_models(sizes):
    """Create test models of different sizes."""
    models = {}
    
    size_configs = {
        "small": {"hidden_dims": [256, 128, 64], "input_dim": 512},
        "medium": {"hidden_dims": [512, 256, 128, 64], "input_dim": 1024},
        "large": {"hidden_dims": [1024, 512, 256, 128, 64], "input_dim": 2048}
    }
    
    for size in sizes:
        if size not in size_configs:
            continue
            
        config = size_configs[size]
        layers = []
        
        prev_dim = config["input_dim"]
        for hidden_dim in config["hidden_dims"]:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 10))
        
        model = nn.Sequential(*layers)
        models[size] = model
        
        # Print model info
        params = sum(p.numel() for p in model.parameters())
        print(f"  {size.capitalize()} model: {params:,} parameters")
    
    return models


def compress_with_tensorslim(model, compression_ratio, device):
    """Compress model using TensorSlim."""
    start_time = time.time()
    
    compressed_model = compress_model(
        model,
        compression_ratio=compression_ratio,
        inplace=False
    )
    
    compression_time = time.time() - start_time
    
    # Calculate compression metrics
    original_params = sum(p.numel() for p in model.parameters())
    compressed_params = sum(p.numel() for p in compressed_model.parameters())
    actual_ratio = original_params / compressed_params
    
    return {
        'compressed_model': compressed_model,
        'compression_time': compression_time,
        'original_params': original_params,
        'compressed_params': compressed_params,
        'actual_compression_ratio': actual_ratio,
        'parameter_reduction': 1 - (compressed_params / original_params)
    }


def compress_with_pruning(model, compression_ratio, device):
    """Simulate pruning compression (simplified)."""
    import copy
    
    start_time = time.time()
    
    # Simple magnitude-based pruning simulation
    compressed_model = copy.deepcopy(model)
    
    total_params = 0
    pruned_params = 0
    
    for module in compressed_model.modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.data
            threshold = torch.quantile(torch.abs(weight), compression_ratio)
            mask = torch.abs(weight) > threshold
            
            # Apply mask (set small weights to zero)
            module.weight.data = weight * mask
            
            total_params += weight.numel()
            pruned_params += (mask == 0).sum().item()
    
    compression_time = time.time() - start_time
    
    original_params = sum(p.numel() for p in model.parameters())
    # For pruning, compressed_params is same as original (zeros don't reduce storage)
    compressed_params = original_params
    sparsity = pruned_params / total_params
    
    return {
        'compressed_model': compressed_model,
        'compression_time': compression_time,
        'original_params': original_params,
        'compressed_params': compressed_params,
        'actual_compression_ratio': 1.0,  # No actual size reduction
        'parameter_reduction': sparsity,
        'sparsity': sparsity
    }


def compress_with_quantization(model, compression_ratio, device):
    """Simulate quantization compression."""
    import copy
    
    start_time = time.time()
    
    # Simple quantization simulation
    compressed_model = copy.deepcopy(model)
    
    # Map compression ratio to bit width
    if compression_ratio <= 0.25:
        bit_width = 8
    elif compression_ratio <= 0.5:
        bit_width = 4
    else:
        bit_width = 2
    
    # Simulate quantization by reducing precision
    for module in compressed_model.modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.data
            
            # Simple uniform quantization
            min_val, max_val = weight.min(), weight.max()
            scale = (max_val - min_val) / (2 ** bit_width - 1)
            quantized = torch.round((weight - min_val) / scale) * scale + min_val
            
            module.weight.data = quantized
    
    compression_time = time.time() - start_time
    
    original_params = sum(p.numel() for p in model.parameters())
    # Quantization reduces effective storage
    compressed_params = original_params * (bit_width / 32)  # Assuming 32-bit original
    actual_ratio = original_params / compressed_params
    
    return {
        'compressed_model': compressed_model,
        'compression_time': compression_time,
        'original_params': original_params,
        'compressed_params': compressed_params,
        'actual_compression_ratio': actual_ratio,
        'parameter_reduction': 1 - (compressed_params / original_params),
        'bit_width': bit_width
    }


def benchmark_inference(model, input_size, device, num_runs=50):
    """Benchmark model inference speed."""
    model.eval()
    model = model.to(device)
    
    # Create test input
    test_input = torch.randn(16, input_size, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(test_input)
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            _ = model(test_input)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
                
            end_time = time.time()
            times.append(end_time - start_time)
    
    return {
        'avg_time_ms': sum(times) / len(times) * 1000,
        'std_time_ms': torch.tensor(times).std().item() * 1000,
        'throughput_samples_per_sec': 16 / (sum(times) / len(times))
    }


def evaluate_quality(original_model, compressed_model, input_size, device):
    """Evaluate compression quality."""
    original_model.eval()
    compressed_model.eval()
    
    original_model = original_model.to(device)
    compressed_model = compressed_model.to(device)
    
    # Create test inputs
    test_inputs = [torch.randn(8, input_size, device=device) for _ in range(10)]
    
    quality_metrics = []
    
    with torch.no_grad():
        for test_input in test_inputs:
            original_output = original_model(test_input)
            compressed_output = compressed_model(test_input)
            
            # Calculate similarity metrics
            cosine_sim = torch.nn.functional.cosine_similarity(
                original_output.flatten(),
                compressed_output.flatten(),
                dim=0
            ).item()
            
            mse = torch.nn.functional.mse_loss(original_output, compressed_output).item()
            
            relative_error = (torch.norm(original_output - compressed_output) / 
                            torch.norm(original_output)).item()
            
            quality_metrics.append({
                'cosine_similarity': cosine_sim,
                'mse': mse,
                'relative_error': relative_error
            })
    
    # Average metrics
    avg_metrics = {}
    for key in quality_metrics[0].keys():
        avg_metrics[key] = sum(m[key] for m in quality_metrics) / len(quality_metrics)
    
    return avg_metrics


def run_single_benchmark(model, model_size, method, compression_ratio, device):
    """Run a single benchmark configuration."""
    input_sizes = {"small": 512, "medium": 1024, "large": 2048}
    input_size = input_sizes[model_size]
    
    # Compress model
    if method == "tensorslim":
        compression_result = compress_with_tensorslim(model, compression_ratio, device)
    elif method == "pruning":
        compression_result = compress_with_pruning(model, compression_ratio, device)
    elif method == "quantization":
        compression_result = compress_with_quantization(model, compression_ratio, device)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    compressed_model = compression_result['compressed_model']
    
    # Benchmark inference
    original_perf = benchmark_inference(model, input_size, device)
    compressed_perf = benchmark_inference(compressed_model, input_size, device)
    
    # Evaluate quality
    quality_metrics = evaluate_quality(model, compressed_model, input_size, device)
    
    # Combine results
    result = {
        'model_size': model_size,
        'method': method,
        'target_compression_ratio': compression_ratio,
        'actual_compression_ratio': compression_result['actual_compression_ratio'],
        'parameter_reduction': compression_result['parameter_reduction'],
        'compression_time': compression_result['compression_time'],
        'original_inference_ms': original_perf['avg_time_ms'],
        'compressed_inference_ms': compressed_perf['avg_time_ms'],
        'inference_speedup': original_perf['avg_time_ms'] / compressed_perf['avg_time_ms'],
        'cosine_similarity': quality_metrics['cosine_similarity'],
        'mse': quality_metrics['mse'],
        'relative_error': quality_metrics['relative_error']
    }
    
    return result


def create_visualizations(results_df, output_dir):
    """Create benchmark visualization plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Compression Ratio vs Quality
    plt.figure(figsize=(12, 8))
    
    for method in results_df['method'].unique():
        method_data = results_df[results_df['method'] == method]
        plt.scatter(method_data['actual_compression_ratio'], 
                   method_data['cosine_similarity'],
                   label=method.capitalize(), s=60, alpha=0.7)
    
    plt.xlabel('Compression Ratio')
    plt.ylabel('Cosine Similarity')
    plt.title('Compression Ratio vs Quality')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'compression_vs_quality.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Method Comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Compression ratio comparison
    sns.boxplot(data=results_df, x='method', y='actual_compression_ratio', ax=axes[0,0])
    axes[0,0].set_title('Compression Ratio by Method')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Quality comparison
    sns.boxplot(data=results_df, x='method', y='cosine_similarity', ax=axes[0,1])
    axes[0,1].set_title('Quality by Method')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Speed comparison
    sns.boxplot(data=results_df, x='method', y='inference_speedup', ax=axes[1,0])
    axes[1,0].set_title('Inference Speedup by Method')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Compression time comparison
    sns.boxplot(data=results_df, x='method', y='compression_time', ax=axes[1,1])
    axes[1,1].set_title('Compression Time by Method')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'method_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Model Size Impact
    plt.figure(figsize=(12, 8))
    
    for size in results_df['model_size'].unique():
        size_data = results_df[results_df['model_size'] == size]
        plt.scatter(size_data['actual_compression_ratio'],
                   size_data['cosine_similarity'],
                   label=f'{size.capitalize()} Model', s=60, alpha=0.7)
    
    plt.xlabel('Compression Ratio')
    plt.ylabel('Cosine Similarity')
    plt.title('Model Size Impact on Compression')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'model_size_impact.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Efficiency Frontier
    plt.figure(figsize=(12, 8))
    
    # Plot efficiency frontier (quality vs compression)
    for method in results_df['method'].unique():
        method_data = results_df[results_df['method'] == method]
        
        # Sort by compression ratio
        method_data = method_data.sort_values('actual_compression_ratio')
        
        plt.plot(method_data['actual_compression_ratio'],
                method_data['cosine_similarity'],
                marker='o', label=method.capitalize(), linewidth=2, markersize=6)
    
    plt.xlabel('Compression Ratio')
    plt.ylabel('Cosine Similarity (Quality)')
    plt.title('Efficiency Frontier: Quality vs Compression')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'efficiency_frontier.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Visualizations saved to: {output_dir}")


def main():
    """Main benchmark function."""
    args = parse_args()
    
    print("🚀 TensorSlim Compression Benchmark")
    print("=" * 50)
    
    # Setup
    device = setup_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nConfiguration:")
    print(f"  Model sizes: {args.model_sizes}")
    print(f"  Methods: {args.methods}")
    print(f"  Compression ratios: {args.compression_ratios}")
    print(f"  Trials per config: {args.num_trials}")
    print(f"  Device: {device}")
    
    # Create test models
    print(f"\n🏗️  Creating test models...")
    models = create_test_models(args.model_sizes)
    
    # Run benchmarks
    print(f"\n⚡ Running benchmarks...")
    results = []
    
    total_configs = len(args.model_sizes) * len(args.methods) * len(args.compression_ratios) * args.num_trials
    current_config = 0
    
    for model_size in args.model_sizes:
        if model_size not in models:
            continue
            
        model = models[model_size]
        
        for method in args.methods:
            for compression_ratio in args.compression_ratios:
                for trial in range(args.num_trials):
                    current_config += 1
                    
                    print(f"  [{current_config}/{total_configs}] "
                          f"{model_size} | {method} | ratio={compression_ratio} | trial={trial+1}")
                    
                    try:
                        result = run_single_benchmark(
                            model, model_size, method, compression_ratio, device
                        )
                        result['trial'] = trial
                        results.append(result)
                        
                    except Exception as e:
                        print(f"    ❌ Failed: {e}")
                        continue
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        print("❌ No successful benchmark results")
        return
    
    print(f"\n✅ Completed {len(results_df)} benchmark runs")
    
    # Calculate summary statistics
    print(f"\n📊 Summary Statistics:")
    summary = results_df.groupby(['model_size', 'method']).agg({
        'actual_compression_ratio': ['mean', 'std'],
        'cosine_similarity': ['mean', 'std'],
        'inference_speedup': ['mean', 'std'],
        'compression_time': ['mean', 'std']
    }).round(3)
    
    print(summary)
    
    # Find best configurations
    print(f"\n🏆 Best Configurations:")
    
    # Best compression ratio
    best_compression = results_df.loc[results_df['actual_compression_ratio'].idxmax()]
    print(f"  Highest compression: {best_compression['method']} "
          f"({best_compression['actual_compression_ratio']:.1f}x) "
          f"with {best_compression['cosine_similarity']:.3f} quality")
    
    # Best quality
    best_quality = results_df.loc[results_df['cosine_similarity'].idxmax()]
    print(f"  Highest quality: {best_quality['method']} "
          f"({best_quality['cosine_similarity']:.3f}) "
          f"with {best_quality['actual_compression_ratio']:.1f}x compression")
    
    # Best efficiency (quality/compression ratio)
    results_df['efficiency'] = results_df['cosine_similarity'] / results_df['actual_compression_ratio']
    best_efficiency = results_df.loc[results_df['efficiency'].idxmax()]
    print(f"  Best efficiency: {best_efficiency['method']} "
          f"(efficiency: {best_efficiency['efficiency']:.3f})")
    
    # Best speedup
    best_speedup = results_df.loc[results_df['inference_speedup'].idxmax()]
    print(f"  Fastest inference: {best_speedup['method']} "
          f"({best_speedup['inference_speedup']:.1f}x speedup)")
    
    # Save results
    print(f"\n💾 Saving results...")
    
    # Save raw data
    results_df.to_csv(output_dir / 'benchmark_results.csv', index=False)
    
    # Save summary
    summary.to_csv(output_dir / 'benchmark_summary.csv')
    
    # Create visualizations
    try:
        create_visualizations(results_df, output_dir)
    except Exception as e:
        print(f"⚠️  Failed to create visualizations: {e}")
    
    # Create report
    report = f"""
TensorSlim Compression Benchmark Report
======================================

Configuration:
- Model sizes: {', '.join(args.model_sizes)}
- Methods: {', '.join(args.methods)}
- Compression ratios: {', '.join(map(str, args.compression_ratios))}
- Trials per config: {args.num_trials}
- Total benchmark runs: {len(results_df)}

Key Findings:
- Highest compression: {best_compression['method']} ({best_compression['actual_compression_ratio']:.1f}x)
- Highest quality: {best_quality['method']} ({best_quality['cosine_similarity']:.3f})
- Best efficiency: {best_efficiency['method']} ({best_efficiency['efficiency']:.3f})
- Fastest inference: {best_speedup['method']} ({best_speedup['inference_speedup']:.1f}x)

Method Performance Summary:
{summary.to_string()}

Recommendations:
- For maximum compression: Use {best_compression['method']}
- For maximum quality: Use {best_quality['method']}
- For balanced efficiency: Use {best_efficiency['method']}
- For fastest inference: Use {best_speedup['method']}

Files Generated:
- benchmark_results.csv: Raw benchmark data
- benchmark_summary.csv: Summary statistics
- *.png: Visualization plots
"""
    
    with open(output_dir / 'benchmark_report.txt', 'w') as f:
        f.write(report)
    
    print(f"✅ Results saved to: {output_dir}")
    print(f"\n🎉 Benchmark Complete!")
    print(f"View detailed results in: {output_dir / 'benchmark_report.txt'}")


if __name__ == "__main__":
    main()