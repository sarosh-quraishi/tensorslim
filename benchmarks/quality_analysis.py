"""
Quality analysis benchmark for TensorSlim compression methods.

This benchmark analyzes the quality impact of different compression
configurations and compares reconstruction accuracy across methods.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import TensorSlim
import sys
sys.path.insert(0, '../src')

import tensorslim
from tensorslim import compress_model
from tensorslim.core.randomized_svd import randomized_svd, AdaptiveRandomizedSVD
from tensorslim.utils import evaluate_compression_quality, CompressionMetrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Quality analysis benchmark")
    
    parser.add_argument(
        "--output",
        type=str,
        default="quality_results.json",
        help="Output file for results"
    )
    
    parser.add_argument(
        "--compression-ratios",
        nargs="+",
        type=float,
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        help="Compression ratios to test"
    )
    
    parser.add_argument(
        "--matrix-sizes",
        nargs="+",
        type=int,
        default=[200, 500, 1000],
        help="Matrix sizes for SVD quality analysis"
    )
    
    parser.add_argument(
        "--num-trials",
        type=int,
        default=10,
        help="Number of trials per configuration"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (cpu, cuda, auto)"
    )
    
    parser.add_argument(
        "--create-plots",
        action="store_true",
        help="Create visualization plots"
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


def create_test_matrix(size, condition_number=None, rank=None, device='cpu'):
    """Create test matrix with controlled properties."""
    if rank is not None and condition_number is not None:
        # Create matrix with specific rank and condition number
        U = torch.randn(size, rank, device=device)
        V = torch.randn(rank, size, device=device)
        
        # Create singular values with specific condition number
        singular_values = torch.logspace(0, -torch.log10(torch.tensor(condition_number)), rank, device=device)
        S = torch.diag(singular_values)
        
        matrix = U @ S @ V
    else:
        # Random matrix
        matrix = torch.randn(size, size, device=device)
    
    return matrix


def analyze_svd_quality(matrix, rank, num_trials=10):
    """Analyze SVD quality metrics."""
    results = {
        'standard_svd': {},
        'randomized_svd': {},
        'adaptive_svd': {}
    }
    
    # Standard SVD (ground truth)
    try:
        U_std, S_std, Vt_std = torch.svd(matrix.cpu())
        U_std, S_std, Vt_std = U_std[:, :rank], S_std[:rank], Vt_std[:rank, :]
        
        reconstructed_std = U_std @ torch.diag(S_std) @ Vt_std
        
        frobenius_error = torch.norm(matrix.cpu() - reconstructed_std, p='fro').item()
        relative_error = frobenius_error / torch.norm(matrix.cpu(), p='fro').item()
        
        results['standard_svd'] = {
            'frobenius_error': frobenius_error,
            'relative_error': relative_error,
            'singular_values': S_std.tolist()
        }
        
    except Exception as e:
        results['standard_svd'] = {'error': str(e)}
    
    # Randomized SVD
    rand_errors = []
    rand_similarities = []
    
    for _ in range(num_trials):
        try:
            U_rand, S_rand, Vt_rand = randomized_svd(matrix, rank)
            reconstructed_rand = U_rand @ torch.diag(S_rand) @ Vt_rand
            
            frobenius_error = torch.norm(matrix - reconstructed_rand, p='fro').item()
            relative_error = frobenius_error / torch.norm(matrix, p='fro').item()
            
            rand_errors.append(relative_error)
            
            # Cosine similarity with standard SVD if available
            if 'singular_values' in results['standard_svd']:
                similarity = torch.nn.functional.cosine_similarity(
                    S_std, S_rand[:len(S_std)], dim=0
                ).item()
                rand_similarities.append(similarity)
                
        except Exception as e:
            continue
    
    if rand_errors:
        results['randomized_svd'] = {
            'avg_relative_error': np.mean(rand_errors),
            'std_relative_error': np.std(rand_errors),
            'min_relative_error': np.min(rand_errors),
            'max_relative_error': np.max(rand_errors)
        }
        
        if rand_similarities:
            results['randomized_svd']['avg_similarity_to_standard'] = np.mean(rand_similarities)
    
    # Adaptive SVD
    adaptive_errors = []
    adaptive_similarities = []
    
    for _ in range(num_trials):
        try:
            svd = AdaptiveRandomizedSVD(rank=rank, target_quality=0.95)
            U_adapt, S_adapt, Vt_adapt = svd.fit_transform(matrix)
            reconstructed_adapt = U_adapt @ torch.diag(S_adapt) @ Vt_adapt
            
            frobenius_error = torch.norm(matrix - reconstructed_adapt, p='fro').item()
            relative_error = frobenius_error / torch.norm(matrix, p='fro').item()
            
            adaptive_errors.append(relative_error)
            
            # Similarity with standard SVD
            if 'singular_values' in results['standard_svd']:
                similarity = torch.nn.functional.cosine_similarity(
                    S_std, S_adapt[:len(S_std)], dim=0
                ).item()
                adaptive_similarities.append(similarity)
                
        except Exception as e:
            continue
    
    if adaptive_errors:
        results['adaptive_svd'] = {
            'avg_relative_error': np.mean(adaptive_errors),
            'std_relative_error': np.std(adaptive_errors),
            'min_relative_error': np.min(adaptive_errors),
            'max_relative_error': np.max(adaptive_errors)
        }
        
        if adaptive_similarities:
            results['adaptive_svd']['avg_similarity_to_standard'] = np.mean(adaptive_similarities)
    
    return results


def create_test_models():
    """Create test models for compression analysis."""
    models = {
        'simple_mlp': nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        ),
        
        'deep_mlp': nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        ),
        
        'cnn_model': nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    }
    
    return models


def analyze_model_compression_quality(models, compression_ratios, device, num_trials=5):
    """Analyze quality impact of model compression."""
    results = {}
    
    for model_name, model in models.items():
        print(f"  Analyzing {model_name}...")
        
        model = model.to(device)
        model_results = {
            'original_parameters': sum(p.numel() for p in model.parameters()),
            'compression_analysis': {}
        }
        
        for ratio in compression_ratios:
            print(f"    Compression ratio: {ratio}")
            
            ratio_results = {
                'trials': [],
                'avg_metrics': {}
            }
            
            for trial in range(num_trials):
                try:
                    # Compress model
                    compressed_model = compress_model(
                        model,
                        compression_ratio=ratio,
                        inplace=False
                    )
                    
                    # Calculate compression metrics
                    compressed_params = sum(p.numel() for p in compressed_model.parameters())
                    actual_ratio = model_results['original_parameters'] / compressed_params
                    
                    # Create test inputs
                    if 'cnn' in model_name:
                        test_input = torch.randn(16, 3, 32, 32, device=device)
                    else:
                        input_size = 784 if 'simple' in model_name else 1024
                        test_input = torch.randn(16, input_size, device=device)
                    
                    # Compare outputs
                    model.eval()
                    compressed_model.eval()
                    
                    with torch.no_grad():
                        original_output = model(test_input)
                        compressed_output = compressed_model(test_input)
                    
                    # Calculate quality metrics
                    cosine_sim = torch.nn.functional.cosine_similarity(
                        original_output.flatten(),
                        compressed_output.flatten(),
                        dim=0
                    ).item()
                    
                    mse = torch.nn.functional.mse_loss(original_output, compressed_output).item()
                    
                    relative_error = (torch.norm(original_output - compressed_output) / 
                                    torch.norm(original_output)).item()
                    
                    trial_result = {
                        'actual_compression_ratio': actual_ratio,
                        'cosine_similarity': cosine_sim,
                        'mse': mse,
                        'relative_error': relative_error,
                        'compressed_parameters': compressed_params
                    }
                    
                    ratio_results['trials'].append(trial_result)
                    
                except Exception as e:
                    print(f"      Trial {trial} failed: {e}")
                    continue
            
            # Calculate average metrics
            if ratio_results['trials']:
                for metric in ['actual_compression_ratio', 'cosine_similarity', 'mse', 'relative_error']:
                    values = [trial[metric] for trial in ratio_results['trials']]
                    ratio_results['avg_metrics'][metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }
            
            model_results['compression_analysis'][ratio] = ratio_results
        
        results[model_name] = model_results
    
    return results


def create_quality_plots(results, output_dir):
    """Create quality analysis plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. SVD Quality vs Rank
    if 'svd_analysis' in results:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Collect data for different matrix sizes
        sizes = []
        ranks = []
        rand_errors = []
        adapt_errors = []
        
        for result in results['svd_analysis']:
            if 'randomized_svd' in result and 'avg_relative_error' in result['randomized_svd']:
                sizes.append(result['matrix_size'])
                ranks.append(result['rank'])
                rand_errors.append(result['randomized_svd']['avg_relative_error'])
                
                if 'adaptive_svd' in result and 'avg_relative_error' in result['adaptive_svd']:
                    adapt_errors.append(result['adaptive_svd']['avg_relative_error'])
                else:
                    adapt_errors.append(np.nan)
        
        if sizes:
            # Error vs Rank
            axes[0, 0].scatter(ranks, rand_errors, label='Randomized SVD', alpha=0.7)
            if not all(np.isnan(adapt_errors)):
                axes[0, 0].scatter(ranks, adapt_errors, label='Adaptive SVD', alpha=0.7)
            axes[0, 0].set_xlabel('Rank')
            axes[0, 0].set_ylabel('Relative Error')
            axes[0, 0].set_title('SVD Error vs Rank')
            axes[0, 0].legend()
            axes[0, 0].set_yscale('log')
            
            # Error vs Matrix Size
            axes[0, 1].scatter(sizes, rand_errors, label='Randomized SVD', alpha=0.7)
            if not all(np.isnan(adapt_errors)):
                axes[0, 1].scatter(sizes, adapt_errors, label='Adaptive SVD', alpha=0.7)
            axes[0, 1].set_xlabel('Matrix Size')
            axes[0, 1].set_ylabel('Relative Error')
            axes[0, 1].set_title('SVD Error vs Matrix Size')
            axes[0, 1].legend()
            axes[0, 1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'svd_quality_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Model Compression Quality
    if 'model_compression' in results:
        for model_name, model_data in results['model_compression'].items():
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            ratios = []
            similarities = []
            errors = []
            actual_ratios = []
            
            for ratio, ratio_data in model_data['compression_analysis'].items():
                if ratio_data['avg_metrics']:
                    ratios.append(ratio)
                    similarities.append(ratio_data['avg_metrics']['cosine_similarity']['mean'])
                    errors.append(ratio_data['avg_metrics']['relative_error']['mean'])
                    actual_ratios.append(ratio_data['avg_metrics']['actual_compression_ratio']['mean'])
            
            if ratios:
                # Quality vs Compression Ratio
                axes[0, 0].plot(ratios, similarities, 'o-', linewidth=2, markersize=6)
                axes[0, 0].set_xlabel('Target Compression Ratio')
                axes[0, 0].set_ylabel('Cosine Similarity')
                axes[0, 0].set_title(f'{model_name.replace("_", " ").title()} - Quality vs Compression')
                axes[0, 0].grid(True, alpha=0.3)
                
                # Error vs Compression Ratio
                axes[0, 1].plot(ratios, errors, 'o-', color='red', linewidth=2, markersize=6)
                axes[0, 1].set_xlabel('Target Compression Ratio')
                axes[0, 1].set_ylabel('Relative Error')
                axes[0, 1].set_title(f'{model_name.replace("_", " ").title()} - Error vs Compression')
                axes[0, 1].grid(True, alpha=0.3)
                axes[0, 1].set_yscale('log')
                
                # Actual vs Target Compression
                axes[1, 0].plot(ratios, actual_ratios, 'o-', color='green', linewidth=2, markersize=6)
                axes[1, 0].plot([min(ratios), max(ratios)], [1/max(ratios), 1/min(ratios)], '--', alpha=0.5, label='Ideal')
                axes[1, 0].set_xlabel('Target Compression Ratio')
                axes[1, 0].set_ylabel('Actual Compression Ratio')
                axes[1, 0].set_title(f'{model_name.replace("_", " ").title()} - Actual vs Target Compression')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
                
                # Quality vs Actual Compression
                axes[1, 1].scatter(actual_ratios, similarities, s=60, alpha=0.7)
                axes[1, 1].set_xlabel('Actual Compression Ratio')
                axes[1, 1].set_ylabel('Cosine Similarity')
                axes[1, 1].set_title(f'{model_name.replace("_", " ").title()} - Quality vs Actual Compression')
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / f'{model_name}_quality_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"📊 Quality plots saved to: {output_dir}")


def generate_quality_report(results, output_file):
    """Generate quality analysis report."""
    report_file = Path(output_file).with_suffix('.txt')
    
    with open(report_file, 'w') as f:
        f.write("TensorSlim Quality Analysis Report\n")
        f.write("=" * 40 + "\n\n")
        
        # SVD Analysis
        if 'svd_analysis' in results:
            f.write("SVD Quality Analysis\n")
            f.write("-" * 25 + "\n\n")
            
            for result in results['svd_analysis']:
                size = result['matrix_size']
                rank = result['rank']
                
                f.write(f"Matrix: {size}x{size}, Rank: {rank}\n")
                
                if 'standard_svd' in result and 'relative_error' in result['standard_svd']:
                    f.write(f"  Standard SVD Error: {result['standard_svd']['relative_error']:.6f}\n")
                
                if 'randomized_svd' in result and 'avg_relative_error' in result['randomized_svd']:
                    rand = result['randomized_svd']
                    f.write(f"  Randomized SVD Error: {rand['avg_relative_error']:.6f} ± {rand['std_relative_error']:.6f}\n")
                
                if 'adaptive_svd' in result and 'avg_relative_error' in result['adaptive_svd']:
                    adapt = result['adaptive_svd']
                    f.write(f"  Adaptive SVD Error: {adapt['avg_relative_error']:.6f} ± {adapt['std_relative_error']:.6f}\n")
                
                f.write("\n")
        
        # Model Compression Analysis
        if 'model_compression' in results:
            f.write("\nModel Compression Quality Analysis\n")
            f.write("-" * 35 + "\n\n")
            
            for model_name, model_data in results['model_compression'].items():
                f.write(f"Model: {model_name.replace('_', ' ').title()}\n")
                f.write(f"Original Parameters: {model_data['original_parameters']:,}\n\n")
                
                for ratio, ratio_data in model_data['compression_analysis'].items():
                    if not ratio_data['avg_metrics']:
                        continue
                        
                    f.write(f"  Compression Ratio: {ratio}\n")
                    
                    metrics = ratio_data['avg_metrics']
                    f.write(f"    Actual Ratio: {metrics['actual_compression_ratio']['mean']:.2f}x\n")
                    f.write(f"    Cosine Similarity: {metrics['cosine_similarity']['mean']:.4f} ± {metrics['cosine_similarity']['std']:.4f}\n")
                    f.write(f"    Relative Error: {metrics['relative_error']['mean']:.6f} ± {metrics['relative_error']['std']:.6f}\n")
                    f.write(f"    MSE: {metrics['mse']['mean']:.6f} ± {metrics['mse']['std']:.6f}\n")
                    
                    # Quality assessment
                    similarity = metrics['cosine_similarity']['mean']
                    if similarity > 0.99:
                        quality = "Excellent"
                    elif similarity > 0.95:
                        quality = "Very Good"
                    elif similarity > 0.90:
                        quality = "Good"
                    elif similarity > 0.80:
                        quality = "Fair"
                    else:
                        quality = "Poor"
                    
                    f.write(f"    Quality Assessment: {quality}\n\n")
                
                f.write("\n")
        
        f.write("Analysis completed successfully!\n")
    
    print(f"📄 Quality report saved to: {report_file}")


def main():
    """Main quality analysis function."""
    args = parse_args()
    
    print("🔍 TensorSlim Quality Analysis Benchmark")
    print("=" * 50)
    
    device = setup_device(args.device)
    
    print(f"\nConfiguration:")
    print(f"  Matrix sizes: {args.matrix_sizes}")
    print(f"  Compression ratios: {args.compression_ratios}")
    print(f"  Trials per config: {args.num_trials}")
    print(f"  Device: {device}")
    print(f"  Create plots: {args.create_plots}")
    
    results = {
        'config': {
            'matrix_sizes': args.matrix_sizes,
            'compression_ratios': args.compression_ratios,
            'num_trials': args.num_trials,
            'device': str(device)
        },
        'svd_analysis': [],
        'model_compression': {}
    }
    
    # SVD Quality Analysis
    print(f"\n🔢 Running SVD quality analysis...")
    
    for size in args.matrix_sizes:
        for ratio in [0.1, 0.3, 0.5]:  # Test a few ratios for SVD
            rank = max(1, int(size * ratio))
            print(f"  Matrix size: {size}x{size}, Rank: {rank}")
            
            try:
                matrix = create_test_matrix(size, device=device)
                result = analyze_svd_quality(matrix, rank, args.num_trials)
                result['matrix_size'] = size
                result['rank'] = rank
                result['compression_ratio'] = ratio
                
                results['svd_analysis'].append(result)
                
            except Exception as e:
                print(f"    ❌ Failed: {e}")
                continue
    
    # Model Compression Quality Analysis
    print(f"\n🏗️  Running model compression quality analysis...")
    
    models = create_test_models()
    
    try:
        compression_results = analyze_model_compression_quality(
            models, args.compression_ratios, device, args.num_trials
        )
        results['model_compression'] = compression_results
    except Exception as e:
        print(f"❌ Model compression analysis failed: {e}")
    
    # Save results
    print(f"\n💾 Saving results...")
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Results saved to: {output_path}")
    
    # Generate report
    generate_quality_report(results, args.output)
    
    # Create plots
    if args.create_plots:
        try:
            plot_dir = output_path.parent / "quality_plots"
            create_quality_plots(results, plot_dir)
        except Exception as e:
            print(f"⚠️  Failed to create plots: {e}")
    
    # Summary
    print(f"\n📊 Quality Analysis Summary:")
    
    if results['svd_analysis']:
        # Average SVD errors
        rand_errors = [r['randomized_svd'].get('avg_relative_error', 0) 
                      for r in results['svd_analysis'] 
                      if 'randomized_svd' in r and 'avg_relative_error' in r['randomized_svd']]
        
        if rand_errors:
            avg_error = np.mean(rand_errors)
            print(f"  Average randomized SVD error: {avg_error:.6f}")
    
    if results['model_compression']:
        print(f"  Models analyzed: {len(results['model_compression'])}")
        
        # Find best quality configurations
        best_quality = 0
        best_config = None
        
        for model_name, model_data in results['model_compression'].items():
            for ratio, ratio_data in model_data['compression_analysis'].items():
                if ratio_data['avg_metrics']:
                    similarity = ratio_data['avg_metrics']['cosine_similarity']['mean']
                    if similarity > best_quality:
                        best_quality = similarity
                        best_config = (model_name, ratio)
        
        if best_config:
            print(f"  Best quality: {best_quality:.4f} ({best_config[0]}, ratio={best_config[1]})")
    
    print(f"\n🎉 Quality analysis completed successfully!")


if __name__ == "__main__":
    main()