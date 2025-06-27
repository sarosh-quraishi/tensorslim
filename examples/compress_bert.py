"""
Example: Compress BERT model using TensorSlim.

This example demonstrates how to compress a BERT model from HuggingFace
using TensorSlim with quality monitoring and performance comparison.
"""

import torch
import time
import argparse
from pathlib import Path

# Import TensorSlim
import sys
sys.path.insert(0, '../src')

import tensorslim
from tensorslim.integrations import compress_huggingface_model, HuggingFaceCompressor
from tensorslim.utils import evaluate_compression_quality, create_compression_report


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compress BERT model with TensorSlim")
    
    parser.add_argument(
        "--model-name",
        type=str,
        default="bert-base-uncased",
        help="HuggingFace model name to compress"
    )
    
    parser.add_argument(
        "--compression-ratio",
        type=float,
        default=0.5,
        help="Target compression ratio (0-1)"
    )
    
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=0.95,
        help="Minimum quality threshold to maintain"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./compressed_bert",
        help="Directory to save compressed model"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for performance testing"
    )
    
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=128,
        help="Sequence length for testing"
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


def create_sample_inputs(batch_size, sequence_length, vocab_size=30522):
    """Create sample inputs for testing."""
    input_ids = torch.randint(0, vocab_size, (batch_size, sequence_length))
    attention_mask = torch.ones(batch_size, sequence_length)
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }


def benchmark_model(model, inputs, device, num_runs=50, warmup_runs=10):
    """Benchmark model inference performance."""
    model.eval()
    model = model.to(device)
    
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Warmup
    print(f"Warming up with {warmup_runs} runs...")
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(**inputs)
    
    # Benchmark
    print(f"Benchmarking with {num_runs} runs...")
    times = []
    
    with torch.no_grad():
        for i in range(num_runs):
            start_time = time.time()
            outputs = model(**inputs)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            times.append(end_time - start_time)
            
            if (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/{num_runs} runs")
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    throughput = batch_size / avg_time
    
    return {
        'avg_time_ms': avg_time * 1000,
        'min_time_ms': min_time * 1000,
        'max_time_ms': max_time * 1000,
        'throughput_samples_per_sec': throughput,
        'memory_allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024 if device.type == 'cuda' else None
    }


def compare_outputs(original_model, compressed_model, inputs, device):
    """Compare outputs between original and compressed models."""
    original_model.eval()
    compressed_model.eval()
    
    original_model = original_model.to(device)
    compressed_model = compressed_model.to(device)
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        original_outputs = original_model(**inputs)
        compressed_outputs = compressed_model(**inputs)
    
    # Compare last hidden states
    original_hidden = original_outputs.last_hidden_state
    compressed_hidden = compressed_outputs.last_hidden_state
    
    # Calculate similarity metrics
    cosine_sim = torch.nn.functional.cosine_similarity(
        original_hidden.flatten(),
        compressed_hidden.flatten(),
        dim=0
    ).item()
    
    mse = torch.nn.functional.mse_loss(original_hidden, compressed_hidden).item()
    
    relative_error = torch.norm(original_hidden - compressed_hidden).item() / torch.norm(original_hidden).item()
    
    return {
        'cosine_similarity': cosine_sim,
        'mse': mse,
        'relative_error': relative_error
    }


def main():
    """Main function."""
    args = parse_args()
    
    print("🚀 TensorSlim BERT Compression Example")
    print("=" * 50)
    
    # Setup
    device = setup_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nConfiguration:")
    print(f"  Model: {args.model_name}")
    print(f"  Compression ratio: {args.compression_ratio}")
    print(f"  Quality threshold: {args.quality_threshold}")
    print(f"  Output directory: {output_dir}")
    print(f"  Device: {device}")
    
    # Check if HuggingFace is available
    try:
        import transformers
        print(f"  HuggingFace Transformers: {transformers.__version__}")
    except ImportError:
        print("❌ HuggingFace Transformers not available. Please install with:")
        print("   uv add tensorslim[huggingface]")
        return
    
    # Load original model
    print(f"\n📥 Loading original model: {args.model_name}")
    
    try:
        from transformers import AutoModel, AutoTokenizer
        original_model = AutoModel.from_pretrained(args.model_name)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        print(f"✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Model statistics
    original_params = sum(p.numel() for p in original_model.parameters())
    original_size_mb = sum(p.numel() * p.element_size() for p in original_model.parameters()) / (1024 * 1024)
    
    print(f"\n📊 Original Model Statistics:")
    print(f"  Parameters: {original_params:,}")
    print(f"  Model size: {original_size_mb:.1f} MB")
    
    # Create test inputs
    test_inputs = create_sample_inputs(args.batch_size, args.sequence_length)
    print(f"\n🧪 Created test inputs: batch_size={args.batch_size}, seq_len={args.sequence_length}")
    
    # Benchmark original model
    print(f"\n⏱️  Benchmarking original model...")
    original_benchmark = benchmark_model(original_model, test_inputs, device)
    
    print(f"  Average time: {original_benchmark['avg_time_ms']:.2f} ms")
    print(f"  Throughput: {original_benchmark['throughput_samples_per_sec']:.1f} samples/sec")
    if original_benchmark['memory_allocated_mb']:
        print(f"  GPU memory: {original_benchmark['memory_allocated_mb']:.1f} MB")
    
    # Compress model
    print(f"\n🗜️  Compressing model...")
    print(f"  Target compression ratio: {args.compression_ratio}")
    print(f"  Quality threshold: {args.quality_threshold}")
    
    try:
        compressor = HuggingFaceCompressor(
            compression_ratio=args.compression_ratio,
            quality_threshold=args.quality_threshold,
            preserve_embeddings=True,
            device=device
        )
        
        compressed_model, compression_info = compressor.compress_model(original_model)
        print(f"✅ Model compressed successfully")
        
    except Exception as e:
        print(f"❌ Compression failed: {e}")
        return
    
    # Compression statistics
    compressed_params = sum(p.numel() for p in compressed_model.parameters())
    compressed_size_mb = sum(p.numel() * p.element_size() for p in compressed_model.parameters()) / (1024 * 1024)
    
    actual_compression_ratio = original_params / compressed_params
    size_reduction = original_size_mb - compressed_size_mb
    
    print(f"\n📈 Compression Results:")
    print(f"  Compressed parameters: {compressed_params:,}")
    print(f"  Compressed size: {compressed_size_mb:.1f} MB")
    print(f"  Actual compression ratio: {actual_compression_ratio:.2f}x")
    print(f"  Parameter reduction: {(1 - compressed_params/original_params)*100:.1f}%")
    print(f"  Size reduction: {size_reduction:.1f} MB")
    print(f"  Layers compressed: {len(compression_info['compression_stats'])}")
    
    # Benchmark compressed model
    print(f"\n⏱️  Benchmarking compressed model...")
    compressed_benchmark = benchmark_model(compressed_model, test_inputs, device)
    
    print(f"  Average time: {compressed_benchmark['avg_time_ms']:.2f} ms")
    print(f"  Throughput: {compressed_benchmark['throughput_samples_per_sec']:.1f} samples/sec")
    
    # Performance comparison
    speedup = original_benchmark['avg_time_ms'] / compressed_benchmark['avg_time_ms']
    throughput_improvement = compressed_benchmark['throughput_samples_per_sec'] / original_benchmark['throughput_samples_per_sec']
    
    print(f"\n🏃 Performance Comparison:")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Throughput improvement: {throughput_improvement:.2f}x")
    
    # Quality comparison
    print(f"\n🎯 Quality Assessment...")
    quality_metrics = compare_outputs(original_model, compressed_model, test_inputs, device)
    
    print(f"  Cosine similarity: {quality_metrics['cosine_similarity']:.4f}")
    print(f"  Relative error: {quality_metrics['relative_error']:.4f}")
    print(f"  MSE: {quality_metrics['mse']:.6f}")
    
    # Quality grade
    if quality_metrics['cosine_similarity'] > 0.99:
        quality_grade = "A+ (Excellent)"
    elif quality_metrics['cosine_similarity'] > 0.95:
        quality_grade = "A (Very Good)"
    elif quality_metrics['cosine_similarity'] > 0.90:
        quality_grade = "B (Good)"
    elif quality_metrics['cosine_similarity'] > 0.80:
        quality_grade = "C (Fair)"
    else:
        quality_grade = "D (Poor)"
    
    print(f"  Quality grade: {quality_grade}")
    
    # Save compressed model
    print(f"\n💾 Saving compressed model...")
    
    try:
        # Save model and tokenizer
        compressed_model.save_pretrained(output_dir / "model")
        tokenizer.save_pretrained(output_dir / "model")
        
        # Save compression info
        import json
        with open(output_dir / "compression_info.json", 'w') as f:
            # Make serializable
            serializable_info = {}
            for k, v in compression_info.items():
                if isinstance(v, dict):
                    serializable_info[k] = {str(kk): str(vv) for kk, vv in v.items()}
                else:
                    serializable_info[k] = str(v)
            json.dump(serializable_info, f, indent=2)
        
        # Save benchmark results
        results = {
            'model_name': args.model_name,
            'compression_ratio': args.compression_ratio,
            'actual_compression_ratio': actual_compression_ratio,
            'parameter_reduction_pct': (1 - compressed_params/original_params)*100,
            'size_reduction_mb': size_reduction,
            'original_benchmark': original_benchmark,
            'compressed_benchmark': compressed_benchmark,
            'quality_metrics': quality_metrics,
            'speedup': speedup,
            'quality_grade': quality_grade
        }
        
        with open(output_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Saved to: {output_dir}")
        
    except Exception as e:
        print(f"⚠️  Failed to save model: {e}")
    
    # Summary
    print(f"\n🎉 Compression Complete!")
    print(f"=" * 50)
    print(f"📊 Results Summary:")
    print(f"  • {actual_compression_ratio:.1f}x compression ratio")
    print(f"  • {(1 - compressed_params/original_params)*100:.1f}% parameter reduction")
    print(f"  • {size_reduction:.1f} MB size reduction")
    print(f"  • {speedup:.2f}x inference speedup")
    print(f"  • {quality_grade} quality")
    print(f"  • Model saved to: {output_dir}")
    
    if quality_metrics['cosine_similarity'] > 0.95:
        print(f"\n✅ Excellent compression quality! Ready for deployment.")
    elif quality_metrics['cosine_similarity'] > 0.90:
        print(f"\n✅ Good compression quality. Consider additional validation.")
    else:
        print(f"\n⚠️  Quality below threshold. Consider reducing compression ratio.")


if __name__ == "__main__":
    main()