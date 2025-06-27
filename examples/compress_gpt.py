"""
Example: Compress GPT model using TensorSlim.

This example demonstrates how to compress a GPT model from HuggingFace
with text generation capabilities preserved.
"""

import torch
import time
import argparse
from pathlib import Path

# Import TensorSlim
import sys
sys.path.insert(0, '../src')

import tensorslim
from tensorslim.integrations import HuggingFaceCompressor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compress GPT model with TensorSlim")
    
    parser.add_argument(
        "--model-name",
        type=str,
        default="gpt2",
        help="HuggingFace GPT model name to compress"
    )
    
    parser.add_argument(
        "--compression-ratio",
        type=float,
        default=0.4,
        help="Target compression ratio (0-1)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./compressed_gpt",
        help="Directory to save compressed model"
    )
    
    parser.add_argument(
        "--test-prompts",
        nargs="+",
        default=[
            "The future of artificial intelligence is",
            "Once upon a time in a distant galaxy",
            "The key to solving climate change",
            "In the world of machine learning"
        ],
        help="Test prompts for generation comparison"
    )
    
    parser.add_argument(
        "--max-length",
        type=int,
        default=100,
        help="Maximum generation length"
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


def generate_text(model, tokenizer, prompt, max_length=100, device='cpu'):
    """Generate text using the model."""
    model.eval()
    model = model.to(device)
    
    # Encode prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


def benchmark_generation(model, tokenizer, prompts, max_length, device, runs=5):
    """Benchmark text generation performance."""
    model.eval()
    model = model.to(device)
    
    total_time = 0
    total_tokens = 0
    
    print(f"Benchmarking generation with {runs} runs per prompt...")
    
    for prompt in prompts:
        for _ in range(runs):
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            start_time = time.time()
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    do_sample=False,  # Deterministic for benchmarking
                    pad_token_id=tokenizer.eos_token_id
                )
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            
            # Count tokens generated
            tokens_generated = outputs.shape[1] - inputs['input_ids'].shape[1]
            
            total_time += (end_time - start_time)
            total_tokens += tokens_generated
    
    avg_time_per_prompt = total_time / (len(prompts) * runs)
    tokens_per_second = total_tokens / total_time
    
    return {
        'avg_time_per_prompt': avg_time_per_prompt,
        'tokens_per_second': tokens_per_second,
        'total_time': total_time,
        'total_tokens': total_tokens
    }


def compare_generations(original_model, compressed_model, tokenizer, prompts, max_length, device):
    """Compare text generations between models."""
    print(f"\n📝 Comparing Text Generation Quality")
    print("-" * 60)
    
    comparisons = []
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\nPrompt {i}: {prompt}")
        print("=" * (len(prompt) + 10))
        
        # Generate with original model
        original_text = generate_text(original_model, tokenizer, prompt, max_length, device)
        
        # Generate with compressed model  
        compressed_text = generate_text(compressed_model, tokenizer, prompt, max_length, device)
        
        print(f"\n🔹 Original:")
        print(f"  {original_text}")
        
        print(f"\n🔸 Compressed:")
        print(f"  {compressed_text}")
        
        # Simple quality metrics
        original_tokens = tokenizer.encode(original_text)
        compressed_tokens = tokenizer.encode(compressed_text)
        
        # Length similarity
        length_ratio = len(compressed_tokens) / len(original_tokens)
        
        # Token overlap (simple measure)
        common_tokens = set(original_tokens) & set(compressed_tokens)
        token_overlap = len(common_tokens) / max(len(set(original_tokens)), 1)
        
        comparison = {
            'prompt': prompt,
            'original_text': original_text,
            'compressed_text': compressed_text,
            'original_length': len(original_tokens),
            'compressed_length': len(compressed_tokens),
            'length_ratio': length_ratio,
            'token_overlap': token_overlap
        }
        
        comparisons.append(comparison)
        
        print(f"\n📊 Metrics:")
        print(f"  Length ratio: {length_ratio:.2f}")
        print(f"  Token overlap: {token_overlap:.2f}")
    
    return comparisons


def main():
    """Main function."""
    args = parse_args()
    
    print("🚀 TensorSlim GPT Compression Example")
    print("=" * 50)
    
    # Setup
    device = setup_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nConfiguration:")
    print(f"  Model: {args.model_name}")
    print(f"  Compression ratio: {args.compression_ratio}")
    print(f"  Max generation length: {args.max_length}")
    print(f"  Output directory: {output_dir}")
    print(f"  Device: {device}")
    
    # Check HuggingFace availability
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
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        original_model = AutoModelForCausalLM.from_pretrained(args.model_name)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        
        # Set pad token if not exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
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
    print(f"  Architecture: {original_model.config.architectures}")
    
    # Test original generation
    print(f"\n🧪 Testing original model generation...")
    sample_prompt = args.test_prompts[0]
    sample_generation = generate_text(original_model, tokenizer, sample_prompt, 50, device)
    print(f"  Prompt: {sample_prompt}")
    print(f"  Generated: {sample_generation}")
    
    # Benchmark original model
    print(f"\n⏱️  Benchmarking original model...")
    original_benchmark = benchmark_generation(
        original_model, tokenizer, args.test_prompts[:2], args.max_length, device, runs=3
    )
    print(f"  Avg time per prompt: {original_benchmark['avg_time_per_prompt']:.2f}s")
    print(f"  Tokens per second: {original_benchmark['tokens_per_second']:.1f}")
    
    # Compress model
    print(f"\n🗜️  Compressing model...")
    print(f"  Target compression ratio: {args.compression_ratio}")
    
    try:
        compressor = HuggingFaceCompressor(
            compression_ratio=args.compression_ratio,
            quality_threshold=0.90,  # Slightly lower for generative models
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
    
    # Test compressed generation
    print(f"\n🧪 Testing compressed model generation...")
    compressed_sample_generation = generate_text(compressed_model, tokenizer, sample_prompt, 50, device)
    print(f"  Prompt: {sample_prompt}")
    print(f"  Generated: {compressed_sample_generation}")
    
    # Benchmark compressed model
    print(f"\n⏱️  Benchmarking compressed model...")
    compressed_benchmark = benchmark_generation(
        compressed_model, tokenizer, args.test_prompts[:2], args.max_length, device, runs=3
    )
    print(f"  Avg time per prompt: {compressed_benchmark['avg_time_per_prompt']:.2f}s")
    print(f"  Tokens per second: {compressed_benchmark['tokens_per_second']:.1f}")
    
    # Performance comparison
    speedup = original_benchmark['avg_time_per_prompt'] / compressed_benchmark['avg_time_per_prompt']
    throughput_improvement = compressed_benchmark['tokens_per_second'] / original_benchmark['tokens_per_second']
    
    print(f"\n🏃 Performance Comparison:")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Token generation speedup: {throughput_improvement:.2f}x")
    
    # Quality comparison
    print(f"\n🎯 Generation Quality Comparison...")
    generation_comparisons = compare_generations(
        original_model, compressed_model, tokenizer, 
        args.test_prompts, args.max_length, device
    )
    
    # Calculate average quality metrics
    avg_length_ratio = sum(c['length_ratio'] for c in generation_comparisons) / len(generation_comparisons)
    avg_token_overlap = sum(c['token_overlap'] for c in generation_comparisons) / len(generation_comparisons)
    
    print(f"\n📊 Average Quality Metrics:")
    print(f"  Length preservation: {avg_length_ratio:.2f}")
    print(f"  Token overlap: {avg_token_overlap:.2f}")
    
    # Quality grade
    if avg_token_overlap > 0.7 and 0.8 < avg_length_ratio < 1.2:
        quality_grade = "A (Excellent)"
    elif avg_token_overlap > 0.5 and 0.6 < avg_length_ratio < 1.4:
        quality_grade = "B (Good)"
    elif avg_token_overlap > 0.3:
        quality_grade = "C (Fair)"
    else:
        quality_grade = "D (Poor)"
    
    print(f"  Overall quality: {quality_grade}")
    
    # Save compressed model
    print(f"\n💾 Saving compressed model...")
    
    try:
        # Save model and tokenizer
        compressed_model.save_pretrained(output_dir / "model")
        tokenizer.save_pretrained(output_dir / "model")
        
        # Save results
        results = {
            'model_name': args.model_name,
            'compression_ratio': args.compression_ratio,
            'actual_compression_ratio': actual_compression_ratio,
            'parameter_reduction_pct': (1 - compressed_params/original_params)*100,
            'size_reduction_mb': size_reduction,
            'original_benchmark': original_benchmark,
            'compressed_benchmark': compressed_benchmark,
            'generation_comparisons': generation_comparisons,
            'avg_quality_metrics': {
                'length_ratio': avg_length_ratio,
                'token_overlap': avg_token_overlap
            },
            'speedup': speedup,
            'quality_grade': quality_grade
        }
        
        import json
        with open(output_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save sample generations
        with open(output_dir / "sample_generations.txt", 'w') as f:
            f.write("GPT Model Compression - Sample Generations\n")
            f.write("=" * 50 + "\n\n")
            
            for i, comparison in enumerate(generation_comparisons, 1):
                f.write(f"Prompt {i}: {comparison['prompt']}\n")
                f.write("-" * 40 + "\n")
                f.write(f"Original:\n{comparison['original_text']}\n\n")
                f.write(f"Compressed:\n{comparison['compressed_text']}\n\n")
                f.write(f"Length ratio: {comparison['length_ratio']:.2f}\n")
                f.write(f"Token overlap: {comparison['token_overlap']:.2f}\n")
                f.write("\n" + "=" * 50 + "\n\n")
        
        print(f"✅ Saved to: {output_dir}")
        
    except Exception as e:
        print(f"⚠️  Failed to save model: {e}")
    
    # Summary
    print(f"\n🎉 GPT Compression Complete!")
    print(f"=" * 50)
    print(f"📊 Results Summary:")
    print(f"  • {actual_compression_ratio:.1f}x compression ratio")
    print(f"  • {(1 - compressed_params/original_params)*100:.1f}% parameter reduction")
    print(f"  • {size_reduction:.1f} MB size reduction")
    print(f"  • {speedup:.2f}x generation speedup")
    print(f"  • {quality_grade} generation quality")
    print(f"  • Model saved to: {output_dir}")
    
    if "Excellent" in quality_grade or "Good" in quality_grade:
        print(f"\n✅ Good compression quality! Generation capabilities preserved.")
    else:
        print(f"\n⚠️  Quality below optimal. Consider reducing compression ratio.")
    
    print(f"\n💡 Next steps:")
    print(f"  • Test with your specific prompts and use cases")
    print(f"  • Evaluate on downstream tasks if applicable")
    print(f"  • Fine-tune if needed for your domain")


if __name__ == "__main__":
    main()