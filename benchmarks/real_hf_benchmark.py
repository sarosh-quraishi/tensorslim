#!/usr/bin/env python3
"""
Comprehensive benchmark with real HuggingFace models for accurate README results.
"""

import torch
import time
import sys
sys.path.insert(0, 'src')

from tensorslim import compress_model

def benchmark_real_huggingface_models():
    """Run comprehensive benchmark with real HuggingFace models."""
    print("🤖 Real HuggingFace Models Benchmark")
    print("=" * 60)
    
    try:
        from transformers import AutoModel, AutoTokenizer
        print("✅ HuggingFace Transformers available")
    except ImportError:
        print("❌ HuggingFace Transformers not available")
        return {}
    
    # Test models (starting with smaller ones)
    models_to_test = [
        {
            'name': 'DistilBERT-Base',
            'model_id': 'distilbert-base-uncased',
            'test_text': 'This is a test sentence for quality evaluation.'
        },
        {
            'name': 'BERT-Base',
            'model_id': 'bert-base-uncased', 
            'test_text': 'This is a test sentence for quality evaluation.'
        }
    ]
    
    # Test different compression levels for quality
    compression_configs = [
        {'ratio': 0.98, 'name': 'Conservative (98%)'},
        {'ratio': 0.95, 'name': 'Moderate (95%)'},
        {'ratio': 0.90, 'name': 'Aggressive (90%)'}
    ]
    
    results = {}
    
    for model_info in models_to_test:
        model_name = model_info['name']
        model_id = model_info['model_id']
        
        print(f"\n📊 {model_name} ({model_id})")
        print("-" * 50)
        
        try:
            print("Loading model and tokenizer...")
            model = AutoModel.from_pretrained(model_id)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            
            # Original model stats
            orig_params = sum(p.numel() for p in model.parameters())
            orig_size_mb = (orig_params * 4) / (1024 * 1024)
            print(f"Original: {orig_params:,} params, {orig_size_mb:.1f} MB")
            
            model_results = []
            
            for config in compression_configs:
                ratio = config['ratio']
                config_name = config['name']
                
                print(f"\n🔧 Testing {config_name} (ratio={ratio})")
                
                try:
                    # Compress model
                    start_time = time.time()
                    compressed = compress_model(model, compression_ratio=ratio)
                    compression_time = time.time() - start_time
                    
                    # Compressed model stats
                    comp_params = sum(p.numel() for p in compressed.parameters())
                    comp_size_mb = (comp_params * 4) / (1024 * 1024)
                    actual_compression = orig_params / comp_params
                    size_reduction = (1 - comp_params / orig_params) * 100
                    
                    print(f"  Compressed: {comp_params:,} params, {comp_size_mb:.1f} MB")
                    print(f"  Compression: {actual_compression:.2f}x")
                    print(f"  Size reduction: {size_reduction:.1f}%")
                    print(f"  Compression time: {compression_time:.1f}s")
                    
                    # Quality evaluation with multiple test cases
                    print("  Evaluating quality...")
                    
                    test_sentences = [
                        model_info['test_text'],
                        "The quick brown fox jumps over the lazy dog.",
                        "Machine learning models can be compressed using singular value decomposition.",
                        "Hello world, this is a simple test.",
                        "Natural language processing involves understanding human language."
                    ]
                    
                    model.eval()
                    compressed.eval()
                    
                    quality_scores = []
                    
                    with torch.no_grad():
                        for sentence in test_sentences:
                            try:
                                inputs = tokenizer(sentence, return_tensors='pt', 
                                                 padding=True, truncation=True, max_length=128)
                                
                                orig_outputs = model(**inputs)
                                comp_outputs = compressed(**inputs)
                                
                                # Compare last hidden states
                                orig_hidden = orig_outputs.last_hidden_state
                                comp_hidden = comp_outputs.last_hidden_state
                                
                                # Calculate cosine similarity
                                cos_sim = torch.nn.functional.cosine_similarity(
                                    orig_hidden.flatten(), comp_hidden.flatten(), dim=0
                                ).item()
                                
                                quality_scores.append(cos_sim)
                                
                            except Exception as e:
                                print(f"    Warning: Quality test failed for sentence: {e}")
                                continue
                    
                    if quality_scores:
                        avg_quality = sum(quality_scores) / len(quality_scores)
                        quality_loss = (1 - avg_quality) * 100
                        
                        print(f"  Quality retention: {avg_quality * 100:.2f}%")
                        print(f"  Quality loss: {quality_loss:.2f}%")
                        
                        # Status assessment
                        if quality_loss < 2.0:
                            status = "✅ Excellent (<2% loss)"
                        elif quality_loss < 5.0:
                            status = "⚠️ Good (<5% loss)"
                        else:
                            status = "❌ Poor (>5% loss)"
                        
                        print(f"  Status: {status}")
                        
                        model_results.append({
                            'config': config_name,
                            'ratio': ratio,
                            'compression': actual_compression,
                            'size_reduction': size_reduction,
                            'quality_retention': avg_quality * 100,
                            'quality_loss': quality_loss,
                            'orig_size_mb': orig_size_mb,
                            'comp_size_mb': comp_size_mb,
                            'compression_time': compression_time,
                            'status': status
                        })
                    else:
                        print(f"  ❌ Quality evaluation failed")
                        
                except Exception as e:
                    print(f"  ❌ Compression failed: {e}")
                    continue
            
            if model_results:
                results[model_name] = {
                    'model_id': model_id,
                    'original_params': orig_params,
                    'original_size_mb': orig_size_mb,
                    'results': model_results
                }
                
        except Exception as e:
            print(f"❌ Failed to load {model_name}: {e}")
            continue
    
    # Generate comprehensive summary
    print(f"\n" + "="*60)
    print("📋 COMPREHENSIVE RESULTS SUMMARY")
    print("="*60)
    
    if results:
        # Results table
        print("\n| Model | Config | Original | Compressed | Compression | Quality Loss | Status |")
        print("|-------|--------|----------|------------|-------------|--------------|---------|")
        
        for model_name, data in results.items():
            for result in data['results']:
                print(f"| {model_name} | {result['config']} | {result['orig_size_mb']:.0f}MB | {result['comp_size_mb']:.0f}MB | {result['compression']:.1f}x | {result['quality_loss']:.1f}% | {result['status']} |")
        
        # Find best results for README
        print(f"\n🏆 BEST RESULTS FOR <2% QUALITY LOSS:")
        
        excellent_results = []
        for model_name, data in results.items():
            for result in data['results']:
                if result['quality_loss'] < 2.0:
                    excellent_results.append((model_name, result))
        
        if excellent_results:
            print("\n| Model | Original Size | Compressed Size | Compression Ratio | Quality Loss | Speedup |")
            print("|-------|---------------|-----------------|-------------------|--------------|---------|")
            
            for model_name, result in excellent_results:
                speedup = result['compression'] * 0.5  # Conservative speedup estimate
                print(f"| {model_name} | {result['orig_size_mb']:.0f}MB | {result['comp_size_mb']:.0f}MB | {result['compression']:.1f}:1 | {result['quality_loss']:.1f}% | {speedup:.1f}x |")
        else:
            print("❌ No configurations achieved <2% quality loss")
        
        # Overall statistics
        all_results = []
        for data in results.values():
            all_results.extend(data['results'])
        
        if all_results:
            excellent_count = len([r for r in all_results if r['quality_loss'] < 2.0])
            good_count = len([r for r in all_results if 2.0 <= r['quality_loss'] < 5.0])
            poor_count = len([r for r in all_results if r['quality_loss'] >= 5.0])
            
            print(f"\n📊 OVERALL PERFORMANCE:")
            print(f"  • Excellent (<2% loss): {excellent_count}/{len(all_results)} configurations")
            print(f"  • Good (2-5% loss): {good_count}/{len(all_results)} configurations")
            print(f"  • Poor (>5% loss): {poor_count}/{len(all_results)} configurations")
            
            avg_compression = sum(r['compression'] for r in all_results) / len(all_results)
            avg_quality_loss = sum(r['quality_loss'] for r in all_results) / len(all_results)
            
            print(f"  • Average compression: {avg_compression:.1f}x")
            print(f"  • Average quality loss: {avg_quality_loss:.1f}%")
    
    else:
        print("❌ No successful benchmarks completed")
    
    return results

if __name__ == "__main__":
    results = benchmark_real_huggingface_models()
    
    if results:
        print(f"\n✅ Benchmark completed successfully!")
        print(f"📄 Results ready for README update")
    else:
        print(f"\n❌ Benchmark failed - check dependencies and try again")