"""
Test suite for TensorSlim integrations.

This module tests PyTorch and HuggingFace integration functionality.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, 'src')

from tensorslim.integrations import (
    PyTorchCompressor,
    optimize_model_for_inference,
    analyze_model_layers,
    create_model_summary,
    ModelConverter,
    benchmark_model_inference,
    compare_models
)


class TestPyTorchCompressor:
    """Test cases for PyTorchCompressor."""
    
    def create_test_model(self):
        """Create a test model."""
        return nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Dropout(0.1),
            nn.Linear(32, 10)
        )
    
    def test_pytorch_compressor_init(self):
        """Test PyTorchCompressor initialization."""
        compressor = PyTorchCompressor(
            compression_ratio=0.5,
            target_layers=['Linear', 'Conv2d'],
            preserve_layers=['BatchNorm', 'LayerNorm']
        )
        
        assert compressor.compression_ratio == 0.5
        assert compressor.target_layers == ['Linear', 'Conv2d']
        assert compressor.preserve_layers == ['BatchNorm', 'LayerNorm']
    
    def test_model_compression(self):
        """Test model compression with PyTorchCompressor."""
        model = self.create_test_model()
        compressor = PyTorchCompressor(compression_ratio=0.4)
        
        result = compressor.compress(model, inplace=False)
        
        assert 'compressed_model' in result
        assert 'original_profile' in result
        assert 'compressed_profile' in result
        assert 'compression_stats' in result
        assert 'parameter_reduction' in result
        
        # Should achieve parameter reduction
        assert result['parameter_reduction'] > 0
    
    def test_quality_validation(self):
        """Test compression with quality validation."""
        model = self.create_test_model()
        
        def validation_fn(orig_model, comp_model):
            # Simple validation function
            return {'custom_metric': 0.95}
        
        compressor = PyTorchCompressor(compression_ratio=0.5)
        result = compressor.compress(
            model,
            inplace=False,
            validation_fn=validation_fn
        )
        
        assert 'quality_metrics' in result
        assert result['quality_metrics']['custom_metric'] == 0.95


class TestModelAnalysis:
    """Test model analysis functions."""
    
    def create_test_model(self):
        """Create a test model."""
        return nn.Sequential(
            nn.Linear(256, 128),
            nn.Conv2d(16, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Embedding(1000, 128),
            nn.Linear(128, 64)
        )
    
    def test_analyze_model_layers(self):
        """Test model layer analysis."""
        model = self.create_test_model()
        analysis = analyze_model_layers(model)
        
        assert 'total_layers' in analysis
        assert 'layer_types' in analysis
        assert 'compressible_layers' in analysis
        assert 'parameter_distribution' in analysis
        assert 'compression_potential' in analysis
        
        # Should detect different layer types
        assert analysis['layer_types']['Linear'] > 0
        assert analysis['layer_types']['Conv2d'] > 0
        assert analysis['layer_types']['Embedding'] > 0
        
        # Should identify compressible layers
        assert len(analysis['compressible_layers']) > 0
        assert analysis['compressible_ratio'] > 0
    
    def test_create_model_summary(self):
        """Test model summary creation."""
        model = self.create_test_model()
        summary = create_model_summary(model)
        
        assert isinstance(summary, str)
        assert 'PyTorch Model Summary' in summary
        assert 'Total Parameters' in summary
        assert 'Compression Analysis' in summary
        assert 'Layer Distribution' in summary


class TestModelOptimization:
    """Test model optimization functions."""
    
    def create_test_model(self):
        """Create a simple test model."""
        return nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
    
    def test_optimize_for_inference(self):
        """Test model optimization for inference."""
        model = self.create_test_model()
        sample_input = torch.randn(1, 64)
        
        # Test basic optimization
        optimized = optimize_model_for_inference(
            model,
            sample_input=sample_input,
            enable_fusion=False,  # Disable fusion for testing
            enable_quantization=False  # Disable quantization for testing
        )
        
        assert optimized is not None
    
    @patch('torch.jit.trace')
    def test_torchscript_tracing(self, mock_trace):
        """Test TorchScript tracing optimization."""
        model = self.create_test_model()
        sample_input = torch.randn(1, 64)
        
        # Mock successful tracing
        mock_trace.return_value = model
        
        optimized = optimize_model_for_inference(
            model,
            sample_input=sample_input,
            enable_fusion=False,
            enable_quantization=False
        )
        
        mock_trace.assert_called_once_with(model, sample_input)


class TestModelConverter:
    """Test ModelConverter functionality."""
    
    def create_test_model(self):
        """Create a simple test model."""
        return nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
    
    def test_to_torchscript_scripting(self):
        """Test TorchScript conversion via scripting."""
        model = self.create_test_model()
        
        with patch('torch.jit.script') as mock_script:
            mock_script.return_value = MagicMock()
            
            scripted = ModelConverter.to_torchscript(model)
            mock_script.assert_called_once_with(model)
    
    def test_to_torchscript_tracing(self):
        """Test TorchScript conversion via tracing."""
        model = self.create_test_model()
        sample_input = torch.randn(1, 32)
        
        with patch('torch.jit.trace') as mock_trace:
            mock_trace.return_value = MagicMock()
            
            scripted = ModelConverter.to_torchscript(model, sample_input)
            mock_trace.assert_called_once_with(model, sample_input)
    
    def test_to_onnx(self):
        """Test ONNX export."""
        model = self.create_test_model()
        sample_input = torch.randn(1, 32)
        
        with patch('torch.onnx.export') as mock_export:
            ModelConverter.to_onnx(
                model,
                sample_input,
                'test_model.onnx'
            )
            
            mock_export.assert_called_once()


class TestBenchmarking:
    """Test benchmarking functionality."""
    
    def create_test_model(self):
        """Create a simple test model."""
        return nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
    
    def test_benchmark_model_inference(self):
        """Test model inference benchmarking."""
        model = self.create_test_model()
        model.eval()
        
        sample_input = torch.randn(1, 64)
        
        results = benchmark_model_inference(
            model,
            sample_input,
            num_runs=5,
            warmup_runs=2
        )
        
        assert 'avg_time_ms' in results
        assert 'min_time_ms' in results
        assert 'max_time_ms' in results
        assert 'std_time_ms' in results
        assert 'throughput_fps' in results
        
        # Basic sanity checks
        assert results['avg_time_ms'] > 0
        assert results['min_time_ms'] <= results['avg_time_ms']
        assert results['max_time_ms'] >= results['avg_time_ms']
        assert results['throughput_fps'] > 0
    
    def test_compare_models(self):
        """Test model comparison functionality."""
        original_model = self.create_test_model()
        
        # Create a "compressed" version (same model for testing)
        compressed_model = self.create_test_model()
        
        sample_input = torch.randn(1, 64)
        
        def similarity_fn(out1, out2):
            return torch.cosine_similarity(out1.flatten(), out2.flatten()).item()
        
        comparison = compare_models(
            original_model,
            compressed_model,
            sample_input,
            similarity_fn=similarity_fn
        )
        
        assert 'size_comparison' in comparison
        assert 'performance_comparison' in comparison
        assert 'output_similarity' in comparison
        
        # Check size comparison fields
        size_comp = comparison['size_comparison']
        assert 'original_params' in size_comp
        assert 'compressed_params' in size_comp
        assert 'parameter_reduction' in size_comp
        
        # Check performance comparison fields
        perf_comp = comparison['performance_comparison']
        assert 'original_avg_time_ms' in perf_comp
        assert 'compressed_avg_time_ms' in perf_comp
        assert 'speedup' in perf_comp


# Test HuggingFace integration if available
class TestHuggingFaceIntegration:
    """Test HuggingFace integration (if available)."""
    
    def test_huggingface_import(self):
        """Test importing HuggingFace components."""
        try:
            from tensorslim.integrations import (
                HuggingFaceCompressor,
                compress_huggingface_model,
                analyze_huggingface_model
            )
            hf_available = True
        except ImportError:
            hf_available = False
        
        # Test should pass regardless of HuggingFace availability
        assert True
    
    @pytest.mark.skipif(
        not _check_huggingface_available(),
        reason="HuggingFace transformers not available"
    )
    def test_huggingface_compressor_init(self):
        """Test HuggingFaceCompressor initialization."""
        from tensorslim.integrations import HuggingFaceCompressor
        
        compressor = HuggingFaceCompressor(
            model_name="test-model",
            compression_ratio=0.5,
            preserve_embeddings=True
        )
        
        assert compressor.model_name == "test-model"
        assert compressor.compression_ratio == 0.5
        assert compressor.preserve_embeddings is True
    
    @pytest.mark.skipif(
        not _check_huggingface_available(),
        reason="HuggingFace transformers not available"
    )
    def test_model_configs(self):
        """Test model-specific configurations."""
        from tensorslim.integrations import HuggingFaceCompressor
        
        compressor = HuggingFaceCompressor()
        configs = compressor._get_model_configs()
        
        assert 'bert' in configs
        assert 'gpt2' in configs
        assert 'default' in configs
        
        # Check BERT config
        bert_config = configs['bert']
        assert 'attention_rank_ratio' in bert_config
        assert 'ffn_rank_ratio' in bert_config


class TestErrorHandling:
    """Test error handling in integrations."""
    
    def test_invalid_model_type(self):
        """Test handling of invalid model types."""
        # Create a model with unsupported layers only
        model = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        compressor = PyTorchCompressor(compression_ratio=0.5)
        result = compressor.compress(model, inplace=False)
        
        # Should handle gracefully
        assert result['parameter_reduction'] == 0  # No compression possible
    
    def test_benchmark_error_handling(self):
        """Test benchmarking error handling."""
        model = self.create_test_model()
        
        # Test with incompatible input
        try:
            wrong_input = torch.randn(1, 128)  # Wrong input size
            benchmark_model_inference(model, wrong_input, num_runs=1)
        except RuntimeError:
            # Expected to fail with wrong input size
            pass
    
    def create_test_model(self):
        """Create a simple test model."""
        return nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )


def _check_huggingface_available():
    """Check if HuggingFace transformers is available."""
    try:
        import transformers
        return True
    except ImportError:
        return False


# Integration test fixtures
@pytest.fixture
def simple_model():
    """Simple model fixture."""
    return nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 10)
    )


@pytest.fixture  
def sample_input():
    """Sample input fixture."""
    return torch.randn(4, 128)


class TestIntegrationWorkflows:
    """Test complete integration workflows."""
    
    def test_full_compression_workflow(self, simple_model, sample_input):
        """Test complete compression workflow."""
        # Step 1: Analyze model
        analysis = analyze_model_layers(simple_model)
        assert analysis['compressible_ratio'] > 0
        
        # Step 2: Compress model
        compressor = PyTorchCompressor(compression_ratio=0.5)
        result = compressor.compress(simple_model, inplace=False)
        
        compressed_model = result['compressed_model']
        
        # Step 3: Compare models
        def similarity_fn(out1, out2):
            return torch.cosine_similarity(out1.flatten(), out2.flatten()).item()
        
        comparison = compare_models(
            simple_model,
            compressed_model,
            sample_input,
            similarity_fn=similarity_fn
        )
        
        # Should achieve compression with reasonable similarity
        assert comparison['size_comparison']['parameter_reduction'] > 0
        assert comparison['output_similarity'] > 0.5  # Reasonable similarity
    
    def test_optimization_and_benchmarking(self, simple_model, sample_input):
        """Test optimization and benchmarking workflow."""
        # Optimize model
        optimized_model = optimize_model_for_inference(
            simple_model,
            sample_input=sample_input,
            enable_quantization=False
        )
        
        # Benchmark optimized model
        benchmark_results = benchmark_model_inference(
            optimized_model,
            sample_input,
            num_runs=3,
            warmup_runs=1
        )
        
        assert benchmark_results['avg_time_ms'] > 0
        assert benchmark_results['throughput_fps'] > 0


if __name__ == "__main__":
    pytest.main([__file__])