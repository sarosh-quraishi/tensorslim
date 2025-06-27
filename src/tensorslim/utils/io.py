"""
Input/output utilities for TensorSlim.

This module provides utilities for saving, loading, and serializing
compressed models and compression metadata.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import os
import json
import pickle
import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelSaver:
    """
    Utility class for saving and loading compressed models.
    
    This class handles serialization of both the compressed model and
    associated compression metadata.
    """
    
    @staticmethod
    def save_compressed_model(
        model: nn.Module,
        save_path: str,
        compression_info: Optional[Dict[str, Any]] = None,
        include_state_dict: bool = True,
        include_full_model: bool = False
    ) -> None:
        """
        Save a compressed model with metadata.
        
        Args:
            model: Compressed model to save
            save_path: Path to save the model
            compression_info: Compression metadata
            include_state_dict: Whether to save state dict
            include_full_model: Whether to save full model object
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model state dict
        if include_state_dict:
            state_dict_path = save_path / "model_state_dict.pth"
            torch.save(model.state_dict(), state_dict_path)
            logger.info(f"Model state dict saved to: {state_dict_path}")
        
        # Save full model
        if include_full_model:
            model_path = save_path / "model.pth"
            torch.save(model, model_path)
            logger.info(f"Full model saved to: {model_path}")
        
        # Save compression info
        if compression_info is not None:
            info_path = save_path / "compression_info.json"
            
            # Make compression info JSON serializable
            serializable_info = ModelSaver._make_json_serializable(compression_info)
            
            with open(info_path, 'w') as f:
                json.dump(serializable_info, f, indent=2)
            
            logger.info(f"Compression info saved to: {info_path}")
        
        # Save model architecture info
        arch_info = {
            'model_class': model.__class__.__name__,
            'model_module': model.__class__.__module__,
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        }
        
        arch_path = save_path / "model_architecture.json"
        with open(arch_path, 'w') as f:
            json.dump(arch_info, f, indent=2)
        
        logger.info(f"Model architecture info saved to: {arch_path}")
    
    @staticmethod
    def load_compressed_model(
        load_path: str,
        model_class: Optional[type] = None,
        model_args: Optional[tuple] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        map_location: Optional[Union[str, torch.device]] = None
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Load a compressed model with metadata.
        
        Args:
            load_path: Path to load the model from
            model_class: Model class for reconstruction
            model_args: Arguments for model construction
            model_kwargs: Keyword arguments for model construction
            map_location: Device mapping for loading
            
        Returns:
            Tuple of (loaded_model, compression_info)
        """
        load_path = Path(load_path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {load_path}")
        
        # Load compression info
        compression_info = {}
        info_path = load_path / "compression_info.json"
        if info_path.exists():
            with open(info_path, 'r') as f:
                compression_info = json.load(f)
        
        # Try to load full model first
        model_path = load_path / "model.pth"
        if model_path.exists():
            model = torch.load(model_path, map_location=map_location)
            logger.info(f"Loaded full model from: {model_path}")
            return model, compression_info
        
        # Load from state dict
        state_dict_path = load_path / "model_state_dict.pth"
        if state_dict_path.exists():
            if model_class is None:
                # Try to get model class from architecture info
                arch_path = load_path / "model_architecture.json"
                if arch_path.exists():
                    with open(arch_path, 'r') as f:
                        arch_info = json.load(f)
                    
                    # This is a simplified approach - in practice, you'd need
                    # more sophisticated model reconstruction
                    raise ValueError(
                        "Model class must be provided when loading from state dict. "
                        "Architecture reconstruction not implemented."
                    )
                else:
                    raise ValueError("Model class must be provided when loading from state dict")
            
            # Construct model
            model_args = model_args or ()
            model_kwargs = model_kwargs or {}
            model = model_class(*model_args, **model_kwargs)
            
            # Load state dict
            state_dict = torch.load(state_dict_path, map_location=map_location)
            model.load_state_dict(state_dict)
            
            logger.info(f"Loaded model from state dict: {state_dict_path}")
            return model, compression_info
        
        raise FileNotFoundError("No valid model files found in the specified path")
    
    @staticmethod
    def _make_json_serializable(obj: Any) -> Any:
        """Convert object to JSON serializable format."""
        if isinstance(obj, dict):
            return {k: ModelSaver._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [ModelSaver._make_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return [ModelSaver._make_json_serializable(item) for item in obj]
        elif isinstance(obj, torch.Tensor):
            return {
                '__tensor__': True,
                'data': obj.detach().cpu().numpy().tolist(),
                'shape': list(obj.shape),
                'dtype': str(obj.dtype)
            }
        elif hasattr(obj, '__dict__'):
            return str(obj)  # Convert objects to string representation
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)


class ConfigurationManager:
    """
    Manager for TensorSlim configuration files.
    
    This class handles loading and saving of compression configurations
    for reproducible compression experiments.
    """
    
    @staticmethod
    def save_compression_config(
        config: Dict[str, Any],
        config_path: str
    ) -> None:
        """
        Save compression configuration to file.
        
        Args:
            config: Configuration dictionary
            config_path: Path to save configuration
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Configuration saved to: {config_path}")
    
    @staticmethod
    def load_compression_config(config_path: str) -> Dict[str, Any]:
        """
        Load compression configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        logger.info(f"Configuration loaded from: {config_path}")
        return config
    
    @staticmethod
    def create_default_config() -> Dict[str, Any]:
        """
        Create default compression configuration.
        
        Returns:
            Default configuration dictionary
        """
        return {
            "compression": {
                "method": "randomized_svd",
                "compression_ratio": 0.5,
                "quality_threshold": 0.95,
                "target_layers": ["Linear", "Conv2d"],
                "preserve_layers": ["BatchNorm", "LayerNorm"]
            },
            "svd_parameters": {
                "n_oversamples": 10,
                "n_power_iterations": 2,
                "random_state": 42
            },
            "transformer_specific": {
                "attention_rank": 64,
                "ffn_rank": 128,
                "output_rank": 256,
                "preserve_embeddings": True,
                "preserve_layernorm": True
            },
            "quality_assessment": {
                "enable_validation": True,
                "validation_samples": 100,
                "metrics": ["cosine_similarity", "relative_error", "mse"]
            }
        }


class ExperimentLogger:
    """
    Logger for compression experiments and results.
    
    This class provides structured logging of compression experiments
    for analysis and comparison.
    """
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiments = []
        self.current_experiment = None
    
    def start_experiment(
        self,
        experiment_name: str,
        model_name: str,
        config: Dict[str, Any]
    ) -> str:
        """
        Start a new compression experiment.
        
        Args:
            experiment_name: Name of the experiment
            model_name: Name of the model being compressed
            config: Compression configuration
            
        Returns:
            Experiment ID
        """
        import time
        from datetime import datetime
        
        experiment_id = f"{experiment_name}_{int(time.time())}"
        
        self.current_experiment = {
            'id': experiment_id,
            'name': experiment_name,
            'model_name': model_name,
            'config': config,
            'start_time': datetime.now().isoformat(),
            'results': {},
            'metrics': {},
            'status': 'running'
        }
        
        logger.info(f"Started experiment: {experiment_id}")
        return experiment_id
    
    def log_result(self, key: str, value: Any) -> None:
        """
        Log a result for the current experiment.
        
        Args:
            key: Result key
            value: Result value
        """
        if self.current_experiment is None:
            raise ValueError("No active experiment. Call start_experiment first.")
        
        self.current_experiment['results'][key] = value
        logger.debug(f"Logged result {key}: {value}")
    
    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Log metrics for the current experiment.
        
        Args:
            metrics: Metrics dictionary
        """
        if self.current_experiment is None:
            raise ValueError("No active experiment. Call start_experiment first.")
        
        self.current_experiment['metrics'].update(metrics)
        logger.debug(f"Logged metrics: {list(metrics.keys())}")
    
    def finish_experiment(self, status: str = 'completed') -> None:
        """
        Finish the current experiment.
        
        Args:
            status: Final status of the experiment
        """
        if self.current_experiment is None:
            raise ValueError("No active experiment to finish.")
        
        from datetime import datetime
        
        self.current_experiment['status'] = status
        self.current_experiment['end_time'] = datetime.now().isoformat()
        
        # Save experiment
        experiment_file = self.log_dir / f"{self.current_experiment['id']}.json"
        
        with open(experiment_file, 'w') as f:
            serializable_experiment = ModelSaver._make_json_serializable(self.current_experiment)
            json.dump(serializable_experiment, f, indent=2)
        
        self.experiments.append(self.current_experiment.copy())
        
        logger.info(f"Finished experiment: {self.current_experiment['id']} ({status})")
        self.current_experiment = None
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """
        List all experiments in the log directory.
        
        Returns:
            List of experiment summaries
        """
        experiments = []
        
        for experiment_file in self.log_dir.glob("*.json"):
            try:
                with open(experiment_file, 'r') as f:
                    experiment = json.load(f)
                
                # Create summary
                summary = {
                    'id': experiment.get('id'),
                    'name': experiment.get('name'),
                    'model_name': experiment.get('model_name'),
                    'start_time': experiment.get('start_time'),
                    'status': experiment.get('status'),
                    'compression_ratio': experiment.get('results', {}).get('compression_ratio'),
                    'quality_score': experiment.get('metrics', {}).get('quality_score')
                }
                
                experiments.append(summary)
                
            except Exception as e:
                logger.warning(f"Failed to load experiment {experiment_file}: {e}")
        
        return sorted(experiments, key=lambda x: x.get('start_time', ''), reverse=True)
    
    def load_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """
        Load a specific experiment by ID.
        
        Args:
            experiment_id: ID of the experiment to load
            
        Returns:
            Experiment data
        """
        experiment_file = self.log_dir / f"{experiment_id}.json"
        
        if not experiment_file.exists():
            raise FileNotFoundError(f"Experiment not found: {experiment_id}")
        
        with open(experiment_file, 'r') as f:
            return json.load(f)


def export_model_summary(
    model: nn.Module,
    save_path: str,
    format: str = 'json'
) -> None:
    """
    Export model summary to file.
    
    Args:
        model: Model to summarize
        save_path: Path to save summary
        format: Output format ('json' or 'txt')
    """
    from ..integrations.pytorch import create_model_summary, analyze_model_layers
    
    summary_data = analyze_model_layers(model)
    
    if format.lower() == 'json':
        with open(save_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
    elif format.lower() == 'txt':
        summary_text = create_model_summary(model)
        with open(save_path, 'w') as f:
            f.write(summary_text)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Model summary exported to: {save_path}")


def create_compression_report_file(
    evaluation_results: Dict[str, Any],
    save_path: str,
    model_name: str = "Model"
) -> None:
    """
    Create and save a compression quality report to file.
    
    Args:
        evaluation_results: Results from quality evaluation
        save_path: Path to save the report
        model_name: Name of the model
    """
    from .metrics import create_compression_report
    
    report = create_compression_report(evaluation_results, model_name)
    
    with open(save_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Compression report saved to: {save_path}")