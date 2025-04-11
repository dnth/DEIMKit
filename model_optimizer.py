#!/usr/bin/env python3
"""
Independent model optimization framework for PyTorch models.
This framework allows loading and optimizing PyTorch models through various techniques.
"""

import os
from PIL import Image
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any, Callable

import torch
import torch.nn as nn
import torch_pruning as tp
from src.deimkit.predictor import load_model
from custom_pruning import CustomPruner

# Optional imports with graceful fallbacks
try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("ONNX not available. Install with: pip install onnx onnxruntime")

try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    print("TensorRT not available. Optional for GPU acceleration.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ModelOptimizer")

class ModelOptimizer:
    """Core model optimization class that works with any PyTorch model."""
    
    def __init__(
        self, 
        model_path: Optional[str] = None,
        model: Optional[nn.Module] = None,
        model_name: Optional[str] = None,
        input_shape: Tuple[int, ...] = (1, 3, 640, 640),
        device: str = "auto",
        model_type: str = "custom",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the model optimizer.
        
        Args:
            model_path: Path to PyTorch model (.pth/.pt file)
            model: PyTorch model instance (alternative to model_path)
            input_shape: Input tensor shape for the model
            device: Device to run on ('cpu', 'cuda', 'auto')
            model_type: Type of model ('deim', 'yolo', 'custom')
            config: Additional configuration for the model
        """
        self.device = self._resolve_device(device)
        self.model_path = model_path
        self.model_name = model_name
        self.input_shape = input_shape
        self.model_type = model_type.lower()
        self.predictor = None
        
        # Stats tracking
        self.original_size = 0
        self.optimized_size = 0
        self.original_params = 0
        self.optimized_params = 0
        
        # Output paths
        self.output_dir = Path("optimized_models")
        self.output_dir.mkdir(exist_ok=True)
        
        # Load model
        if model is not None:
            self.model = model
        elif model_path:
            self.model = self._load_model(model_path)
        else:
            self.model = None
            logger.warning("No model provided. Use load_model() to load a model.")
        
        if self.model:
            self.model.to(self.device)
            self.model.eval()
            self.original_params = self._count_parameters(self.model)
            logger.info(f"Model loaded with {self.original_params:,} parameters")
    
    def _resolve_device(self, device: str) -> str:
        """Resolve the device to use."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load model from path based on model type."""
          
        logger.info(f"Attempting to load model with: model_name={self.model_name}, device={self.device}, checkpoint={model_path}")
        
        # Initialize predictor with the model
        predictor = load_model(
            model_name=self.model_name,
            device=self.device,
            checkpoint=model_path
        )

        self.predictor = predictor
        
        if hasattr(predictor, 'model'):
            logger.info(f"Model loaded successfully: {type(predictor.model).__name__}")
            return predictor.model
        else:
            logger.error(f"Predictor does not contain a model attribute: {predictor}")
            raise ValueError("Model loading failed: predictor object has no 'model' attribute")
    
    def _count_parameters(self, model: nn.Module) -> int:
        """Count the number of parameters in the model."""
        return sum(p.numel() for p in model.parameters())
    
    def prune_model(
        self,
        backbone_sparsity: float = 0.3,
        encoder_sparsity: float = 0.2,
        decoder_sparsity: float = 0.1,
        method: str = "l1",
        fine_tune: bool = False,
        fine_tune_epochs: int = 5,
    ) -> nn.Module:
        """
        Prune the model to reduce size with minimal accuracy impact.
        
        Args:
            backbone_sparsity: Amount to prune from backbone (0.0-1.0)
            encoder_sparsity: Amount to prune from encoder (0.0-1.0)
            decoder_sparsity: Amount to prune from decoder (0.0-1.0)
            method: Pruning method ('l1', 'l2', 'random')
            fine_tune: Whether to fine-tune after pruning
            fine_tune_epochs: Number of epochs for fine-tuning
            
        Returns:
            Pruned PyTorch model
        """

        model = self.model.eval()


        pruner = CustomPruner(model, self.predictor)
        pruned_model = pruner.prune_model(sparsity=0.3)

        return pruned_model
        
    def save_model(
        self, 
        model: Optional[nn.Module] = None,
        output_path: Optional[str] = None,
        format: str = "pth",
    ) -> str:
        """
        Save the model to a file.
        
        Args:
            model: Model to save, defaults to self.model
            output_path: Path to save the model
            format: Format to save the model ('pth', 'torchscript')
            
        Returns:
            Path to the saved model
        """
        if model is None:
            if self.model is None:
                raise ValueError("No model provided and no model loaded")
            model = self.model
        
        # Default output path
        if output_path is None:
            model_name = type(model).__name__
            output_path = os.path.join(self.output_dir, f"{model_name}_optimized.{format}")
        
        # Save in requested format
        if format == "pth" or format == "pt":
            torch.save(model.state_dict(), output_path)
        elif format == "torchscript":
            # Create a TorchScript model
            example_input = torch.randn(self.input_shape).to(self.device)
            traced_model = torch.jit.trace(model, example_input)
            torch.jit.save(traced_model, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Model saved to {output_path}")
        return output_path

    def cluster_weights(
        self,
        model: nn.Module,
        n_clusters: int = 32,
        min_elements_per_centroid: int = 16,
        components: List[str] = None
    ) -> nn.Module:
        """
        Apply weight clustering to reduce model size and improve inference speed.
        
        Args:
            model: PyTorch model to cluster
            n_clusters: Number of centroids to use for clustering
            min_elements_per_centroid: Minimum cluster size
            components: List of component names to cluster ('backbone', 'encoder', 'decoder')
                        If None, cluster all components
        
        Returns:
            Model with clustered weights
        """
        logger.info(f"Clustering model weights with {n_clusters} clusters")
        
        if components is None:
            components = ['backbone', 'encoder', 'decoder']
        
        # Identify model components
        backbone_modules, encoder_modules, decoder_modules = self._identify_model_components(model)
        
        # Get modules to cluster based on components
        modules_to_cluster = []
        if 'backbone' in components and backbone_modules:
            modules_to_cluster.extend(backbone_modules)
        if 'encoder' in components and encoder_modules:
            modules_to_cluster.extend(encoder_modules)
        if 'decoder' in components and decoder_modules:
            modules_to_cluster.extend(decoder_modules)
        
        if not modules_to_cluster:
            logger.warning("No modules found for clustering")
            return model
        
        logger.info(f"Found {len(modules_to_cluster)} modules for clustering")
        
        # Count parameters before clustering
        params_before = sum(m.weight.numel() for m in modules_to_cluster 
                          if hasattr(m, 'weight') and m.weight is not None)
        
        # Apply K-means clustering to each module
        from sklearn.cluster import KMeans
        import numpy as np
        
        total_size_reduction = 0
        for i, module in enumerate(modules_to_cluster):
            if not hasattr(module, 'weight') or module.weight is None:
                continue
                
            # Get weight tensor
            weight = module.weight.detach().cpu().numpy()
            original_shape = weight.shape
            flattened = weight.reshape(-1)
            
            # Skip if too small
            if flattened.size < n_clusters * min_elements_per_centroid:
                logger.debug(f"Skipping module {i}: too small ({flattened.size} < {n_clusters * min_elements_per_centroid})")
                continue
            
            # Apply K-means clustering
            kmeans = KMeans(n_clusters=min(n_clusters, flattened.size // min_elements_per_centroid), 
                            random_state=42, n_init=1)
            cluster_ids = kmeans.fit_predict(flattened.reshape(-1, 1))
            centroids = kmeans.cluster_centers_.reshape(-1)
            
            # Replace weights with centroids
            quantized = np.array([centroids[cluster_id] for cluster_id in cluster_ids])
            clustered_weight = quantized.reshape(original_shape)
            
            # Update module weights
            with torch.no_grad():
                module.weight.copy_(torch.tensor(clustered_weight, 
                                              dtype=module.weight.dtype, 
                                              device=module.weight.device))
            
            # Calculate size reduction
            original_size = flattened.size * 32  # Assuming 32-bit floats
            clustered_size = (len(centroids) * 32) + (flattened.size * np.log2(len(centroids)) / 8)
            reduction = 1.0 - (clustered_size / original_size)
            total_size_reduction += reduction * flattened.size
            
            logger.debug(f"Module {i}: {flattened.size} weights clustered to {len(centroids)} centroids, "
                        f"{reduction:.2%} size reduction")
        
        avg_reduction = total_size_reduction / params_before if params_before > 0 else 0
        logger.info(f"Weight clustering complete. Average size reduction: {avg_reduction:.2%}")
        
        return model