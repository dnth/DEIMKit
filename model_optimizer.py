#!/usr/bin/env python3
"""
Independent model optimization framework for PyTorch models.
This framework allows loading and optimizing PyTorch models through various techniques.
"""

import os
import json
import time
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any, Callable

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from src.deimkit.predictor import load_model


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
        self.config = config or {}
        
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
        try:            
            logger.info(f"Attempting to load model with: model_name={self.model_name}, device={self.device}, checkpoint={model_path}")
            
            # Initialize predictor with the model
            predictor = load_model(
                model_name=self.model_name,
                device=self.device,
                checkpoint=model_path
            )
            
            if hasattr(predictor, 'model'):
                logger.info(f"Model loaded successfully: {type(predictor.model).__name__}")
                return predictor.model
            else:
                logger.error(f"Predictor does not contain a model attribute: {predictor}")
                raise ValueError("Model loading failed: predictor object has no 'model' attribute")
        except ImportError as e:
            logger.warning(f"DEIMKit import error: {e}")
            logger.warning("DEIMKit not available, attempting to load as generic model")
            if os.path.exists(model_path):
                # Try to load as a standard PyTorch model
                try:
                    model = torch.load(model_path, map_location=self.device)
                    logger.info(f"Loaded model from {model_path}: {type(model)}")
                    
                    if isinstance(model, nn.Module):
                        return model
                    elif isinstance(model, dict) and "model" in model:
                        logger.info("Found model in dictionary under 'model' key")
                        return model["model"]
                    elif isinstance(model, dict) and "state_dict" in model:
                        logger.info("Found state_dict in dictionary, creating model from state_dict")
                        # Create model from state dict
                        return self._create_model_from_state_dict(model["state_dict"])
                    else:
                        logger.error(f"Unknown model format: {type(model)}")
                        raise ValueError(f"Loaded object is not a model: {type(model)}")
                except Exception as e:
                    logger.error(f"Error loading model from {model_path}: {e}")
                    raise
            else:
                logger.error(f"Model file not found: {model_path}")
                raise FileNotFoundError(f"Model file not found: {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise ValueError(f"Failed to load model from {model_path}: {str(e)}")
    
    def _create_model_from_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> nn.Module:
        """Create a model from a state dict by guessing the structure."""
        # This is a placeholder - in a real implementation, you'd need heuristics
        # to determine the model architecture or require the user to provide it
        logger.warning("Creating placeholder model from state_dict - functionality limited")
        
        # For now, just create a simple container
        class ModelContainer(nn.Module):
            def __init__(self, state_dict):
                super().__init__()
                self.state_dict_data = state_dict
                
            def forward(self, x):
                raise NotImplementedError("This is a placeholder model for state_dict storage")
        
        model = ModelContainer(state_dict)
        model.load_state_dict(state_dict)
        return model
    
    def _count_parameters(self, model: nn.Module) -> int:
        """Count the number of parameters in the model."""
        return sum(p.numel() for p in model.parameters())
    
    def prune_model(
        self,
        backbone_sparsity: float = 0.3,
        encoder_sparsity: float = 0.2,
        decoder_sparsity: float = 0.1,
        method: str = "l1_unstructured",
        fine_tune: bool = False,
        fine_tune_epochs: int = 5,
    ) -> nn.Module:
        """
        Prune the model to reduce size with minimal accuracy impact.
        
        Args:
            backbone_sparsity: Amount to prune from backbone (0.0-1.0)
            encoder_sparsity: Amount to prune from encoder (0.0-1.0)
            decoder_sparsity: Amount to prune from decoder (0.0-1.0)
            method: Pruning method ('l1_unstructured', 'random_unstructured')
            fine_tune: Whether to fine-tune after pruning
            fine_tune_epochs: Number of epochs for fine-tuning
            
        Returns:
            Pruned PyTorch model
        """
        logger.info(f"Pruning model with sparsity: backbone={backbone_sparsity}, "
                   f"encoder={encoder_sparsity}, decoder={decoder_sparsity}")
        
        if self.model is None:
            raise ValueError("No model loaded. Load a model first.")
        
        # Create a deep copy to avoid modifying the original
        model = self._clone_model(self.model)
        
        # Identify model components for targeted pruning
        backbone_modules, encoder_modules, decoder_modules = self._identify_model_components(model)
        
        # Apply pruning to each component
        if backbone_modules and backbone_sparsity > 0:
            self._prune_component(backbone_modules, backbone_sparsity, method)
            
        if encoder_modules and encoder_sparsity > 0:
            self._prune_component(encoder_modules, encoder_sparsity, method)
            
        if decoder_modules and decoder_sparsity > 0:
            self._prune_component(decoder_modules, decoder_sparsity, method)
        
        # Make pruning permanent to save memory
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                prune.remove(module, 'weight')
        
        # Count parameters after pruning
        self.optimized_params = self._count_parameters(model)
        reduction = 1.0 - (self.optimized_params / self.original_params)
        
        logger.info(f"Pruning complete. Parameters reduced from {self.original_params:,} "
                  f"to {self.optimized_params:,} ({reduction:.2%} reduction)")
        
        # Fine-tune if requested
        if fine_tune:
            logger.warning("Fine-tuning functionality requires dataset and loss function")
            logger.warning("Implement custom fine-tuning in your application")
        
        return model
    
    def _clone_model(self, model: nn.Module) -> nn.Module:
        """Create a deep copy of the model."""
        # Save model state dict
        state_dict = model.state_dict()
        
        try:
            # Try to create a new instance through load_model if available
            if self.model_name:
                try:
                    from src.deimkit.predictor import load_model
                    new_model = load_model(
                        model_name=self.model_name,
                        device=self.device,
                        checkpoint=None  # Don't load weights from checkpoint
                    ).model
                    
                    # Load the state dict into the new model
                    new_model.load_state_dict(state_dict)
                    logger.info("Created model copy via load_model()")
                    return new_model
                except Exception as e:
                    logger.warning(f"Failed to clone model via load_model: {e}")
            
            # Fallback: Create a simple copy via state dict transfer
            logger.info("Creating model copy via state dict")
            copy = type(model)()  # Create new instance of same class
            copy.load_state_dict(state_dict)
            return copy
        except Exception as e:
            logger.warning(f"Error in deep copy, using original model: {e}")
            return model
    
    def _identify_model_components(self, model: nn.Module) -> Tuple[List, List, List]:
        """
        Identify backbone, encoder, and decoder components of the model.
        This is a heuristic approach and might need customization for specific models.
        """
        backbone_modules = []
        encoder_modules = []
        decoder_modules = []
        
        # Check if the model has explicit components
        if hasattr(model, 'backbone'):
            backbone_modules = [m for m in model.backbone.modules() 
                              if isinstance(m, (nn.Conv2d, nn.Linear))]
        
        if hasattr(model, 'encoder'):
            encoder_modules = [m for m in model.encoder.modules() 
                             if isinstance(m, (nn.Conv2d, nn.Linear))]
        
        if hasattr(model, 'decoder'):
            decoder_modules = [m for m in model.decoder.modules() 
                             if isinstance(m, (nn.Conv2d, nn.Linear))]
        
        # If no explicit components found, use a heuristic approach
        if not any([backbone_modules, encoder_modules, decoder_modules]):
            all_modules = [m for m in model.modules() 
                         if isinstance(m, (nn.Conv2d, nn.Linear))]
            
            # Simple heuristic: first 60% backbone, next 20% encoder, last 20% decoder
            n = len(all_modules)
            backbone_modules = all_modules[:int(n * 0.6)]
            encoder_modules = all_modules[int(n * 0.6):int(n * 0.8)]
            decoder_modules = all_modules[int(n * 0.8):]
        
        return backbone_modules, encoder_modules, decoder_modules
    
    def _prune_component(
        self, 
        modules: List[nn.Module], 
        sparsity: float,
        method: str = "l1_unstructured"
    ) -> None:
        """Apply pruning to a list of modules."""
        if sparsity <= 0.0:
            return  # Skip if no pruning requested
        
        if method == "l1_unstructured":
            pruning_method = prune.l1_unstructured
        elif method == "random_unstructured":
            pruning_method = prune.random_unstructured
        else:
            raise ValueError(f"Unsupported pruning method: {method}")
        
        logger.info(f"Applying {method} pruning with sparsity {sparsity} to {len(modules)} modules")
        
        total_params_before = 0
        prunable_params = 0
        
        # Count parameters before pruning
        for module in modules:
            if hasattr(module, 'weight') and module.weight is not None:
                total_params_before += module.weight.numel()
                prunable_params += module.weight.numel()
        
        logger.info(f"Total parameters in modules before pruning: {total_params_before:,}")
        
        # Apply pruning to each module
        for i, module in enumerate(modules):
            if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, 'weight'):
                if module.weight is None or module.weight.numel() == 0:
                    continue
                    
                # Check if the module already has a pruning mask
                if hasattr(module.weight, 'mask'):
                    logger.info(f"Module {i} already has a pruning mask, removing")
                    prune.remove(module, 'weight')
                
                # Apply pruning
                pruning_method(module, name="weight", amount=sparsity)
                
                # Log diagnostics
                if hasattr(module.weight, 'mask'):
                    nonzero = torch.sum(module.weight.mask != 0).item()
                    total = module.weight.mask.numel()
                    actual_sparsity = 1.0 - (nonzero / total)
                    logger.debug(f"Module {i}: pruned to {actual_sparsity:.2%} sparsity ({nonzero:,}/{total:,} weights)")
        
        # Count parameters after pruning to verify reduction
        total_params_after = 0
        for module in modules:
            if hasattr(module, 'weight'):
                if hasattr(module.weight, 'mask'):
                    total_params_after += torch.sum(module.weight.mask != 0).item()
                else:
                    total_params_after += module.weight.numel()
        
        reduction = 1.0 - (total_params_after / total_params_before) if total_params_before > 0 else 0
        logger.info(f"Pruned {sparsity:.2%} of weights: {total_params_before - total_params_after:,} parameters removed")
        logger.info(f"Actual sparsity achieved: {reduction:.2%}")
    
    def quantize_model(
        self,
        quantization_type: str = "dynamic",
        dtype: str = "qint8",
        calibration_data: Optional[torch.Tensor] = None,
    ) -> nn.Module:
        """
        Quantize the model to reduce memory footprint and improve inference speed.
        
        Args:
            quantization_type: Type of quantization ('static', 'dynamic', 'qat')
            dtype: Data type for quantization ('qint8', 'quint8', 'qint16', 'quint16')
            calibration_data: Data for calibration in static quantization
            
        Returns:
            Quantized PyTorch model
        """
        logger.info(f"Quantizing model with {quantization_type} quantization to {dtype}")
        
        if self.model is None:
            raise ValueError("No model loaded. Load a model first.")
        
        model = self.model
        
        # Prepare model for quantization
        if quantization_type == "dynamic":
            # Dynamic quantization
            quantized_model = torch.quantization.quantize_dynamic(
                model, qconfig_spec={nn.Linear}, dtype=torch.qint8
            )
            
        elif quantization_type == "static":
            if calibration_data is None:
                raise ValueError("Static quantization requires calibration_data")
                
            # Static quantization requires more setup
            model.eval()
            
            # Prepare the model for static quantization
            model_prepared = torch.quantization.prepare(model)
            
            # Calibrate with the provided data
            with torch.no_grad():
                for data in calibration_data:
                    model_prepared(data.to(self.device))
            
            # Convert to quantized model
            quantized_model = torch.quantization.convert(model_prepared)
            
        elif quantization_type == "qat":
            logger.warning("QAT requires training, please use a custom implementation")
            return model
        
        else:
            raise ValueError(f"Unsupported quantization type: {quantization_type}")
        
        # Update parameters count (this is an estimate)
        self.optimized_params = self._count_parameters(quantized_model)
        
        return quantized_model
    
    def export_to_onnx(
        self, 
        output_path: Optional[str] = None,
        input_shape: Optional[Tuple[int, ...]] = None,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
        optimize: bool = True,
    ) -> str:
        """
        Export the model to ONNX format.
        
        Args:
            output_path: Path to save the ONNX model
            input_shape: Shape of the input tensor, defaults to self.input_shape
            dynamic_axes: Dynamic axes for variable-length inputs
            optimize: Whether to optimize the ONNX model
            
        Returns:
            Path to the saved ONNX model
        """
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX and ONNXRuntime are required. Please install them.")
            
        if self.model is None:
            raise ValueError("No model loaded. Load a model first.")
        
        # Use default input shape if not provided
        if input_shape is None:
            input_shape = self.input_shape
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # Default output path if not provided
        if output_path is None:
            model_name = "model"
            if hasattr(self.model, '__class__'):
                model_name = self.model.__class__.__name__
            output_path = os.path.join(self.output_dir, f"{model_name}.onnx")
        
        # Default dynamic axes if not provided
        if dynamic_axes is None and len(input_shape) >= 4:
            # For image input, typically batch and spatial dimensions can be dynamic
            dynamic_axes = {
                'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                'output': {0: 'batch_size'}
            }
        
        # Export to ONNX
        logger.info(f"Exporting model to ONNX: {output_path}")
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes
        )
        
        # Optimize ONNX model if requested
        if optimize:
            self._optimize_onnx_model(output_path)
        
        logger.info(f"Model exported to {output_path}")
        return output_path
    
    def _optimize_onnx_model(self, onnx_path: str) -> None:
        """Optimize an ONNX model for better inference performance."""
        if not ONNX_AVAILABLE:
            logger.warning("ONNX optimization requires onnx and onnxruntime")
            return
        
        try:
            # Load model
            onnx_model = onnx.load(onnx_path)
            
            # Basic optimization
            from onnxruntime.transformers import optimizer
            opt_model = optimizer.optimize_model(
                onnx_path,
                model_type='generic',
                num_heads=12,  # Default value for many transformer models
                hidden_size=768  # Default value for many transformer models
            )
            
            # Save optimized model
            opt_model.save_model_to_file(onnx_path)
            
            logger.info(f"ONNX model optimized and saved to {onnx_path}")
            
        except Exception as e:
            logger.warning(f"ONNX optimization failed: {e}")
    
    def export_to_tensorrt(
        self,
        onnx_path: Optional[str] = None,
        output_path: Optional[str] = None,
        precision: str = "fp16",
        workspace_size: int = 1 << 30,  # 1GB
        input_shape: Optional[Tuple[int, ...]] = None,
    ) -> str:
        """
        Export the model to TensorRT format for GPU inference acceleration.
        
        Args:
            onnx_path: Path to the ONNX model, if None, will export to ONNX first
            output_path: Path to save the TensorRT engine
            precision: Precision mode ('fp32', 'fp16', 'int8')
            workspace_size: Maximum workspace size for TensorRT
            input_shape: Input shape for the model
            
        Returns:
            Path to the saved TensorRT engine
        """
        if not TRT_AVAILABLE:
            raise ImportError("TensorRT is required. Please install it.")
        
        # Export to ONNX first if needed
        if onnx_path is None:
            onnx_path = self.export_to_onnx(input_shape=input_shape)
        
        # Default output path
        if output_path is None:
            output_path = onnx_path.replace(".onnx", ".engine")
        
        logger.info(f"Converting ONNX model to TensorRT: {output_path}")
        
        # Create TensorRT builder and network
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        
        # Parse ONNX file
        parser = trt.OnnxParser(network, logger)
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                raise RuntimeError("Failed to parse ONNX model")
        
        # Configure builder
        config = builder.create_builder_config()
        config.max_workspace_size = workspace_size
        
        # Set precision mode
        if precision == "fp16" and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        elif precision == "int8" and builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            # INT8 requires calibration which is not implemented here
            logger.warning("INT8 requires calibration which is not implemented")
        
        # Build and save the engine
        engine = builder.build_engine(network, config)
        
        with open(output_path, 'wb') as f:
            f.write(engine.serialize())
        
        logger.info(f"TensorRT engine saved to {output_path}")
        return output_path
    
    def benchmark(
        self,
        input_shape: Optional[Tuple[int, ...]] = None,
        iterations: int = 100,
        warmup: int = 10,
    ) -> Dict[str, float]:
        """
        Benchmark the model's inference performance.
        
        Args:
            input_shape: Shape of the input tensor for benchmarking
            iterations: Number of iterations for benchmarking
            warmup: Number of warmup iterations
            
        Returns:
            Dictionary with benchmark results
        """
        if self.model is None:
            raise ValueError("No model loaded. Load a model first.")
        
        # Use default input shape if not provided
        if input_shape is None:
            input_shape = self.input_shape
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # Warmup
        logger.info(f"Warming up for {warmup} iterations...")
        with torch.no_grad():
            for _ in range(warmup):
                _ = self.model(dummy_input)
        
        # Benchmark
        logger.info(f"Benchmarking for {iterations} iterations...")
        start_time = time.time()
        with torch.no_grad():
            for _ in range(iterations):
                _ = self.model(dummy_input)
        end_time = time.time()
        
        # Calculate metrics
        elapsed_time = end_time - start_time
        avg_time = elapsed_time / iterations
        fps = iterations / elapsed_time
        
        # For CUDA, wait for all operations to complete
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        # Return benchmark results
        results = {
            "total_time": elapsed_time,
            "average_time": avg_time,
            "fps": fps,
            "iterations": iterations
        }
        
        logger.info(f"Benchmark results: {results}")
        return results
    
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

    def optimize_deformable_attention(self, model: nn.Module) -> nn.Module:
        """
        Replace standard deformable attention modules with CUDA-optimized versions
        for better performance.
        
        Args:
            model (nn.Module): The PyTorch model to optimize
            
        Returns:
            nn.Module: Model with optimized deformable attention
        """
        logger.info("Optimizing deformable attention modules with CUDA implementation...")
        
        try:
            import sys
            cuda_path = os.path.join(os.path.dirname(__file__), 'cuda')
            if cuda_path not in sys.path:
                sys.path.append(cuda_path)
            
            try:
                from deformable_attention_cuda import DeformableAttention, optimize_deformable_attention_in_model
                logger.info("CUDA implementation for deformable attention loaded successfully")
                model = optimize_deformable_attention_in_model(model)
                return model
            except ImportError:
                try:
                    logger.info("CUDA module not found, attempting to build...")
                    current_dir = os.getcwd()
                    os.chdir(cuda_path)
                    
                    # Run build command
                    build_result = os.system('python setup.py build_ext --inplace')
                    os.chdir(current_dir)
                    
                    if build_result == 0:
                        from deformable_attention_cuda import DeformableAttention, optimize_deformable_attention_in_model
                        logger.info("CUDA implementation for deformable attention built and loaded successfully")
                        model = optimize_deformable_attention_in_model(model)
                        return model
                    else:
                        logger.warning("Failed to build CUDA extension. Using CPU fallback.")
                        return model
                except Exception as e:
                    logger.warning(f"Error building CUDA extension: {str(e)}. Using CPU fallback.")
                    return model
        except Exception as e:
            logger.warning(f"Error optimizing deformable attention: {str(e)}. Using original model.")
            return model

    def distill_knowledge(
        self,
        teacher_model_path: str,
        student_model_name: str = "deim_nano",
        output_path: Optional[str] = None,
        alpha: float = 0.5,
        temperature: float = 4.0,
        epochs: int = 10,
        batch_size: int = 8,
        learning_rate: float = 1e-4,
        dataset_path: Optional[str] = None,
    ) -> nn.Module:
        """
        Distill knowledge from a larger teacher model to a smaller student model.
        Optimized for drone/edge deployment with minimal resources.
        
        Args:
            teacher_model_path: Path to the teacher model checkpoint
            student_model_name: Name of the student model (e.g., 'deim_nano', 'deim_small')
            output_path: Path to save the distilled model
            alpha: Weight for distillation loss vs task loss (0-1)
            temperature: Temperature for softening logits
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for optimization
            dataset_path: Path to dataset for distillation (if None, uses COCO val2017)
            
        Returns:
            Distilled student model
        """
        try:
            from torch.utils.data import DataLoader
            import torch.optim as optim
            import torch.nn.functional as F
            
            logger.info(f"Starting knowledge distillation from teacher to {student_model_name}")
            
            # Load teacher model
            if not os.path.exists(teacher_model_path):
                raise FileNotFoundError(f"Teacher model not found at {teacher_model_path}")
            
            logger.info(f"Loading teacher model from {teacher_model_path}")
            teacher_model = self._load_model(teacher_model_path)
            teacher_model.eval()  # Set to evaluation mode
            
            # Load or create student model
            from src.deimkit.predictor import load_model
            logger.info(f"Initializing student model: {student_model_name}")
            student_predictor = load_model(model_name=student_model_name, device=self.device)
            student_model = student_predictor.model
            
            # Prepare dataset
            if dataset_path is None:
                logger.warning("No dataset provided, using a small synthetic dataset")
                # Create synthetic dataset with random images
                X = torch.randn(100, 3, 640, 640)
                dataset = [(x, None) for x in X]  # No labels needed for distillation
            else:
                logger.info(f"Loading dataset from {dataset_path}")
                # Implement your dataset loading logic here
                # For example:
                # from src.deimkit.data import DEIMDataset
                # dataset = DEIMDataset(dataset_path)
                raise NotImplementedError("Custom dataset loading not implemented")
            
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Optimizer
            optimizer = optim.Adam(student_model.parameters(), lr=learning_rate)
            
            # Training loop
            logger.info(f"Starting distillation training for {epochs} epochs")
            for epoch in range(epochs):
                epoch_loss = 0.0
                for i, (inputs, _) in enumerate(dataloader):
                    inputs = inputs.to(self.device)
                    
                    # Forward pass for student
                    student_model.train()
                    student_outputs = student_model(inputs)
                    
                    # Forward pass for teacher (no gradient)
                    with torch.no_grad():
                        teacher_outputs = teacher_model(inputs)
                    
                    # Calculate distillation loss
                    # This assumes both models output logits in the same format
                    # Modify as needed for your specific model outputs
                    student_logits = student_outputs['logits'] if isinstance(student_outputs, dict) else student_outputs
                    teacher_logits = teacher_outputs['logits'] if isinstance(teacher_outputs, dict) else teacher_outputs
                    
                    # Knowledge distillation loss
                    distillation_loss = F.kl_div(
                        F.log_softmax(student_logits / temperature, dim=1),
                        F.softmax(teacher_logits / temperature, dim=1),
                        reduction='batchmean'
                    ) * (temperature ** 2)
                    
                    # Update weights
                    optimizer.zero_grad()
                    distillation_loss.backward()
                    optimizer.step()
                    
                    epoch_loss += distillation_loss.item()
                    
                    if i % 10 == 0:
                        logger.info(f"Epoch {epoch+1}/{epochs}, Batch {i}, Loss: {distillation_loss.item():.4f}")
                
                avg_loss = epoch_loss / len(dataloader)
                logger.info(f"Epoch {epoch+1}/{epochs} completed, Avg Loss: {avg_loss:.4f}")
            
            # Save the distilled model
            if output_path is None:
                output_path = os.path.join(self.output_dir, f"{student_model_name}_distilled.pth")
            
            torch.save(student_model.state_dict(), output_path)
            logger.info(f"Distilled model saved to {output_path}")
                        
            # Quantize the model
            student_model = self.quantize_model(
                model=student_model,
                quantization_type="dynamic"
            )
            
            return student_model
            
        except Exception as e:
            logger.error(f"Error during knowledge distillation: {str(e)}")
            raise

    def optimize_for_drone_deployment(
        self,
        model: Optional[nn.Module] = None,
        distill_from: Optional[str] = None,
        output_dir: str = "optimized_models/drone",
    ) -> Dict[str, Any]:
        """
        Optimize a model specifically for drone deployment with minimal resources.
        Applies aggressive pruning, quantization, and optionally knowledge distillation.
        
        Args:
            model: Model to optimize (if None, uses self.model)
            distill_from: Path to larger teacher model for knowledge distillation
            output_dir: Directory to save optimized models
            
        Returns:
            Dictionary with paths to optimized models and performance metrics
        """
        logger.info("Optimizing model for drone deployment")
        
        if model is None:
            if self.model is None:
                raise ValueError("No model provided and no model loaded")
            model = self.model
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Track results and paths
        results = {}
        
        # Apply knowledge distillation if requested
        if distill_from is not None:
            logger.info(f"Applying knowledge distillation from {distill_from}")
            model = self.distill_knowledge(
                teacher_model_path=distill_from,
                student_model_name=self.model_name or "deim_nano",
                output_path=os.path.join(output_dir, "distilled_model.pth")
            )
            results["distilled_model_path"] = os.path.join(output_dir, "distilled_model.pth")
        
        # 1. Apply aggressive pruning
        pruned_model = self.prune_model(
            model=model,
            backbone_sparsity=0.5,
            encoder_sparsity=0.4,
            decoder_sparsity=0.3,
        )
        
        # Save pruned model
        pruned_path = os.path.join(output_dir, "pruned_model.pth")
        torch.save(pruned_model.state_dict(), pruned_path)
        results["pruned_model_path"] = pruned_path
        
        # 2. Apply dynamic quantization
        quantized_model = self.quantize_model(
            model=pruned_model,
            quantization_type="dynamic"
        )
        
        # Save quantized model
        quantized_path = os.path.join(output_dir, "quantized_model.pth")
        torch.save(quantized_model.state_dict(), quantized_path)
        results["quantized_model_path"] = quantized_path
        
        # 3. Export to ONNX if available
        if ONNX_AVAILABLE:
            onnx_path = os.path.join(output_dir, "optimized_model.onnx")
            self.export_to_onnx(
                model=quantized_model,
                output_path=onnx_path,
                optimize=True
            )
            results["onnx_model_path"] = onnx_path
        
        # 4. Benchmark the optimized model
        try:
            benchmark_results = self.benchmark(model=quantized_model)
            results["benchmark"] = benchmark_results
        except Exception as e:
            logger.warning(f"Benchmarking failed: {e}")
        
        logger.info("Model optimization for drone deployment complete")
        return results
    
    # Replace the old optimize_model with this simplified version
    def optimize_model(
        self, 
        distill_from: Optional[str] = None,
        output_dir: str = "optimized_models",
    ) -> Dict[str, Any]:
        """
        Optimize the model for drone/edge deployment.
        
        Args:
            distill_from: Path to larger teacher model for knowledge distillation
            output_dir: Directory to save optimized models
            
        Returns:
            Dictionary with paths to optimized models and performance metrics
        """
        return self.optimize_for_drone_deployment(
            model=self.model,
            distill_from=distill_from,
            output_dir=output_dir
        ) 