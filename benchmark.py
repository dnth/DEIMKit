#!/usr/bin/env python3
"""
Benchmarking tool for comparing different model implementations.
This tool can compare DEIM variants and YOLOv10 with different optimization techniques.
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn

from model_optimizer import ModelOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Benchmark")

# Try importing YOLOv10 from Ultralytics
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("Ultralytics YOLO not available. Install with: pip install ultralytics")

@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    model_path: str
    model_type: str  # 'deim' or 'yolo'
    model_size: str  # 'n', 's', 'm', 'l', 'x'
    optimization: Optional[str] = None  # 'pruned', 'quantized', 'onnx', 'tensorrt'
    input_shape: Tuple[int, ...] = (1, 3, 640, 640)
    device: str = "auto"
    batch_sizes: List[int] = field(default_factory=lambda: [1, 4, 8])
    iterations: int = 100
    warmup: int = 10
    custom_name: Optional[str] = None
    
    @property
    def name(self) -> str:
        """Generate a descriptive name for this benchmark configuration."""
        opt = f"_{self.optimization}" if self.optimization else ""
        custom = f"_{self.custom_name}" if self.custom_name else ""
        return f"{self.model_type}_{self.model_size}{opt}{custom}"

class ModelBenchmark:
    """Framework for benchmarking and comparing different model implementations."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        """
        Initialize the benchmarking framework.
        
        Args:
            output_dir: Directory to save benchmark results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Check for CUDA
        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("CUDA not available, using CPU")
        
        # Results storage
        self.results = {}
    
    def benchmark_model(self, config: BenchmarkConfig) -> Dict[str, Any]:
        """
        Benchmark a single model configuration.
        
        Args:
            config: Benchmark configuration
            
        Returns:
            Dictionary with benchmark results
        """
        logger.info(f"Benchmarking {config.name}...")
        
        # Resolve device
        device = config.device
        if device == "auto":
            device = "cuda" if self.cuda_available else "cpu"
        
        results = {
            "config": {k: v for k, v in config.__dict__.items()},
            "metrics": {},
            "batch_results": {}
        }
        
        # Load model based on type
        try:
            model = self._load_model(config)
            
            # Run benchmarks for each batch size
            for batch_size in config.batch_sizes:
                # Adjust input shape for batch size
                input_shape = list(config.input_shape)
                input_shape[0] = batch_size
                input_shape = tuple(input_shape)
                
                # Benchmark
                batch_results = self._run_benchmark(
                    model, 
                    input_shape,
                    device,
                    config.iterations,
                    config.warmup
                )
                
                results["batch_results"][str(batch_size)] = batch_results
            
            # Calculate aggregate metrics
            results["metrics"] = self._calculate_aggregate_metrics(results["batch_results"])
            
            # Store results
            self.results[config.name] = results
            
            logger.info(f"Benchmark complete for {config.name}")
            logger.info(f"Average latency: {results['metrics']['average_latency']:.2f} ms")
            logger.info(f"Average throughput: {results['metrics']['average_throughput']:.2f} FPS")
            
            return results
            
        except Exception as e:
            logger.error(f"Error benchmarking {config.name}: {e}")
            return {"error": str(e)}
    
    def _load_model(self, config: BenchmarkConfig) -> nn.Module:
        """Load a model based on the benchmark configuration."""
        if config.model_type.lower() == "deim":
            return self._load_deim_model(config)
        elif config.model_type.lower() == "yolo":
            return self._load_yolo_model(config)
        else:
            # Generic model loading through ModelOptimizer
            optimizer = ModelOptimizer(
                model_path=config.model_path,
                input_shape=config.input_shape,
                device=config.device,
                model_type=config.model_type
            )
            return optimizer.model
    
    def _load_deim_model(self, config: BenchmarkConfig) -> nn.Module:
        """Load a DEIM model."""
        try:
            # Try loading through DEIMKit if available
            try:
                from src.deimkit.predictor import load_model
                
                # Map size code to model name
                size_map = {
                    "n": "deim_hgnetv2_n",
                    "s": "deim_hgnetv2_s",
                    "m": "deim_hgnetv2_m",
                    "l": "deim_hgnetv2_l",
                    "x": "deim_hgnetv2_x",
                }
                
                model_name = size_map.get(config.model_size.lower(), config.model_size)
                
                # Initialize predictor with the model
                predictor = load_model(
                    model_name=model_name,
                    device=config.device,
                    checkpoint=config.model_path if os.path.exists(config.model_path) else None,
                    image_size=config.input_shape[2:] if len(config.input_shape) >= 4 else None
                )
                
                return predictor.model
                
            except ImportError:
                logger.warning("DEIMKit not available, loading as generic PyTorch model")
                if os.path.exists(config.model_path):
                    # Try to load as a standard PyTorch model
                    try:
                        model = torch.load(config.model_path, map_location=config.device)
                        if isinstance(model, dict) and "model" in model:
                            model = model["model"]
                        elif isinstance(model, dict) and "state_dict" in model:
                            # Create a simple model container for the state dict
                            class ModelContainer(nn.Module):
                                def __init__(self, state_dict):
                                    super().__init__()
                                    # Create placeholder parameters
                                    for key, param in state_dict.items():
                                        self.register_parameter(key.replace(".", "_"), nn.Parameter(param))
                                
                                def forward(self, x):
                                    # This is just a container, not meant for actual inference
                                    return x
                            
                            container = ModelContainer(model["state_dict"])
                            return container
                        
                        return model
                    except Exception as e:
                        logger.error(f"Error loading model: {e}")
                        raise
                else:
                    logger.error(f"Model file not found: {config.model_path}")
                    raise FileNotFoundError(f"Model file not found: {config.model_path}")
        except Exception as e:
            logger.error(f"Error loading DEIM model: {e}")
            raise
    
    def _load_yolo_model(self, config: BenchmarkConfig) -> nn.Module:
        """Load a YOLO model."""
        if not YOLO_AVAILABLE:
            raise ImportError("Ultralytics YOLO not available. Install with: pip install ultralytics")
        
        try:
            # Map size code to model name
            size_map = {
                "n": "yolov10n",
                "s": "yolov10s",
                "m": "yolov10m",
                "b": "yolov10b",  # Balanced variant
                "l": "yolov10l",
                "x": "yolov10x",
            }
            
            model_name = size_map.get(config.model_size.lower(), config.model_size)
            
            # Load from path if it exists, otherwise use pretrained
            if os.path.exists(config.model_path):
                model = YOLO(config.model_path)
            else:
                model = YOLO(model_name)
            
            # Extract the PyTorch model
            return model.model
            
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            raise
    
    def _run_benchmark(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        device: str,
        iterations: int,
        warmup: int
    ) -> Dict[str, float]:
        """Run benchmark on a model."""
        model.to(device)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy_input)
        
        # Benchmark
        latencies = []
        with torch.no_grad():
            for _ in range(iterations):
                start_time = time.time()
                _ = model(dummy_input)
                
                # For CUDA, synchronize to get accurate timing
                if device == "cuda":
                    torch.cuda.synchronize()
                    
                latencies.append((time.time() - start_time) * 1000)  # Convert to ms
        
        # Calculate metrics
        avg_latency = sum(latencies) / len(latencies)
        p50_latency = sorted(latencies)[len(latencies) // 2]
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
        p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]
        min_latency = min(latencies)
        max_latency = max(latencies)
        throughput = 1000 / avg_latency  # FPS
        
        # Calculate standard deviation
        std_dev = (sum((l - avg_latency) ** 2 for l in latencies) / len(latencies)) ** 0.5
        
        return {
            "average_latency": avg_latency,
            "p50_latency": p50_latency,
            "p95_latency": p95_latency,
            "p99_latency": p99_latency,
            "min_latency": min_latency,
            "max_latency": max_latency,
            "throughput": throughput,
            "std_dev": std_dev,
            "latencies": latencies,
            "input_shape": input_shape,
            "device": device,
            "iterations": iterations
        }
    
    def _calculate_aggregate_metrics(self, batch_results: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Calculate aggregate metrics across all batch sizes."""
        # Default to metrics from batch size 1 if available
        default_batch = "1" if "1" in batch_results else list(batch_results.keys())[0]
        
        metrics = {
            "average_latency": batch_results[default_batch]["average_latency"],
            "p95_latency": batch_results[default_batch]["p95_latency"],
            "min_latency": min(r["min_latency"] for r in batch_results.values()),
            "max_latency": max(r["max_latency"] for r in batch_results.values()),
            "average_throughput": sum(r["throughput"] for r in batch_results.values()) / len(batch_results),
            "max_throughput": max(r["throughput"] for r in batch_results.values()),
            "batch_sizes_tested": len(batch_results)
        }
        
        return metrics
    
    def benchmark_multiple(self, configs: List[BenchmarkConfig]) -> Dict[str, Dict[str, Any]]:
        """
        Benchmark multiple model configurations.
        
        Args:
            configs: List of benchmark configurations
            
        Returns:
            Dictionary with benchmark results for each configuration
        """
        results = {}
        for config in configs:
            results[config.name] = self.benchmark_model(config)
        
        return results
    
    def save_results(self, filename: str = "benchmark_results.json") -> str:
        """
        Save benchmark results to a JSON file.
        
        Args:
            filename: Name of the output file
            
        Returns:
            Path to the saved file
        """
        output_path = self.output_dir / filename
        
        # Convert results to JSON-serializable format (remove numpy arrays)
        serializable_results = {}
        for model_name, result in self.results.items():
            serializable_results[model_name] = {
                "config": result.get("config", {}),
                "metrics": result.get("metrics", {}),
                "batch_results": {}
            }
            
            for batch_size, batch_result in result.get("batch_results", {}).items():
                serializable_batch = {k: v for k, v in batch_result.items() if k != "latencies"}
                if "latencies" in batch_result:
                    # Store statistics instead of full array
                    latencies = batch_result["latencies"]
                    serializable_batch["latency_stats"] = {
                        "mean": sum(latencies) / len(latencies),
                        "std": (sum((l - sum(latencies) / len(latencies)) ** 2 for l in latencies) / len(latencies)) ** 0.5,
                        "min": min(latencies),
                        "max": max(latencies),
                        "median": sorted(latencies)[len(latencies) // 2],
                        "p95": sorted(latencies)[int(len(latencies) * 0.95)],
                        "p99": sorted(latencies)[int(len(latencies) * 0.99)]
                    }
                
                serializable_results[model_name]["batch_results"][batch_size] = serializable_batch
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Benchmark results saved to {output_path}")
        return str(output_path)
    
    def print_comparison(self, metric: str = "average_latency") -> None:
        """
        Print a comparison table of benchmark results.
        
        Args:
            metric: Metric to compare ('average_latency', 'throughput', 'p95_latency')
        """
        if not self.results:
            logger.warning("No benchmark results available")
            return
        
        print(f"\n{'Model':<30} | {metric:<20} | {'Throughput (FPS)':<20}")
        print("-" * 75)
        
        for model_name, result in sorted(self.results.items()):
            if "metrics" not in result:
                continue
                
            metrics = result["metrics"]
            value = metrics.get(metric, "N/A")
            throughput = metrics.get("average_throughput", "N/A")
            
            if isinstance(value, (int, float)):
                value_str = f"{value:.2f}" + (" ms" if "latency" in metric else "")
            else:
                value_str = str(value)
                
            if isinstance(throughput, (int, float)):
                throughput_str = f"{throughput:.2f}"
            else:
                throughput_str = str(throughput)
                
            print(f"{model_name:<30} | {value_str:<20} | {throughput_str:<20}")
    
    def plot_comparison(
        self,
        metric: str = "average_latency",
        output_path: Optional[str] = None,
        log_scale: bool = False,
        sort_by: str = "value"  # 'value', 'name', or 'model_type'
    ) -> str:
        """
        Plot a comparison chart of benchmark results.
        
        Args:
            metric: Metric to compare
            output_path: Path to save the plot
            log_scale: Whether to use log scale for y-axis
            sort_by: How to sort the data ('value', 'name', or 'model_type')
            
        Returns:
            Path to the saved plot
        """
        if not self.results:
            logger.warning("No benchmark results available")
            return ""
        
        # Extract data
        model_names = []
        values = []
        model_types = []
        
        for model_name, result in self.results.items():
            if "metrics" not in result or metric not in result["metrics"]:
                continue
                
            model_names.append(model_name)
            values.append(result["metrics"][metric])
            
            # Extract model type from name
            model_type = model_name.split('_')[0]
            model_types.append(model_type)
        
        # Sort data
        if sort_by == "value":
            # Sort by metric value
            sorted_indices = np.argsort(values)
        elif sort_by == "name":
            # Sort by model name
            sorted_indices = np.argsort(model_names)
        elif sort_by == "model_type":
            # Sort by model type first, then by value
            sorted_indices = np.lexsort((values, model_types))
        else:
            sorted_indices = range(len(model_names))
            
        model_names = [model_names[i] for i in sorted_indices]
        values = [values[i] for i in sorted_indices]
        model_types = [model_types[i] for i in sorted_indices]
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Assign colors based on model type
        unique_types = list(set(model_types))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_types)))
        type_to_color = {t: colors[i] for i, t in enumerate(unique_types)}
        
        bar_colors = [type_to_color[t] for t in model_types]
        
        # Plot
        plt.bar(model_names, values, color=bar_colors)
        
        # Formatting
        plt.xlabel('Model')
        
        metric_label = metric.replace('_', ' ').title()
        if "latency" in metric.lower():
            metric_label += " (ms)"
        elif "throughput" in metric.lower():
            metric_label += " (FPS)"
            
        plt.ylabel(metric_label)
        
        plt.title(f'Comparison of {metric_label} Across Models')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if log_scale and all(v > 0 for v in values):
            plt.yscale('log')
        
        # Add legend for model types
        legend_handles = [plt.Rectangle((0, 0), 1, 1, color=type_to_color[t]) for t in unique_types]
        plt.legend(legend_handles, unique_types, loc='best')
        
        # Save plot
        if output_path is None:
            output_path = self.output_dir / f"comparison_{metric}.png"
        
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Comparison plot saved to {output_path}")
        return str(output_path)

    def benchmark_on_images(
        self,
        model: nn.Module,
        image_paths: List[str],
        input_size: Tuple[int, int] = (640, 640),
        device: str = "auto",
        iterations: int = 3,
        warmup: int = 1,
        preprocess_fn: Optional[Callable] = None,
        postprocess_fn: Optional[Callable] = None,
        save_output: bool = False,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Benchmark a model on real image data.
        
        Args:
            model: PyTorch model to benchmark
            image_paths: Paths to image files for benchmarking
            input_size: Size to resize images to (height, width)
            device: Device to run on ('cpu', 'cuda', 'auto')
            iterations: Number of iterations per image
            warmup: Number of warmup iterations
            preprocess_fn: Custom preprocessing function, if None uses default
            postprocess_fn: Custom postprocessing function
            save_output: Whether to save output visualizations
            output_dir: Directory to save outputs, if None uses self.output_dir
        
        Returns:
            Dictionary with benchmark results
        """
        import time
        from PIL import Image
        import torchvision.transforms as T
        
        if device == "auto":
            device = "cuda" if self.cuda_available else "cpu"
        
        logger.info(f"Benchmarking on {len(image_paths)} images with {iterations} iterations per image")
        
        # Set up preprocessing function
        if preprocess_fn is None:
            # Default preprocessing: resize, convert to tensor, normalize
            preprocess_fn = T.Compose([
                T.Resize(input_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        # Move model to device and set eval mode
        model.to(device)
        model.eval()
        
        # Initialize visualization directory
        if save_output:
            if output_dir is None:
                output_dir = self.output_dir / "visualizations"
            else:
                output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
        
        # Collect results
        latencies = []
        all_outputs = []
        
        for img_path in image_paths:
            try:
                # Load and preprocess image
                image = Image.open(img_path).convert("RGB")
                original_size = image.size  # (width, height)
                
                # Apply preprocessing
                input_tensor = preprocess_fn(image).unsqueeze(0).to(device)
                
                # Warmup
                with torch.no_grad():
                    for _ in range(warmup):
                        _ = model(input_tensor)
                        
                        # For CUDA, wait for all operations to complete
                        if device == "cuda":
                            torch.cuda.synchronize()
                
                # Benchmark iterations
                for _ in range(iterations):
                    # Measure inference time
                    start_time = time.time()
                    
                    with torch.no_grad():
                        output = model(input_tensor)
                    
                    # For CUDA, wait for all operations to complete
                    if device == "cuda":
                        torch.cuda.synchronize()
                    
                    end_time = time.time()
                    latency = (end_time - start_time) * 1000  # Convert to ms
                    latencies.append(latency)
                    
                    all_outputs.append((img_path, output))
            
            except Exception as e:
                logger.error(f"Error processing image {img_path}: {e}")
        
        # Process outputs if postprocessing function provided
        if postprocess_fn is not None and save_output:
            logger.info("Applying postprocessing and saving visualizations")
            for img_path, output in all_outputs:
                try:
                    # Load original image for visualization
                    image = Image.open(img_path).convert("RGB")
                    
                    # Apply postprocessing
                    processed_output = postprocess_fn(output, image)
                    
                    # Save visualization
                    if save_output:
                        filename = os.path.basename(img_path)
                        output_path = output_dir / f"output_{filename}"
                        
                        if isinstance(processed_output, Image.Image):
                            processed_output.save(output_path)
                        elif isinstance(processed_output, np.ndarray):
                            Image.fromarray(processed_output).save(output_path)
                        else:
                            logger.warning(f"Unsupported output type: {type(processed_output)}")
                
                except Exception as e:
                    logger.error(f"Error in postprocessing for {img_path}: {e}")
        
        # Calculate statistics
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        min_latency = min(latencies) if latencies else 0
        max_latency = max(latencies) if latencies else 0
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0
        std_dev = np.std(latencies) if latencies else 0
        
        results = {
            "average_latency": avg_latency,
            "min_latency": min_latency,
            "max_latency": max_latency,
            "p95_latency": p95_latency,
            "std_dev": std_dev,
            "num_images": len(image_paths),
            "iterations_per_image": iterations,
            "throughput": 1000 / avg_latency if avg_latency > 0 else 0,  # FPS
            "device": device
        }
        
        logger.info(f"Image benchmark complete: {results['average_latency']:.2f}ms average latency")
        return results

    def benchmark_deformable_attention(
        self,
        batch_size: int = 8,
        spatial_size: int = 64,
        num_heads: int = 8,
        channels: int = 64,
        num_levels: int = 4,
        num_points: int = 8,
        iterations: int = 100,
        warmup: int = 10,
        compare_cpu_cuda: bool = True,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Benchmark the deformable attention operation comparing CPU vs CUDA implementations.
        
        Args:
            batch_size: Batch size for the test
            spatial_size: Spatial size of the feature map (H*W)
            num_heads: Number of attention heads
            channels: Number of channels per head
            num_levels: Number of feature levels
            num_points: Number of sampling points
            iterations: Number of iterations for the benchmark
            warmup: Number of warmup iterations
            compare_cpu_cuda: Whether to compare CPU and CUDA implementations
            save_results: Whether to save the benchmark results
            
        Returns:
            Dictionary with benchmark results
        """
        logger.info("Benchmarking deformable attention operation...")
        
        if not torch.cuda.is_available() and compare_cpu_cuda:
            logger.warning("CUDA is not available, only running CPU benchmark")
            compare_cpu_cuda = False
        
        try:
            # Try to import the deformable attention module
            import sys
            cuda_path = os.path.join(os.path.dirname(__file__), 'cuda')
            if cuda_path not in sys.path:
                sys.path.append(cuda_path)
                
            try:
                from deformable_attention_cuda import DeformableAttentionFunction, CUDA_AVAILABLE
                logger.info("CUDA implementation for deformable attention loaded successfully")
                cuda_available = CUDA_AVAILABLE
            except ImportError:
                logger.warning("CUDA module not found, trying to build...")
                current_dir = os.getcwd()
                
                # Try to build the module
                os.chdir(cuda_path)
                build_result = os.system('python setup.py build_ext --inplace')
                os.chdir(current_dir)
                
                if build_result == 0:
                    from deformable_attention_cuda import DeformableAttentionFunction, CUDA_AVAILABLE
                    logger.info("CUDA implementation built and loaded successfully")
                    cuda_available = CUDA_AVAILABLE
                else:
                    logger.warning("Failed to build CUDA extension")
                    cuda_available = False
        except Exception as e:
            logger.error(f"Error loading deformable attention module: {str(e)}")
            return {"error": str(e)}
        
        # Create dummy inputs
        height = width = int(spatial_size ** 0.5)
        
        value = torch.randn(batch_size, spatial_size, num_heads, channels)
        sampling_locations = torch.rand(batch_size, batch_size, num_heads, num_levels, num_points, 2)
        attention_weights = torch.rand(batch_size, batch_size, num_heads, num_levels, num_points)
        
        # Function to time execution
        def time_execution(func, *args, device="cpu", warmup_iters=warmup, bench_iters=iterations):
            # Move inputs to device
            device_args = [arg.to(device) for arg in args]
            
            # Warmup
            for _ in range(warmup_iters):
                _ = func(*device_args)
                if device == "cuda":
                    torch.cuda.synchronize()
            
            # Benchmark
            latencies = []
            for _ in range(bench_iters):
                start_time = time.time()
                _ = func(*device_args)
                
                if device == "cuda":
                    torch.cuda.synchronize()
                
                latencies.append((time.time() - start_time) * 1000)  # ms
            
            avg_latency = sum(latencies) / len(latencies)
            throughput = 1000 / avg_latency
            
            return {
                "device": device,
                "average_latency": avg_latency,
                "throughput": throughput,
                "latencies": latencies,
                "min_latency": min(latencies),
                "max_latency": max(latencies),
                "p50_latency": sorted(latencies)[len(latencies) // 2],
                "p95_latency": sorted(latencies)[int(len(latencies) * 0.95)],
                "batch_size": batch_size,
                "spatial_size": spatial_size,
                "num_heads": num_heads,
                "channels": channels
            }
        
        results = {}
        
        # Benchmark CPU implementation
        logger.info("Benchmarking CPU implementation...")
        cpu_results = time_execution(
            DeformableAttentionFunction.forward_cpu,
            value, sampling_locations, attention_weights,
            device="cpu"
        )
        results["cpu"] = cpu_results
        
        # Benchmark CUDA implementation if available
        if cuda_available and compare_cpu_cuda:
            logger.info("Benchmarking CUDA implementation...")
            cuda_results = time_execution(
                lambda v, s, a: DeformableAttentionFunction.apply(v, s, a),
                value, sampling_locations, attention_weights,
                device="cuda"
            )
            results["cuda"] = cuda_results
            
            # Calculate speedup
            speedup = cpu_results["average_latency"] / cuda_results["average_latency"]
            results["speedup"] = speedup
            logger.info(f"CUDA implementation is {speedup:.2f}x faster than CPU")
        
        # Save results
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.output_dir / f"deform_attn_benchmark_{timestamp}.json"
            
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=2, cls=NumpyEncoder)
            
            logger.info(f"Benchmark results saved to {save_path}")
            
            # Create a visualization
            self._plot_deformable_attn_comparison(results, self.output_dir)
        
        return results
    
    def _plot_deformable_attn_comparison(self, results: Dict[str, Any], output_dir: Path):
        """Create visualization for deformable attention benchmark results."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Skip if matplotlib is not available
            if not results.get("cuda"):
                logger.warning("CUDA results not available, skipping comparison plot")
                return
            
            # Prepare data
            devices = ["CPU", "CUDA"]
            avg_latencies = [
                results["cpu"]["average_latency"],
                results["cuda"]["average_latency"]
            ]
            throughputs = [
                results["cpu"]["throughput"],
                results["cuda"]["throughput"]
            ]
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Plot latency comparison
            bar_width = 0.35
            x = np.arange(len(devices))
            
            bars1 = ax1.bar(x, avg_latencies, bar_width, label='Average Latency (ms)')
            ax1.set_xlabel('Device')
            ax1.set_ylabel('Latency (ms)')
            ax1.set_title('Deformable Attention Latency Comparison')
            ax1.set_xticks(x)
            ax1.set_xticklabels(devices)
            
            # Add latency values on bars
            for bar in bars1:
                height = bar.get_height()
                ax1.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
            
            # Plot throughput comparison
            bars2 = ax2.bar(x, throughputs, bar_width, label='Throughput (samples/sec)')
            ax2.set_xlabel('Device')
            ax2.set_ylabel('Throughput (samples/sec)')
            ax2.set_title('Deformable Attention Throughput Comparison')
            ax2.set_xticks(x)
            ax2.set_xticklabels(devices)
            
            # Add throughput values on bars
            for bar in bars2:
                height = bar.get_height()
                ax2.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
            
            # Add speedup annotation
            speedup = results.get("speedup", 0)
            ax1.annotate(f'CUDA is {speedup:.2f}x faster',
                      xy=(0.5, 0.9),
                      xycoords='axes fraction',
                      ha='center',
                      va='center',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Save figure
            plt.tight_layout()
            save_path = output_dir / f"deform_attn_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(save_path)
            logger.info(f"Comparison plot saved to {save_path}")
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")


def get_default_benchmarks(dataset_path: Optional[str] = None) -> Dict[str, List[BenchmarkConfig]]:
    """
    Get default benchmark configurations for common scenarios.
    
    Args:
        dataset_path: Path to a dataset for evaluation (optional)
        
    Returns:
        Dictionary with benchmark configurations for different scenarios
    """
    # CPU configurations
    cpu_configs = [
        # DEIM variants
        BenchmarkConfig(model_path="", model_type="deim", model_size="n", device="cpu"),
        BenchmarkConfig(model_path="", model_type="deim", model_size="s", device="cpu"),
        # YOLOv10 variants
        BenchmarkConfig(model_path="", model_type="yolo", model_size="n", device="cpu"),
        BenchmarkConfig(model_path="", model_type="yolo", model_size="s", device="cpu"),
        # Optimized variants
        BenchmarkConfig(model_path="", model_type="deim", model_size="s", 
                        optimization="pruned", device="cpu", custom_name="pruned_50pct"),
        BenchmarkConfig(model_path="", model_type="deim", model_size="s", 
                        optimization="quantized", device="cpu", custom_name="int8"),
    ]
    
    # GPU configurations (only if CUDA available)
    gpu_configs = []
    if torch.cuda.is_available():
        gpu_configs = [
            # DEIM variants
            BenchmarkConfig(model_path="", model_type="deim", model_size="n", device="cuda"),
            BenchmarkConfig(model_path="", model_type="deim", model_size="s", device="cuda"),
            BenchmarkConfig(model_path="", model_type="deim", model_size="m", device="cuda"),
            # YOLOv10 variants
            BenchmarkConfig(model_path="", model_type="yolo", model_size="n", device="cuda"),
            BenchmarkConfig(model_path="", model_type="yolo", model_size="s", device="cuda"),
            BenchmarkConfig(model_path="", model_type="yolo", model_size="m", device="cuda"),
            # Optimized variants
            BenchmarkConfig(model_path="", model_type="deim", model_size="s", 
                           optimization="tensorrt", device="cuda", custom_name="tensorrt_fp16"),
        ]
    
    return {
        "cpu": cpu_configs,
        "gpu": gpu_configs,
    }


def main():
    """Command-line interface for the benchmark tool."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark tool for model comparison")
    parser.add_argument("--default", action="store_true", help="Run default benchmark suite")
    parser.add_argument("--config", type=str, help="Path to a JSON config file with benchmark configurations")
    parser.add_argument("--output-dir", type=str, default="benchmark_results", help="Output directory for results")
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations for each benchmark")
    parser.add_argument("--benchmark-deformable", action="store_true", help="Benchmark deformable attention operations")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for deformable attention benchmark")
    parser.add_argument("--compare-cpu-cuda", action="store_true", help="Compare CPU and CUDA implementations")
    
    args = parser.parse_args()
    
    benchmark = ModelBenchmark(output_dir=args.output_dir)
    
    if args.benchmark_deformable:
        logger.info("Running deformable attention benchmark")
        benchmark.benchmark_deformable_attention(
            batch_size=args.batch_size,
            iterations=args.iterations,
            compare_cpu_cuda=args.compare_cpu_cuda
        )
    elif args.default:
        logger.info("Running default benchmark suite")
        benchmarks = get_default_benchmarks()
        
        # Run CPU benchmarks
        logger.info("Running CPU benchmarks")
        benchmark.benchmark_multiple(benchmarks["cpu"])
        
        # Run GPU benchmarks if available
        if torch.cuda.is_available():
            logger.info("Running GPU benchmarks")
            benchmark.benchmark_multiple(benchmarks["gpu"])
            
    elif args.config:
        logger.info(f"Loading benchmark configurations from {args.config}")
        with open(args.config, 'r') as f:
            config_data = json.load(f)
            
        configs = []
        for cfg in config_data:
            configs.append(BenchmarkConfig(**cfg))
            
        logger.info(f"Running {len(configs)} benchmark configurations")
        benchmark.benchmark_multiple(configs)
    else:
        logger.error("No benchmark configuration provided")
        parser.print_help()
        return
    
    # Save and visualize results
    benchmark.save_results()
    benchmark.print_comparison(metric="average_latency")
    benchmark.print_comparison(metric="throughput")


if __name__ == "__main__":
    main() 