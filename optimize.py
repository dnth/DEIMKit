#!/usr/bin/env python3
"""
Main interface for optimizing DEIM and YOLO models for edge deployment.
This script provides a simple command-line interface to access all optimization features.
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any

import torch
import torch.nn as nn

from model_optimizer import ModelOptimizer
from benchmark import ModelBenchmark, BenchmarkConfig
from model_distillation import create_distilled_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("OptimizeModel")

def optimize_for_drone(
    model_path: str,
    model_type: str = "deim",
    model_name: str = "deim_hgnetv2_s",
    model_size: str = "s",
    output_dir: str = "optimized_models/drone",
    run_benchmark: bool = True,
    preserve_accuracy: bool = True
) -> str:
    """
    Optimize a model for drone deployment (low power CPU).
    Applies aggressive pruning, weight clustering, and quantization.
    
    Args:
        model_path: Path to the model
        model_type: Type of model ('deim' or 'yolo')
        model_name: Full model name (e.g., 'deim_hgnetv2_s')
        model_size: Size of model ('n', 's', 'm', 'l', 'x')
        output_dir: Output directory for optimized model
        run_benchmark: Whether to run benchmark after optimization
        preserve_accuracy: If True, use more conservative optimization to preserve accuracy
        
    Returns:
        Path to the optimized model
    """
    logger.info(f"Optimizing {model_type}_{model_size} model for drone deployment")
    logger.info(f"Accuracy preservation mode: {preserve_accuracy}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure model_name is set properly if not provided
    if not model_name and model_type.lower() == "deim" and model_size:
        size_map = {
            "n": "deim_hgnetv2_n",
            "s": "deim_hgnetv2_s",
            "m": "deim_hgnetv2_m",
            "l": "deim_hgnetv2_l",
            "x": "deim_hgnetv2_x",
        }
        model_name = size_map.get(model_size.lower(), f"deim_hgnetv2_{model_size}")
    
    # Load model
    optimizer = ModelOptimizer(
        model_path=model_path,
        model_type=model_type,
        model_name=model_name,
        device="cpu"  # Force CPU for drone deployment
    )
    
    # Apply pruning with different sparsity depending on accuracy preservation mode
    if preserve_accuracy:
        # Conservative pruning to preserve accuracy
        pruned_model = optimizer.prune_model(
            backbone_sparsity=0.4,     # 40% sparsity in backbone
            encoder_sparsity=0.3,      # 30% sparsity in encoder
            decoder_sparsity=0.1,      # 10% sparsity in decoder (preserve accuracy of final predictions)
            method="l1_unstructured"
        )
    else:
        # Aggressive pruning for maximum speed
        pruned_model = optimizer.prune_model(
            backbone_sparsity=0.6,     # 60% sparsity in backbone
            encoder_sparsity=0.5,      # 50% sparsity in encoder
            decoder_sparsity=0.3,      # 30% sparsity in decoder
            method="l1_unstructured"
        )
    
    # Save pruned model
    pruned_path = str(output_dir / f"{model_type}_{model_size}_pruned.pth")
    optimizer.save_model(pruned_model, pruned_path)
    
    # Apply weight clustering (fewer clusters = better compression but less accuracy)
    n_clusters = 16 if preserve_accuracy else 8
    clustered_model = optimizer.cluster_weights(
        model=pruned_model,
        n_clusters=n_clusters,
        min_elements_per_centroid=16,
        components=['backbone', 'encoder']  # Don't cluster decoder to preserve accuracy
    )
    
    # Optimize deformable attention if present
    optimized_model = optimizer.optimize_deformable_attention(clustered_model)
    
    # Apply dynamic quantization
    quantized_model = optimizer.quantize_model(
        model=optimized_model,
        quantization_type="dynamic",
        dtype="qint8"
    )
    
    # Save optimized model
    output_path = str(output_dir / f"{model_type}_{model_size}_drone.pth")
    optimizer.save_model(quantized_model, output_path)
    
    # Run benchmark if requested
    if run_benchmark:
        benchmark = ModelBenchmark(output_dir=str(output_dir / "benchmark"))
        
        # Benchmark original model
        original_config = BenchmarkConfig(
            model_path=model_path,
            model_type=model_type,
            model_size=model_size,
            device="cpu",
            custom_name="original"
        )
        benchmark.benchmark_model(original_config)
        
        # Benchmark pruned model only
        pruned_config = BenchmarkConfig(
            model_path=pruned_path,
            model_type=model_type,
            model_size=model_size,
            device="cpu",
            optimization="pruned",
            custom_name="pruned_only"
        )
        benchmark.benchmark_model(pruned_config)
        
        # Benchmark fully optimized model
        optimized_config = BenchmarkConfig(
            model_path=output_path,
            model_type=model_type,
            model_size=model_size,
            device="cpu",
            optimization="full_optimization",
            custom_name="drone_optimized"
        )
        benchmark.benchmark_model(optimized_config)
        
        # Compare results
        benchmark.print_comparison()
        benchmark.plot_comparison(
            metric="average_latency",
            output_path=str(output_dir / "latency_comparison.png")
        )
        benchmark.save_results(filename="drone_benchmark.json")
    
    logger.info(f"Model optimized for drone deployment and saved to {output_path}")
    return output_path


def run_comprehensive_benchmark(
    output_dir: str = "benchmark_results/comprehensive",
    deim_path: Optional[str] = None,
    yolo_path: Optional[str] = None
) -> str:
    """
    Run a comprehensive benchmark comparing different DEIM models and YOLOv10 models
    with various optimization techniques.
    
    Args:
        output_dir: Output directory for benchmark results
        deim_path: Path to DEIM model (optional)
        yolo_path: Path to YOLO model (optional)
        
    Returns:
        Path to benchmark results
    """
    logger.info("Running comprehensive benchmark")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize benchmark
    benchmark = ModelBenchmark(output_dir=str(output_dir))
    
    # Define benchmark configurations
    configs = []
    
    # DEIM model variants (original)
    deim_sizes = ["n", "s", "m", "l", "x"]
    for size in deim_sizes:
        configs.append(BenchmarkConfig(
            model_path=deim_path if deim_path else "",
            model_type="deim",
            model_size=size,
            device="auto"
        ))
    
    # YOLO model variants (original)
    yolo_sizes = ["n", "s", "m", "l", "x"]
    for size in yolo_sizes:
        configs.append(BenchmarkConfig(
            model_path=yolo_path if yolo_path else "",
            model_type="yolo",
            model_size=size,
            device="auto"
        ))
    
    # Run benchmarks
    benchmark.benchmark_multiple(configs)
    
    # Save and visualize results
    benchmark.save_results(filename="comprehensive_benchmark.json")
    benchmark.print_comparison(metric="average_latency")
    benchmark.print_comparison(metric="average_throughput")
    
    # Plot comparisons
    benchmark.plot_comparison(
        metric="average_latency",
        output_path=str(output_dir / "latency_comparison.png"),
        log_scale=True,
        sort_by="value"
    )
    
    benchmark.plot_comparison(
        metric="average_throughput",
        output_path=str(output_dir / "throughput_comparison.png"),
        sort_by="value"
    )
    
    logger.info(f"Comprehensive benchmark complete. Results saved to {output_dir}")
    return str(output_dir / "comprehensive_benchmark.json")


def distill_from_large_to_nano(
    teacher_path: str,
    student_path: str,
    dataset_path: str,
    output_dir: str = "distilled_models",
    epochs: int = 10,
    run_benchmark: bool = True
) -> str:
    """
    Distill knowledge from a large model to a nano model.
    
    Args:
        teacher_path: Path to the teacher model (larger model)
        student_path: Path to the student model (nano model)
        dataset_path: Path to the dataset for distillation
        output_dir: Output directory for distilled model
        epochs: Number of training epochs
        run_benchmark: Whether to run benchmark after distillation
        
    Returns:
        Path to the distilled model
    """
    logger.info("Starting knowledge distillation from large to nano model")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Perform distillation
    distilled_path = create_distilled_model(
        teacher_path=teacher_path,
        student_path=student_path,
        dataset_path=dataset_path,
        output_dir=str(output_dir),
        epochs=epochs,
        batch_size=8,
        temperature=4.0,
        alpha=0.5,
        lr=0.001
    )
    
    # Run benchmark if requested
    if run_benchmark:
        benchmark = ModelBenchmark(output_dir=str(output_dir / "benchmark"))
        
        # Determine model type from teacher path
        model_type = "deim"  # Default
        if "yolo" in teacher_path.lower():
            model_type = "yolo"
        
        # Benchmark teacher model
        teacher_config = BenchmarkConfig(
            model_path=teacher_path,
            model_type=model_type,
            model_size="l",  # Assume large model for teacher
            device="auto",
            custom_name="teacher"
        )
        benchmark.benchmark_model(teacher_config)
        
        # Benchmark original student model
        student_config = BenchmarkConfig(
            model_path=student_path,
            model_type=model_type,
            model_size="n",  # Assume nano model for student
            device="auto",
            custom_name="student_original"
        )
        benchmark.benchmark_model(student_config)
        
        # Benchmark distilled model
        distilled_config = BenchmarkConfig(
            model_path=distilled_path,
            model_type=model_type,
            model_size="n",  # Nano model size
            device="auto",
            custom_name="student_distilled"
        )
        benchmark.benchmark_model(distilled_config)
        
        # Compare results
        benchmark.print_comparison()
        benchmark.plot_comparison(
            metric="average_latency",
            output_path=str(output_dir / "latency_comparison.png")
        )
        benchmark.plot_comparison(
            metric="average_throughput",
            output_path=str(output_dir / "throughput_comparison.png")
        )
        benchmark.save_results(filename="distillation_benchmark.json")
    
    logger.info(f"Distillation complete. Distilled model saved to {distilled_path}")
    return distilled_path


def main():
    """Command-line interface for the optimization framework."""
    parser = argparse.ArgumentParser(description="Optimize DEIM and YOLO models for edge deployment.")
    
    parser.add_argument("--target", choices=["drone", "edge", "server", "benchmark", "distill"], 
                      help="Optimization target (drone, edge device, server, benchmark, or distill)")
    
    parser.add_argument("--model-path", help="Path to the model")
    parser.add_argument("--model-type", choices=["deim", "yolo"], default="deim",
                      help="Model type (deim or yolo)")
    parser.add_argument("--model-size", choices=["n", "s", "m", "l", "x"], default="s",
                      help="Model size (n=nano, s=small, m=medium, l=large, x=xlarge)")
    parser.add_argument("--model-name", help="Full model name (e.g., 'deim_hgnetv2_s')")
    
    parser.add_argument("--output-dir", default="optimized_models",
                      help="Output directory for optimized models")
    
    parser.add_argument("--skip-benchmark", action="store_true",
                      help="Skip benchmarking after optimization")
    
    parser.add_argument("--no-tensorrt", action="store_true",
                      help="Disable TensorRT export (for edge and server targets)")
    
    # Additional parameters for distillation
    parser.add_argument("--teacher-path", help="Path to the teacher model (for distillation)")
    parser.add_argument("--student-path", help="Path to the student model (for distillation)")
    parser.add_argument("--dataset-path", help="Path to the dataset (for distillation)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for distillation")
    
    args = parser.parse_args()
    
    # Derive model_name from model_size if not provided
    if not args.model_name and args.model_type == "deim" and args.model_size:
        size_map = {
            "n": "deim_hgnetv2_n",
            "s": "deim_hgnetv2_s",
            "m": "deim_hgnetv2_m",
            "l": "deim_hgnetv2_l",
            "x": "deim_hgnetv2_x",
        }
        args.model_name = size_map.get(args.model_size, f"deim_hgnetv2_{args.model_size}")
    
    # Execute based on target
    if args.target == "drone":
        if not args.model_path:
            parser.error("--model-path is required for drone optimization")
        
        optimize_for_drone(
            model_path=args.model_path,
            model_type=args.model_type,
            model_name=args.model_name,
            model_size=args.model_size,
            output_dir=os.path.join(args.output_dir, "drone"),
            run_benchmark=not args.skip_benchmark,
            preserve_accuracy=True
        )
    elif args.target == "benchmark":
        run_comprehensive_benchmark(
            output_dir=os.path.join(args.output_dir, "benchmark"),
            deim_path=args.model_path if args.model_type == "deim" else None,
            yolo_path=args.model_path if args.model_type == "yolo" else None
        )
    elif args.target == "distill":
        if not args.teacher_path:
            parser.error("--teacher-path is required for distillation")
        if not args.student_path:
            parser.error("--student-path is required for distillation")
        if not args.dataset_path:
            parser.error("--dataset-path is required for distillation")
        
        distill_from_large_to_nano(
            teacher_path=args.teacher_path,
            student_path=args.student_path,
            dataset_path=args.dataset_path,
            output_dir=os.path.join(args.output_dir, "distilled"),
            epochs=args.epochs,
            run_benchmark=not args.skip_benchmark
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 