#!/usr/bin/env python3
"""
Main interface for optimizing DEIM and YOLO models for edge deployment.
This script provides a simple command-line interface to access all optimization features.
"""

import os
import json
import logging
import argparse
import glob
import time
from pathlib import Path
from typing import Dict, Optional, Any, List, Union

import torch

from model_optimizer import ModelOptimizer
from tools.benchmark.trt_benchmark import TRTInference
from tools.benchmark.dataset import Dataset


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
) -> Dict[str, Any]:
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
        Dictionary with paths to optimized models and benchmark results
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
    
    # Store original model for benchmarking
    original_model = optimizer.model
    
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
        
    # Save the final optimized model
    output_path = str(output_dir / f"{model_type}_{model_size}_drone.pth")
    optimizer.save_model(model=pruned_model, output_path=output_path)
    
    # Store all results
    results = {
        "original_model": original_model,
        "pruned_model": pruned_model,
        "output_path": output_path
    }
    
    # Run benchmark if requested using in-memory models
    if run_benchmark:
        logger.info("skipped Benchmarking...")
    
    logger.info(f"Model optimized for drone deployment and saved to {output_path}")
    return results



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
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 