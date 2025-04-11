#!/usr/bin/env python
"""
TensorRT Engine Export Script
-----------------------------

This script exports an ONNX model to a TensorRT engine using the Python API.
It ensures version compatibility with the installed TensorRT package (10.9.0.34).

Usage:
    poetry run python scripts/export_trt.py \
        --onnx-path model.onnx \
        --engine-path model.engine \
        --input-shape 1,3,416,416 \
        --fp16 \
        --workspace-size 4096
"""

import os
import time
import argparse
import logging
import numpy as np
import tensorrt as trt
from pathlib import Path
import onnx


def setup_logger():
    """Set up a console logger."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    return logging.getLogger('TensorRT-Export')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Export ONNX model to TensorRT engine')
    
    parser.add_argument('--onnx-path', type=str, required=True,
                        help='Path to the ONNX model')
    parser.add_argument('--engine-path', type=str, required=True,
                        help='Path to save the TensorRT engine')
    parser.add_argument('--input-shape', type=str, default='1,3,416,416',
                        help='Input shape in format batch,channels,height,width')
    parser.add_argument('--fp16', action='store_true', default=False,
                        help='Enable FP16 precision')
    parser.add_argument('--int8', action='store_true', default=False,
                        help='Enable INT8 precision')
    parser.add_argument('--workspace-size', type=int, default=4096, 
                        help='Workspace size in MB')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Enable verbose logging')
    parser.add_argument('--version-compatible', action='store_true', default=False,
                        help='Build a version compatible engine')
    parser.add_argument('--debug-info', action='store_true', default=False,
                        help='Save detailed engine information for debugging')
    
    return parser.parse_args()


def build_engine(args, logger):
    """Build TensorRT engine from ONNX model."""
    logger.info(f"Building TensorRT engine from {args.onnx_path}")
    logger.info(f"Using TensorRT version: {trt.__version__}")
    
    # Parse input shape
    input_shape = tuple(map(int, args.input_shape.split(',')))
    logger.info(f"Input shape: {input_shape}")
    
    # Create builder and network
    trt_logger = trt.Logger(trt.Logger.VERBOSE if args.verbose else trt.Logger.INFO)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    
    # Set workspace size - using the newer API for TensorRT 10.x
    # In TensorRT 10.x, max_workspace_size is replaced with memory_pool_limits
    memory_size = args.workspace_size * 1024 * 1024  # Convert to bytes
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, memory_size)
    logger.info(f"Workspace size: {args.workspace_size} MB")
    
    # Set precision flags
    if args.fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("Enabled FP16 precision")
        else:
            logger.warning("FP16 not supported on this platform, using FP32")
    
    if args.int8:
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            logger.info("Enabled INT8 precision")
        else:
            logger.warning("INT8 not supported on this platform, using FP32/FP16")
    
    # Enable version compatibility if requested
    if args.version_compatible:
        config.set_flag(trt.BuilderFlag.VERSION_COMPATIBLE)
        logger.info("Building version compatible engine")
    
    # Parse ONNX model
    parser = trt.OnnxParser(network, trt_logger)
    
    # Load and parse ONNX file
    with open(args.onnx_path, 'rb') as model:
        onnx_model = model.read()
        
    if not parser.parse(onnx_model):
        logger.error("Failed to parse ONNX model")
        for error in range(parser.num_errors):
            logger.error(parser.get_error(error))
        return None
    
    # Load ONNX model to get input shapes
    onnx_model = onnx.load(args.onnx_path)
    
    # Extract input info
    batch_size = input_shape[0]
    min_batch = 1
    opt_batch = batch_size
    max_batch = batch_size * 2  # Allow for some growth in batch size
    
    # Create optimization profile for dynamic batch size
    profile = builder.create_optimization_profile()
    
    # Set shapes for each input
    for inp in onnx_model.graph.input:
        input_name = inp.name
        shape_proto = inp.type.tensor_type.shape
        
        # Get shape dimensions
        dims = [d.dim_value if d.dim_value > 0 else -1 for d in shape_proto.dim]
        logger.info(f"Original dims for {input_name}: {dims}")
        
        if input_name == "images":
            # For the main input tensor (images)
            min_shape = (min_batch, 3, input_shape[2], input_shape[3])
            opt_shape = (opt_batch, 3, input_shape[2], input_shape[3])
            max_shape = (max_batch, 3, input_shape[2], input_shape[3])
            
            logger.info(f"Setting dynamic shapes for {input_name}: min={min_shape}, opt={opt_shape}, max={max_shape}")
            profile.set_shape(input_name, min_shape, opt_shape, max_shape)
            
        elif input_name == "orig_target_sizes":
            # For orig_target_sizes which has shape [batch_size, 2]
            min_shape = (min_batch, 2)
            opt_shape = (opt_batch, 2)
            max_shape = (max_batch, 2)
            
            logger.info(f"Setting dynamic shapes for {input_name}: min={min_shape}, opt={opt_shape}, max={max_shape}")
            profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    
    # Add the profile to the config
    config.add_optimization_profile(profile)
    
    # Build and serialize engine (using TensorRT 10.x API)
    logger.info("Building TensorRT engine (this may take a while)...")
    start_time = time.time()
    
    # In TensorRT 10.x, build_engine is replaced with build_serialized_network
    serialized_engine = builder.build_serialized_network(network, config)
    build_time = time.time() - start_time
    
    if serialized_engine is None:
        logger.error("Failed to build TensorRT engine")
        return None
    
    logger.info(f"Engine built successfully in {build_time:.2f} seconds")
    
    # Save engine to file
    with open(args.engine_path, 'wb') as f:
        f.write(serialized_engine)
    
    logger.info(f"Engine saved to {args.engine_path}")
    
    # Create runtime and deserialize engine for return value
    runtime = trt.Runtime(trt_logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    
    return engine


def main():
    """Main function to build and export TensorRT engine."""
    args = parse_args()
    logger = setup_logger()
    
    # Create output directory if it doesn't exist
    engine_dir = os.path.dirname(args.engine_path)
    if engine_dir:
        os.makedirs(engine_dir, exist_ok=True)
    
    # Build engine
    start_time = time.time()
    engine = build_engine(args, logger)
    total_time = time.time() - start_time
    
    if engine:
        logger.info(f"TensorRT engine created successfully in {total_time:.2f} seconds")
        # Get engine file size
        engine_size_mb = Path(args.engine_path).stat().st_size / (1024 * 1024)
        logger.info(f"Engine file size: {engine_size_mb:.2f} MB")
        
        # Save debug information if requested
        if args.debug_info:
            debug_file = Path(args.engine_path).with_suffix('.info.txt')
            try:
                with open(debug_file, 'w') as f:
                    f.write(f"TensorRT Engine Information\n")
                    f.write(f"===========================\n\n")
                    f.write(f"TensorRT Version: {trt.__version__}\n")
                    f.write(f"Engine Path: {args.engine_path}\n")
                    f.write(f"Engine Size: {engine_size_mb:.2f} MB\n\n")
                    
                    f.write(f"Input Information:\n")
                    f.write(f"----------------\n")
                    for i in range(engine.num_io_tensors):
                        if engine.get_tensor_mode(i) == trt.TensorIOMode.INPUT:
                            name = engine.get_tensor_name(i)
                            dtype = engine.get_tensor_dtype(i)
                            shape = engine.get_tensor_shape(i)
                            f.write(f"  Input {i}: {name}, Shape: {shape}, Type: {dtype}\n")
                    
                    f.write(f"\nOutput Information:\n")
                    f.write(f"-----------------\n")
                    for i in range(engine.num_io_tensors):
                        if engine.get_tensor_mode(i) == trt.TensorIOMode.OUTPUT:
                            name = engine.get_tensor_name(i)
                            dtype = engine.get_tensor_dtype(i)
                            shape = engine.get_tensor_shape(i)
                            f.write(f"  Output {i}: {name}, Shape: {shape}, Type: {dtype}\n")
                
                logger.info(f"Debug information saved to {debug_file}")
            except Exception as e:
                logger.error(f"Failed to save debug information: {e}")
    else:
        logger.error("Failed to create TensorRT engine")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 