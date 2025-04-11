"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""
import os
import torch
import tensorrt as trt
import argparse
import glob
from evaluator import Evaluator
import time
import numpy as np
import statistics
import json
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='TensorRT Benchmark with Evaluation')
    parser.add_argument("--engine_dir", type=str, required=True,
                        help="Directory containing model engine files or path to single engine file.")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Enable verbose logging.")
    args = parser.parse_args()
    return args

class TRTModel:
    def __init__(self, engine_path, device='cuda:0', verbose=False):
        self.engine_path = engine_path
        self.device = torch.device(device)
        self.verbose = verbose
        self.logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)
        
        # Initialize plugins (needed for some models)
        try:
            trt.init_libnvinfer_plugins(self.logger, '')
            print("TensorRT plugins initialized successfully")
        except Exception as e:
            print(f"Warning: Failed to initialize TensorRT plugins: {e}")
        
        # Load engine
        try:
            self.runtime = trt.Runtime(self.logger)
            with open(self.engine_path, 'rb') as f:
                engine_data = f.read()
                if len(engine_data) == 0:
                    raise ValueError(f"Empty engine file: {self.engine_path}")
                print(f"Engine file size: {len(engine_data) / (1024*1024):.2f} MB")
                self.engine = self.runtime.deserialize_cuda_engine(engine_data)
                if self.engine is None:
                    raise RuntimeError("Failed to deserialize engine")
        except Exception as e:
            print(f"Error loading TensorRT engine: {e}")
            raise
            
        # Create execution context    
        self.context = self.engine.create_execution_context()
        
        # Get input and output names for reference
        self.input_names = []
        self.output_names = []
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
                
        print(f"TensorRT model initialized with inputs: {self.input_names}, outputs: {self.output_names}")
        
        # Print detailed shape information
        if verbose:
            print("\nDetailed tensor information:")
            for name in self.input_names:
                shape = tuple(self.engine.get_tensor_shape(name))
                dtype = self.engine.get_tensor_dtype(name)
                print(f"  Input: {name}, shape={shape}, dtype={dtype}")
            
            for name in self.output_names:
                shape = tuple(self.engine.get_tensor_shape(name))
                dtype = self.engine.get_tensor_dtype(name)
                print(f"  Output: {name}, shape={shape}, dtype={dtype}")
        
    def __call__(self, images):
        """
        Run inference on images tensor and return results in the format
        expected by the det_engine.py evaluate function.
        
        Args:
            images: Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            List of dictionaries, one per image in batch, each containing labels, boxes, scores
        """
        try:
            # This method is called by evaluate() with a single tensor argument
            batch_size = images.shape[0]
            
            if self.verbose:
                print(f"Running inference on batch with shape: {images.shape}")
            
            # Create input dictionary based on what our engine expects
            inputs = {}
            
            # Set the primary image input 
            if "images" in self.input_names:
                inputs["images"] = images
            else:
                # This engine must have a different input name for images
                # Use the first input name as fallback
                first_input = self.input_names[0]
                print(f"Warning: 'images' input not found, using '{first_input}' instead")
                inputs[first_input] = images
                
            # Create orig_target_sizes input for each image in batch
            if "orig_target_sizes" in self.input_names:
                h, w = images.shape[2], images.shape[3]
                inputs["orig_target_sizes"] = torch.tensor([[h, w]] * batch_size, 
                                                         device=self.device)
            
            # Make sure all required inputs are provided
            missing_inputs = set(self.input_names) - set(inputs.keys())
            if missing_inputs:
                raise RuntimeError(f"Missing required inputs: {missing_inputs}")
            
            # Run actual inference, which returns direct outputs
            outputs = self._execute(inputs)
            
            # Check if outputs are valid
            if not outputs:
                raise RuntimeError("No outputs returned from TensorRT inference")
            
            # DEBUG: Print output keys and shapes to diagnose differences between models
            print(f"\nDEBUG - TensorRT outputs for {os.path.basename(self.engine_path)}:")
            print(f"Output keys: {list(outputs.keys())}")
            for name, tensor in outputs.items():
                print(f"  {name}: shape={tensor.shape}, dtype={tensor.dtype}")
                # Print sample values for diagnosis
                if tensor.numel() > 0:
                    flat_tensor = tensor.flatten()
                    sample_values = flat_tensor[:min(5, flat_tensor.numel())].tolist()
                    print(f"    Sample values: {sample_values}")
                    # Print statistics for numeric tensors
                    if tensor.dtype.is_floating_point:
                        print(f"    Min: {tensor.min().item()}, Max: {tensor.max().item()}, Mean: {tensor.float().mean().item()}")
                    # Count non-zero elements
                    if tensor.dtype in [torch.int32, torch.int64, torch.bool]:
                        print(f"    Non-zero elements: {torch.count_nonzero(tensor).item()}/{tensor.numel()}")
                else:
                    print(f"    Empty tensor")
            
            if self.verbose:
                for name, tensor in outputs.items():
                    print(f"Output '{name}' shape: {tensor.shape}")
            
            # Now we need to format the outputs as expected by the det_engine.py evaluate function
            # It expects a list of dictionaries, one per image in the batch
            if all(name in outputs for name in ["labels", "boxes", "scores"]):
                labels = outputs["labels"] 
                boxes = outputs["boxes"]
                scores = outputs["scores"]
                
                # DEBUG: Print detailed information about detection outputs
                print(f"Detection counts: labels={labels.shape}, boxes={boxes.shape}, scores={scores.shape}")
                if labels.numel() > 0:
                    print(f"Labels range: min={labels.min().item()}, max={labels.max().item()}")
                    print(f"Scores range: min={scores.min().item()}, max={scores.max().item()}")
                else:
                    print(f"WARNING: No detections found (empty tensors)")
                
                # Create list of results dicts
                results = []
                for i in range(batch_size):
                    # Extract batch item i
                    if len(labels.shape) == 2:  # [batch_size, num_detections]
                        batch_labels = labels[i]
                        batch_boxes = boxes[i]
                        batch_scores = scores[i]
                    else:
                        # If outputs don't have batch dimension, use as is for single image
                        batch_labels = labels
                        batch_boxes = boxes
                        batch_scores = scores
                    
                    # Create dictionary with this batch item's results
                    results.append({
                        "labels": batch_labels,
                        "boxes": batch_boxes,
                        "scores": batch_scores
                    })
                    
                    # DEBUG: Print per-image detection info
                    print(f"Image {i}: {batch_labels.shape[0]} detections")
                    if batch_labels.shape[0] > 0:
                        print(f"  First few detections:")
                        for j in range(min(3, batch_labels.shape[0])):
                            print(f"    Class {batch_labels[j].item()}, Score {batch_scores[j].item():.4f}, Box {batch_boxes[j].tolist()}")
                
                print(f"Returning {len(results)} results for {batch_size} images")
                return results
            else:
                # Try to adapt other output formats that might be present
                print(f"Warning: Expected standard outputs not found. Available: {list(outputs.keys())}")
                
                # Sometimes models output in different format keys
                if "pred_logits" in outputs and "pred_boxes" in outputs:
                    print("Found DETR-style outputs, converting to standard format")
                    
                    # DETR-style outputs need postprocessing
                    pred_logits = outputs["pred_logits"]  # [batch_size, num_queries, num_classes+1]
                    pred_boxes = outputs["pred_boxes"]    # [batch_size, num_queries, 4]
                    
                    # Process each image in the batch
                    results = []
                    for i in range(batch_size):
                        # Get logits and boxes for this image
                        img_logits = pred_logits[i]  # [num_queries, num_classes+1]
                        img_boxes = pred_boxes[i]    # [num_queries, 4]
                        
                        # Convert logits to probabilities
                        img_probs = torch.nn.functional.softmax(img_logits, dim=-1)
                        
                        # Get class with highest probability (excluding background)
                        num_classes = img_probs.shape[-1] - 1  # Last class is background
                        if num_classes > 0:
                            # Get scores for all classes except background
                            scores, labels = torch.max(img_probs[:, :-1], dim=1)
                            
                            # Filter out low probability detections
                            keep = scores > 0.1  # Adjust threshold as needed
                            labels = labels[keep]
                            scores = scores[keep]
                            boxes = img_boxes[keep]
                            
                            # Add 1 to labels if your model expects 1-indexed classes
                            labels = labels + 1
                            
                            results.append({
                                "labels": labels, 
                                "boxes": boxes,
                                "scores": scores
                            })
                            
                            print(f"Converted image {i}: {labels.shape[0]} detections")
                        else:
                            # No valid classes (only background)
                            results.append({
                                "labels": torch.tensor([], device=self.device, dtype=torch.int64),
                                "boxes": torch.tensor([], device=self.device, dtype=torch.float32).reshape(0, 4),
                                "scores": torch.tensor([], device=self.device, dtype=torch.float32)
                            })
                    
                    return results
                
                raise ValueError(f"TensorRT output format not compatible. Available keys: {list(outputs.keys())}")
            
        except Exception as e:
            import traceback
            print(f"Error in TRTModel.__call__: {e}")
            print(traceback.format_exc())
            raise
    
    def _execute(self, inputs):
        """Execute TensorRT engine with the given inputs"""
        # Set the dynamic input shapes if needed
        for name, tensor in inputs.items():
            if name in self.input_names:
                # Check if shape has dynamic dimension (-1)
                orig_shape = self.engine.get_tensor_shape(name)
                if any(d == -1 for d in orig_shape):
                    if self.verbose:
                        print(f"Setting input shape for {name}: {tensor.shape}")
                    self.context.set_input_shape(name, tensor.shape)
        
        # Create output buffers - use static shapes from context
        outputs = {}
        for name in self.output_names:
            shape = self.context.get_tensor_shape(name)
            
            # Convert TensorRT Dims to tuple
            shape_tuple = tuple(shape)
            
            # Replace dynamic dims with batch size from inputs
            if -1 in shape_tuple:
                shape_list = list(shape_tuple)
                # Get batch size from any input tensor
                batch_size = next(iter(inputs.values())).shape[0]
                for i, dim in enumerate(shape_list):
                    if dim == -1:
                        shape_list[i] = batch_size
                shape_tuple = tuple(shape_list)
            
            # Create output tensor with proper shape and type
            dtype = self.engine.get_tensor_dtype(name)
            if dtype == trt.DataType.FLOAT:
                torch_dtype = torch.float32
            elif dtype == trt.DataType.INT64 or dtype == trt.DataType.INT32:
                torch_dtype = torch.int64
            else:
                torch_dtype = torch.float32
            
            # Print detailed shape information if verbose
            if self.verbose:
                print(f"Creating output buffer for {name}: shape={shape_tuple}, dtype={torch_dtype}")
                
            # Now use shape_tuple instead of shape directly
            outputs[name] = torch.zeros(shape_tuple, dtype=torch_dtype, device=self.device)
        
        # Create list of input and output bindings in engine order
        bindings = [None] * self.engine.num_io_tensors
        
        # Debug information about tensor binding order
        if self.verbose:
            print("Engine binding order:")
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                mode = self.engine.get_tensor_mode(name)
                print(f"  {i}: {name} ({'input' if mode == trt.TensorIOMode.INPUT else 'output'})")
        
        # Set up input and output bindings
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if name in inputs:
                bindings[i] = inputs[name].data_ptr()
            elif name in outputs:
                bindings[i] = outputs[name].data_ptr()
            else:
                print(f"Warning: Binding {i} ({name}) not found in inputs or outputs")
                # We should never reach here if all inputs/outputs are properly handled
                raise RuntimeError(f"Missing binding for {name}")
        
        if self.verbose:
            print(f"Executing inference with {len(inputs)} inputs and {len(outputs)} outputs")
        
        # Execute inference
        try:
            success = self.context.execute_v2(bindings)
            if not success:
                raise RuntimeError(f"TensorRT execution failed for engine: {self.engine_path}")
        except Exception as e:
            print(f"Error during TensorRT execution: {e}")
            # Print additional diagnostics
            print(f"Bindings array length: {len(bindings)}")
            print(f"Engine expects {self.engine.num_io_tensors} bindings")
            print(f"Input shapes: {[(k, v.shape) for k, v in inputs.items()]}")
            print(f"Output shapes: {[(k, v.shape) for k, v in outputs.items()]}")
            raise
        
        # Convert output types if needed
        for name, tensor in outputs.items():
            # For INT32 outputs that should be INT64
            if tensor.dtype == torch.int32:
                outputs[name] = tensor.to(torch.int64)
                
            # Check if tensor has any NaN values
            if tensor.dtype.is_floating_point and torch.isnan(tensor).any():
                print(f"Warning: NaN values detected in output '{name}'")
                
        return outputs
    
    def measure_latency(self, input_shape=(1, 3, 416, 416), warmup=50, num_iterations=100):
        """
        Measure model inference latency.
        
        Args:
            input_shape: Input shape (batch, channels, height, width)
            warmup: Number of warmup iterations
            num_iterations: Number of timing iterations
            
        Returns:
            Dictionary with latency statistics in milliseconds
        """
        if not torch.cuda.is_available():
            print("CUDA not available, skipping latency measurement")
            return {"error": "CUDA not available"}
        
        print(f"Measuring latency with input shape {input_shape}...")
        
        # Create dummy input
        dummy_input = torch.randn(input_shape, device=self.device)
        
        # Warmup runs
        print(f"Warmup: {warmup} iterations")
        with torch.no_grad():
            for _ in range(warmup):
                _ = self(dummy_input)
        
        # Sync GPU before timing
        torch.cuda.synchronize()
        
        # Timing runs
        print(f"Timing: {num_iterations} iterations")
        latencies = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start_time = time.time()
                _ = self(dummy_input)
                torch.cuda.synchronize()  # Make sure GPU finished work
                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000  # Convert to ms
                latencies.append(latency_ms)
        
        # Calculate statistics
        mean_latency = statistics.mean(latencies)
        median_latency = statistics.median(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        p90_latency = np.percentile(latencies, 90)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        std_dev = statistics.stdev(latencies) if len(latencies) > 1 else 0
        
        # Throughput calculations (images/second)
        throughput = (input_shape[0] * 1000) / mean_latency
        
        latency_stats = {
            "batch_size": input_shape[0],
            "input_resolution": f"{input_shape[2]}x{input_shape[3]}",
            "mean_latency_ms": mean_latency,
            "median_latency_ms": median_latency,
            "min_latency_ms": min_latency,
            "max_latency_ms": max_latency,
            "p90_latency_ms": p90_latency,
            "p95_latency_ms": p95_latency, 
            "p99_latency_ms": p99_latency,
            "std_dev_ms": std_dev,
            "throughput_fps": throughput,
            "samples": len(latencies)
        }
        
        return latency_stats
        
    def eval(self):
        """Set model to evaluation mode (required for compatibility)"""
        return self

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Check if engine directory exists
    if not os.path.exists(args.engine_dir):
        print(f"Error: Engine directory/file not found: {args.engine_dir}")
        return 1

    # Find all engine files
    engine_files = []
    if os.path.isfile(args.engine_dir) and args.engine_dir.endswith('.engine'):
        engine_files = [args.engine_dir]
    else:
        engine_files = glob.glob(os.path.join(args.engine_dir, "*.engine"))
    
    if not engine_files:
        print(f"Error: No engine files found in {args.engine_dir}")
        return 1
        
    print(f"Found {len(engine_files)} engine file(s)")

    # Prepare results dictionary to save to JSON
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "engines": []
    }

    # Loop through each engine file
    for engine_file in engine_files:
        print(f"\n===== Testing engine: {engine_file} =====")
        trt_model = None
        evaluator = None
        
        # Create result dictionary for this engine
        engine_results = {
            "engine_file": engine_file,
            "file_size_mb": os.path.getsize(engine_file) / (1024 * 1024),
            "latency_results": {},
            "evaluation_results": None
        }
        
        try:
            # Print CUDA device information
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                print(f"CUDA device count: {device_count}")
                for i in range(device_count):
                    props = torch.cuda.get_device_properties(i)
                    print(f"  Device {i}: {props.name}, Total Memory: {props.total_memory / 1024**3:.2f} GB")
            
            # Initialize the TensorRT model
            print("\nInitializing TensorRT model...")
            trt_model = TRTModel(engine_file, verbose=args.verbose)
            
            engine_results["engine_info"] = {
                "input_names": trt_model.input_names,
                "output_names": trt_model.output_names
            }
            
            # Run latency benchmark before evaluation
            print("\n===== Latency Benchmark =====")
            
            # Try to determine the input shape from the engine
            input_shape = None
            for name in trt_model.input_names:
                if name == "images":
                    shape = trt_model.engine.get_tensor_shape(name)
                    # If shape has dynamic dimensions, use default values
                    shape = [1 if dim == -1 else dim for dim in shape]
                    input_shape = tuple(shape)
                    break
            
            # If we couldn't determine input shape, use default
            if input_shape is None:
                input_shape = (1, 3, 416, 416)
                print(f"Using default input shape: {input_shape}")
            else:
                print(f"Using input shape from engine: {input_shape}")
            
            engine_results["input_shape"] = input_shape
            
            # Run latency benchmark for different batch sizes
            batch_sizes = [1, 2, 4, 8]
            for batch_size in batch_sizes:
                # Skip larger batches if they exceed the max batch size
                if batch_size > 1 and trt_model.engine is not None:
                    max_batch = trt_model.engine.get_tensor_shape("images")[0]
                    if max_batch != -1 and batch_size > max_batch:
                        print(f"Skipping batch size {batch_size} (exceeds max batch size {max_batch})")
                        continue
                
                # Create input shape with current batch size
                current_shape = (batch_size,) + input_shape[1:]
                
                # Measure latency
                latency_stats = trt_model.measure_latency(
                    input_shape=current_shape,
                    warmup=50,
                    num_iterations=100
                )
                
                # Save to results
                engine_results["latency_results"][f"batch_{batch_size}"] = latency_stats
                
                # Print latency results
                print(f"\nLatency Results (Batch Size = {batch_size}):")
                print(f"  Mean latency: {latency_stats['mean_latency_ms']:.2f} ms")
                print(f"  Median latency: {latency_stats['median_latency_ms']:.2f} ms")
                print(f"  Min latency: {latency_stats['min_latency_ms']:.2f} ms")
                print(f"  Max latency: {latency_stats['max_latency_ms']:.2f} ms")
                print(f"  P90 latency: {latency_stats['p90_latency_ms']:.2f} ms")
                print(f"  P99 latency: {latency_stats['p99_latency_ms']:.2f} ms")
                print(f"  Throughput: {latency_stats['throughput_fps']:.2f} images/second")
            
            # Initialize the evaluation components
            print("\nSetting up evaluator...")
            evaluator = Evaluator(engine_file)
            
            print("Running evaluator setup...")
            evaluator.setup()
            
            # Print evaluator information
            print("Evaluator setup complete")
            print(f"Evaluator components:")
            print(f"  trainer: {type(evaluator.trainer)}")
            print(f"  evaluator: {type(evaluator.evaluator)}")
            print(f"  val_dataloader: {type(evaluator.val_dataloader)}")
            print(f"  postprocessor: {type(evaluator.postprocessor)}")
            if hasattr(evaluator, "criterion"):
                print(f"  criterion: {type(evaluator.criterion)}")
            
            # Important: Assign the TRT model to the evaluator
            print("Setting evaluator.model to TRTModel...")
            evaluator.model = trt_model
            
            # Run evaluation
            print("\nRunning evaluation...")
            eval_stats = evaluator.evaluate()
            
            # Save evaluation results
            engine_results["evaluation_results"] = eval_stats
            
            # Display results
            print(f"\n===== Evaluation Results for {os.path.basename(engine_file)} =====")
            for k, v in eval_stats.items():
                if isinstance(v, list):
                    print(f"{k}:")
                    for i, x in enumerate(v):
                        print(f"  [{i}]: {x}")
                else:
                    print(f"{k}: {v}")
        
        except KeyboardInterrupt:
            print("\nEvaluation interrupted by user")
            
            # Save what we have so far
            engine_results["error"] = "Interrupted by user"
            all_results["engines"].append(engine_results)
            
            # Save results to file before exiting
            save_results_to_file(all_results)
            break
            
        except Exception as e:
            import traceback
            print(f"Error during benchmark: {type(e).__name__}: {e}")
            print(traceback.format_exc())
            
            # Save error information
            engine_results["error"] = {
                "type": type(e).__name__,
                "message": str(e)
            }
        
        finally:
            # Add this engine's results to the overall results
            all_results["engines"].append(engine_results)
            
            # Safe cleanup to avoid CUDA errors
            print("\nCleaning up resources...")
            
            # Clear references in safe order
            if evaluator is not None:
                if hasattr(evaluator, 'model'):
                    evaluator.model = None
                del evaluator
            
            if trt_model is not None:
                if hasattr(trt_model, 'context') and trt_model.context is not None:
                    # Need to explicitly clear TensorRT context and engine
                    # to avoid invalid handle errors
                    trt_model.context = None
                
                if hasattr(trt_model, 'engine') and trt_model.engine is not None:
                    trt_model.engine = None
                    
                if hasattr(trt_model, 'runtime') and trt_model.runtime is not None:
                    trt_model.runtime = None
                    
                del trt_model
                
            # Force CUDA synchronize and cache clear
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"Warning: CUDA cleanup error: {e}")
                    
            print("Cleanup complete")
    
    # Save all results to file
    save_results_to_file(all_results)
    
    return 0

def save_results_to_file(results):
    """Save benchmark results to a JSON file"""
    # Create directory if it doesn't exist
    output_dir = "/home/mohamed/DEIMKit/benchmark_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_results_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Convert non-serializable objects to strings
    results_copy = process_results_for_json(results)
    
    # Write to file
    with open(filepath, 'w') as f:
        json.dump(results_copy, f, indent=2)
    
    print(f"Benchmark results saved to {filepath}")

def process_results_for_json(obj):
    """Process results to make them JSON serializable"""
    if isinstance(obj, dict):
        return {k: process_results_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [process_results_for_json(item) for item in obj]
    elif isinstance(obj, tuple):
        return [process_results_for_json(item) for item in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif torch.is_tensor(obj):
        return obj.cpu().numpy().tolist()
    else:
        return str(obj)

if __name__ == '__main__':
    main()