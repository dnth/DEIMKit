#!/usr/bin/env python3
"""
Benchmark DEIM and YOLO models on real images and videos.
This script provides a simple way to benchmark models on real-world data.
"""

import os
import glob
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

from benchmark import ModelBenchmark
from model_optimizer import ModelOptimizer
from optimize_model import optimize_for_drone

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("BenchmarkImage")

# Try importing visualization tools
try:
    from src.deimkit.visualization import draw_on_image
    DEIMKIT_VIZ_AVAILABLE = True
except ImportError:
    DEIMKIT_VIZ_AVAILABLE = False
    logger.warning("DEIMKit visualization not available")

# Try importing YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("Ultralytics YOLO not available")

def find_images(input_path: str, extensions: List[str] = ['jpg', 'jpeg', 'png']) -> List[str]:
    """Find all images in a directory with given extensions."""
    images = []
    for ext in extensions:
        pattern = os.path.join(input_path, f"*.{ext}")
        images.extend(glob.glob(pattern))
        
        # Also check subdirectories
        pattern = os.path.join(input_path, f"**/*.{ext}")
        images.extend(glob.glob(pattern, recursive=True))
    
    return sorted(list(set(images)))  # Remove duplicates and sort

def preprocess_deim(image: Image.Image, input_size: Tuple[int, int] = (640, 640)) -> torch.Tensor:
    """Preprocess image for DEIM model."""
    transform = T.Compose([
        T.Resize(input_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def postprocess_deim(output, original_image: Image.Image) -> Image.Image:
    """Postprocess output from DEIM model."""
    if not DEIMKIT_VIZ_AVAILABLE:
        logger.warning("DEIMKit visualization not available, skipping visualization")
        return original_image
    
    # Extract boxes, scores, labels from output
    if isinstance(output, tuple) and len(output) >= 3:
        boxes, labels, scores = output[:3]
    elif isinstance(output, dict):
        boxes = output.get('boxes', [])
        labels = output.get('labels', [])
        scores = output.get('scores', [])
    else:
        logger.warning(f"Unexpected output format: {type(output)}")
        return original_image
    
    # Convert to numpy arrays if they're tensors
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    
    # Draw boxes on image
    result_img = draw_on_image(
        image=np.array(original_image),
        boxes=boxes,
        labels=labels,
        scores=scores,
        class_names=None  # Use default class names
    )
    
    return Image.fromarray(result_img)

def benchmark_model_on_images(
    model_path: str,
    image_dir: str,
    model_type: str = "deim",
    model_name: str = "deim_hgnetv2_s",
    model_size: str = "s",
    output_dir: str = "benchmark_results",
    device: str = "auto",
    save_visualization: bool = True,
    optimize_model: bool = False,
    preserve_accuracy: bool = True
) -> Dict[str, Any]:
    """Benchmark model on real images."""
    # Find all images in the directory
    logger.info(f"Looking for images in {image_dir}")
    image_paths = find_images(image_dir)
    
    if not image_paths:
        logger.error(f"No images found in {image_dir}")
        return {}
    
    logger.info(f"Found {len(image_paths)} images")
    
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    
    if model_type.lower() == "deim":
        # Load DEIM model
        optimizer = ModelOptimizer(
            model_path=model_path,
            model_type=model_type,
            model_name=model_name,
            device=device
        )
        model = optimizer.model
        
        # Set up preprocessing and postprocessing
        preprocess_fn = lambda img: preprocess_deim(img)
        postprocess_fn = postprocess_deim
        
    elif model_type.lower() == "yolo" and YOLO_AVAILABLE:
        # Load YOLO model
        model = YOLO(model_path)
        
        # Use YOLO's built-in benchmarking
        results = benchmark_yolo_on_images(
            model=model,
            image_paths=image_paths,
            output_dir=output_dir,
            save_visualization=save_visualization
        )
        return results
    else:
        logger.error(f"Unsupported model type: {model_type}")
        return {}
    
    # Optimize model if requested
    if optimize_model:
        logger.info("Optimizing model for drone deployment")
        
        # Create a temporary path for the optimized model
        optimized_path = optimize_for_drone(
            model_path=model_path,
            model_type=model_type,
            model_name=model_name,
            model_size=model_size,
            output_dir=str(output_dir / "optimized"),
            run_benchmark=False,
            preserve_accuracy=preserve_accuracy
        )
        
        # Load optimized model
        optimizer = ModelOptimizer(
            model_path=optimized_path,
            model_type=model_type,
            model_name=model_name,
            device=device
        )
        optimized_model = optimizer.model
        
        # Initialize benchmark
        benchmark = ModelBenchmark(output_dir=str(output_dir))
        
        # Benchmark original model
        logger.info("Benchmarking original model")
        original_results = benchmark.benchmark_on_images(
            model=model,
            image_paths=image_paths,
            device=device,
            preprocess_fn=preprocess_fn,
            postprocess_fn=postprocess_fn if save_visualization else None,
            save_output=save_visualization,
            output_dir=str(output_dir / "visualizations" / "original")
        )
        
        # Benchmark optimized model
        logger.info("Benchmarking optimized model")
        optimized_results = benchmark.benchmark_on_images(
            model=optimized_model,
            image_paths=image_paths,
            device=device,
            preprocess_fn=preprocess_fn,
            postprocess_fn=postprocess_fn if save_visualization else None,
            save_output=save_visualization,
            output_dir=str(output_dir / "visualizations" / "optimized")
        )
        
        # Compare results
        speedup = original_results["average_latency"] / optimized_results["average_latency"] if optimized_results["average_latency"] > 0 else 0
        logger.info(f"Original model: {original_results['average_latency']:.2f}ms")
        logger.info(f"Optimized model: {optimized_results['average_latency']:.2f}ms")
        logger.info(f"Speedup: {speedup:.2f}x")
        
        # Save comparison results
        comparison = {
            "original": original_results,
            "optimized": optimized_results,
            "speedup": speedup
        }
        
        return comparison
    else:
        # Just benchmark the original model
        benchmark = ModelBenchmark(output_dir=str(output_dir))
        results = benchmark.benchmark_on_images(
            model=model,
            image_paths=image_paths,
            device=device,
            preprocess_fn=preprocess_fn,
            postprocess_fn=postprocess_fn if save_visualization else None,
            save_output=save_visualization,
            output_dir=str(output_dir / "visualizations")
        )
        
        return {"original": results}

def benchmark_yolo_on_images(
    model,
    image_paths: List[str],
    output_dir: str = "benchmark_results",
    save_visualization: bool = True
) -> Dict[str, Any]:
    """Benchmark YOLO model on images using its built-in methods."""
    logger.info(f"Benchmarking YOLO model on {len(image_paths)} images")
    
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Use YOLO's built-in benchmark
    try:
        results = model(
            image_paths,
            verbose=False,
            stream=False,
            save=save_visualization,
            project=str(output_dir),
            name="yolo_results"
        )
        
        # Calculate metrics manually since we can't easily extract them from YOLO
        latencies = []
        for r in results:
            if hasattr(r, 'speed'):
                # YOLO stores speed info in each result
                latencies.append(r.speed['inference'])  # ms
        
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) >= 20 else max_latency
            std_dev = np.std(latencies)
            
            metrics = {
                "average_latency": avg_latency,
                "min_latency": min_latency,
                "max_latency": max_latency,
                "p95_latency": p95_latency,
                "std_dev": std_dev,
                "num_images": len(image_paths),
                "throughput": 1000 / avg_latency if avg_latency > 0 else 0,  # FPS
            }
            
            logger.info(f"YOLO benchmark complete: {avg_latency:.2f}ms average latency")
            return {"original": metrics}
        else:
            logger.warning("No latency information obtained from YOLO")
            return {}
            
    except Exception as e:
        logger.error(f"Error benchmarking YOLO: {e}")
        return {}

def benchmark_on_video(
    model_path: str,
    video_path: str,
    model_type: str = "deim",
    model_name: str = "deim_hgnetv2_s",
    model_size: str = "s",
    output_dir: str = "benchmark_results",
    device: str = "auto",
    save_visualization: bool = True,
    optimize_model: bool = False,
    max_frames: int = 100
) -> Dict[str, Any]:
    """Benchmark model on video."""
    try:
        import cv2
    except ImportError:
        logger.error("OpenCV (cv2) is required for video benchmarking")
        return {}
    
    logger.info(f"Benchmarking on video: {video_path}")
    
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video file: {video_path}")
        return {}
    
    # Extract frames
    frames = []
    frame_count = 0
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        frames.append(frame_pil)
        frame_count += 1
    
    cap.release()
    logger.info(f"Extracted {len(frames)} frames from video")
    
    # Save frames as temporary images for benchmarking
    temp_dir = output_dir / "temp_frames"
    temp_dir.mkdir(exist_ok=True)
    
    image_paths = []
    for i, frame in enumerate(frames):
        path = temp_dir / f"frame_{i:04d}.jpg"
        frame.save(path)
        image_paths.append(str(path))
    
    # Benchmark on the extracted frames
    try:
        results = benchmark_model_on_images(
            model_path=model_path,
            image_dir=str(temp_dir),
            model_type=model_type,
            model_name=model_name,
            model_size=model_size,
            output_dir=output_dir / "video_benchmark",
            device=device,
            save_visualization=save_visualization,
            optimize_model=optimize_model
        )
        
        # Create a video from output frames if visualization was saved
        if save_visualization:
            visualization_dirs = []
            if optimize_model:
                visualization_dirs = [
                    output_dir / "video_benchmark" / "visualizations" / "original",
                    output_dir / "video_benchmark" / "visualizations" / "optimized"
                ]
            else:
                visualization_dirs = [output_dir / "video_benchmark" / "visualizations"]
            
            for viz_dir in visualization_dirs:
                output_frames = sorted(glob.glob(str(viz_dir / "*.jpg")))
                if output_frames:
                    create_output_video(
                        frame_paths=output_frames,
                        output_path=str(viz_dir / "output_video.mp4"),
                        fps=cap.get(cv2.CAP_PROP_FPS)
                    )
        
        return results
    finally:
        # Clean up temporary files
        for path in image_paths:
            try:
                os.remove(path)
            except:
                pass
        try:
            os.rmdir(temp_dir)
        except:
            pass

def create_output_video(frame_paths: List[str], output_path: str, fps: float = 30.0) -> None:
    """Create a video from a sequence of frames."""
    try:
        import cv2
    except ImportError:
        logger.error("OpenCV (cv2) is required for video creation")
        return
    
    if not frame_paths:
        logger.warning("No frames provided for video creation")
        return
    
    # Get first frame to determine dimensions
    first_frame = cv2.imread(frame_paths[0])
    height, width, _ = first_frame.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write frames
    for path in frame_paths:
        frame = cv2.imread(path)
        out.write(frame)
    
    out.release()
    logger.info(f"Created output video: {output_path}")

def main():
    """Command-line interface for the benchmark tool."""
    parser = argparse.ArgumentParser(description="Benchmark models on images and videos")
    
    # Input options
    parser.add_argument("--model-path", required=True, help="Path to the model")
    parser.add_argument("--input", required=True, help="Path to input image directory or video file")
    parser.add_argument("--input-type", choices=["image", "video"], default="image",
                      help="Type of input (image directory or video file)")
    
    # Model options
    parser.add_argument("--model-type", choices=["deim", "yolo"], default="deim",
                      help="Type of model (DEIM or YOLO)")
    parser.add_argument("--model-name", help="Model name (for DEIM models)")
    parser.add_argument("--model-size", choices=["n", "s", "m", "l", "x"], default="s",
                      help="Model size (n=nano, s=small, m=medium, l=large, x=xlarge)")
    
    # Optimization options
    parser.add_argument("--optimize", action="store_true", help="Optimize model before benchmarking")
    parser.add_argument("--preserve-accuracy", action="store_true", default=True,
                      help="Preserve accuracy during optimization (less aggressive)")
    
    # Output options
    parser.add_argument("--output-dir", default="benchmark_results",
                      help="Output directory for benchmark results")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto",
                      help="Device to run inference on")
    parser.add_argument("--no-visualization", action="store_true",
                      help="Disable saving visualizations")
    
    # Video options
    parser.add_argument("--max-frames", type=int, default=100,
                      help="Maximum number of frames to extract from video")
    
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
    
    # Run appropriate benchmark
    if args.input_type == "image":
        results = benchmark_model_on_images(
            model_path=args.model_path,
            image_dir=args.input,
            model_type=args.model_type,
            model_name=args.model_name,
            model_size=args.model_size,
            output_dir=args.output_dir,
            device=args.device,
            save_visualization=not args.no_visualization,
            optimize_model=args.optimize,
            preserve_accuracy=args.preserve_accuracy
        )
    elif args.input_type == "video":
        results = benchmark_on_video(
            model_path=args.model_path,
            video_path=args.input,
            model_type=args.model_type,
            model_name=args.model_name,
            model_size=args.model_size,
            output_dir=args.output_dir,
            device=args.device,
            save_visualization=not args.no_visualization,
            optimize_model=args.optimize,
            max_frames=args.max_frames
        )
    else:
        parser.error("Invalid input type")
    
    # Print summary
    if results:
        print("\nBenchmark Results Summary:")
        if "original" in results and "optimized" in results:
            orig = results["original"]
            opt = results["optimized"]
            speedup = results.get("speedup", 1.0)
            
            print(f"Original model: {orig['average_latency']:.2f}ms ({orig['throughput']:.2f} FPS)")
            print(f"Optimized model: {opt['average_latency']:.2f}ms ({opt['throughput']:.2f} FPS)")
            print(f"Speedup: {speedup:.2f}x")
        elif "original" in results:
            orig = results["original"]
            print(f"Model performance: {orig['average_latency']:.2f}ms ({orig['throughput']:.2f} FPS)")
    
    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main() 