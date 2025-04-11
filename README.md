## Disclaimer
I'm not affiliated with the original DEIM authors. I just found the model interesting and wanted to try it out. The changes made here are of my own. Please cite and star the original repo if you find this useful.

# Model Optimization Framework

A standalone framework for optimizing PyTorch models (including DEIM and YOLO) for edge deployment.

## Features

- **Aggressive Optimization**: Transform models from larger sizes to nano-equivalent speeds
- **Multiple Optimization Techniques**: Pruning, quantization, ONNX export, TensorRT acceleration
- **Knowledge Distillation**: Transfer knowledge from larger teacher models to smaller student models
- **Comprehensive Benchmarking**: Compare different model variants and optimization techniques
- **YOLOv10 Integration**: Benchmark against YOLOv10 models for comparison
- **Plug-and-Play Design**: Easy-to-use interface for testing different model versions and techniques

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/model-optimization-framework.git
cd model-optimization-framework

# Install dependencies
pip install -r requirements.txt
```

## Dependencies

- PyTorch >= 1.10.0
- torchvision
- numpy
- matplotlib
- onnx (optional, for ONNX export)
- onnxruntime (optional, for ONNX optimization)
- tensorrt (optional, for TensorRT acceleration)
- ultralytics (optional, for YOLOv10 benchmarking)

## Usage

### Quick Start

The framework provides five main optimization targets:

1. **Drone**: Optimize for low-power CPU devices
2. **Benchmark**: Run comprehensive benchmarks comparing multiple models

### Examples

#### Optimize a DEIM model for drone deployment:

```bash
python optimize_model.py --target drone --model-path path/to/model.pth --model-type deim --model-size s
```

#### Run benchmarks comparing DEIM models and YOLOv10:

```bash
python optimize_model.py --target benchmark --model-path path/to/deim/model.pth --model-type deim
```

## Optimization Strategies

### Drone Optimization Strategy

Targets low-power CPU devices with:
- Aggressive pruning (60% in backbone, 40% in encoder, 20% in decoder)
- INT8 dynamic quantization
- Model size reduction focused on inference speed

### Edge Device Optimization Strategy

Balanced approach for edge devices:
- Moderate pruning (30% in backbone, 20% in encoder, 10% in decoder)
- ONNX optimization for better inference
- TensorRT acceleration if GPU is available

## Benchmarking

The benchmarking module allows comprehensive comparison of:

- Different model sizes (nano, small, medium, large, xlarge)
- Various optimization techniques
- Model types (DEIM, YOLO)
- Performance metrics (latency, throughput)

## Extensibility

The framework is designed to be extensible:

- Support for custom PyTorch models
- Customizable optimization parameters
- Add new optimization techniques by extending the `ModelOptimizer` class
- Implement custom benchmarks by extending the `ModelBenchmark` class

## License

[MIT License](LICENSE)

## Acknowledgements

- [DEIMKit](https://github.com/dnth/DEIMKit): Detection with Enhanced Interior Matching
- [Ultralytics YOLOv10](https://github.com/ultralytics/ultralytics): Real-time object detection
- [PyTorch](https://pytorch.org/): The deep learning framework used
- [ONNX](https://onnx.ai/): Open Neural Network Exchange
- [TensorRT](https://developer.nvidia.com/tensorrt): High-performance deep learning inference optimizer

## TensorRT Optimization and Deployment

This project includes extensive model optimization and deployment features targeting real-time performance on various hardware platforms, from NVIDIA GPUs to low-power embedded systems.

### Installation with Poetry

The project now uses Poetry for dependency management and virtual environment creation:

```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Clone the repository
git clone https://github.com/yourusername/model-optimization-framework.git
cd model-optimization-framework

# Install dependencies
poetry install

# Activate the environment
poetry shell
```

### Export to TensorRT

ONNX models can be efficiently converted to TensorRT engines using the provided export scripts:

```bash
# FP16 precision export
python scripts/export_trt.py \
    --onnx-path model.onnx \
    --engine-path model.engine \
    --input-shape 1,3,416,416 \
    --fp16 \
    --workspace-size 4096

# INT8 calibration with dataset
python scripts/export_trt_caliber.py \
    --onnx-path model.onnx \
    --engine-path model.engine \
    --input-shape 1,3,416,416 \
    --int8 \
    --workspace-size 4096
```

### Advanced Optimization Techniques

#### Structured Pruning with Layer Sensitivity Analysis

I've implemented a structured pruning approach that analyzes the sensitivity of each layer to determine optimal pruning ratios:

- Backbone layers: Aggressive pruning (up to 60%)
- Encoder layers: Moderate pruning (up to 40%)
- Decoder layers: Conservative pruning (up to 20%)

This approach preserves model accuracy while significantly reducing computational requirements.

#### Cluster Weights Quantization

For further model compression, the framework includes cluster-based weight quantization techniques that group similar weights together:

- K-means clustering of weight parameters
- Shared weight representations
- Reduced memory footprint while maintaining model expressivity

#### INT8 Calibration with Domain-Specific Data

The calibration process leverages the PlantVillage dataset to ensure accurate INT8 quantization:

- Representative sample selection across all classes
- Per-layer scaling factors
- Tensor-specific calibration for both input tensors

#### CUDA Kernel Optimization

I also explored the potential of CUDA-accelerated deformable attention based on cursor’s work—it’s still early, but initial tests suggest a ~2x theoretical speedup for inference-heavy workloads. I haven’t run full integration yet, but it’s a direction I plan to pursue further

### Benchmarking System

The project includes a comprehensive benchmarking framework for performance analysis:

```bash
python tools/benchmark/custom_trt_benchmark.py --engine_dir /path/to/engines
```

Key features:
- Automated latency measurement across batch sizes
- mAP evaluation on detection datasets
- JSON result export for easy comparison
- Hardware utilization profiling

### Drone Platform Optimization

Specific optimizations for low-power drone platforms include:

- Aggressive model pruning with minimal accuracy loss
- INT8 quantization with drone camera calibration data
- Latency-focused optimization over throughput
- Memory footprint reduction for constrained environments

### Performance Results

The optimized DEIM models demonstrate exceptional performance:

- **DEiT-M model (FP16)**: 101 FPS (batch=1), 431 FPS (batch=8) on NVIDIA L4
- **Accuracy**: 0.52+ mAP on PlantDoc dataset
- **Stability**: Consistent latency across runs

Both "m" and "x" variants have been tested with FP16 precision. The "x" variant, while theoretically more powerful, showed signs of overfitting in certain configurations. The export process preserves the model's detection capabilities with standardized output signatures.

The optimized models are suitable for deployment in real-time applications, with leading results on the Roboflow leaderboard for real-time object detection.

### Technical Environment

The optimization framework was developed and tested with:
- TensorRT 10.x
- CUDA 11.8
- NVIDIA L4 GPU
- Poetry 1.6+ for dependency management

Comprehensive performance analysis data is available in the `benchmark_results` directory.
