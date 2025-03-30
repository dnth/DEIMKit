# DEIMKit CUDA-Accelerated Deformable Attention

This directory contains a custom CUDA implementation of deformable attention for accelerating DEIM models. The implementation provides both forward and backward operations for the deformable attention mechanism, with significant performance improvements over the CPU implementation.

## Features

- CUDA-accelerated deformable attention operation
- Bilinear interpolation for sampling
- Support for multiple levels and sampling points
- Automatic fallback to CPU implementation when CUDA is not available
- Easy integration with existing PyTorch models

## Requirements

- CUDA Toolkit (10.0 or later)
- PyTorch (1.7.0 or later)
- C++ compiler compatible with your CUDA version

## Building the Extension

### Option 1: Using PyTorch's setuptools

The easiest way to build the extension is using the provided `setup.py` script:

```bash
# Navigate to the cuda directory
cd cuda

# Build the extension in place
python setup.py build_ext --inplace
```

### Option 2: Using CMake

Alternatively, you can build the extension using CMake:

```bash
# Navigate to the cuda directory
cd cuda

# Create a build directory
mkdir build
cd build

# Configure and build
cmake ..
make

# Install (optional)
make install
```

## Usage

After building the extension, you can use it in your PyTorch code as follows:

```python
import torch
from deformable_attention_cuda import DeformableAttention, optimize_deformable_attention_in_model

# Create a model with deformable attention
model = YourModelWithDeformableAttention()

# Replace standard implementations with optimized CUDA versions
optimized_model = optimize_deformable_attention_in_model(model)

# Or create a standalone deformable attention module
deform_attn = DeformableAttention(dim=256, num_heads=8, n_points=4, n_levels=1)
```

## Automatic Integration

The DEIMKit framework automatically detects when CUDA is available and will use this optimized implementation. If you're using the `ModelOptimizer` class, you can enable deformable attention optimization with:

```python
from model_optimizer import ModelOptimizer

optimizer = ModelOptimizer(model)
optimized_model = optimizer.optimize_model(target="edge", enable_deformable_attn_optimization=True)
```

## Benchmarking

You can benchmark the performance improvement with:

```bash
python benchmark.py --model path/to/model.pth --input path/to/image.jpg --compare-cuda-cpu
```

## Troubleshooting

If you encounter issues:

1. Check if CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
2. Verify your CUDA and PyTorch versions are compatible
3. Check the build logs for errors
4. Try the CPU fallback implementation if CUDA is not available

## License

This code is part of the DEIMKit project and is subject to the same license terms.

## Acknowledgments

This implementation is inspired by:
- MSDeformAttn from Deformable DETR
- CUDA examples from the PyTorch repository 