from src.deimkit.config import Config
from src.deimkit.custom_exporter import Exporter

# Initialize with your config
config = Config('path/to/config.yml')
exporter = Exporter(config)

# Export to ONNX with GELU->SiLU replacement and deformable attention optimization
exporter.to_onnx(
    checkpoint_path='path/to/model.pth',
    output_path='optimized_model.onnx',
    input_shape=(1, 3, 416, 416),  # Use small batch size to avoid OOM
    replace_gelu=True,  # Replace GELU with SiLU
    optimize_deformable_attn=True,  # Use optimized deformable attention
    deformable_attn_plugin_path='/path/to/libdeformable_attn.so',  # Your CUDA plugin
    opset_version=13  # Lower opset version for better compatibility
)

# Convert to TensorRT engine
exporter.to_tensorrt(
    onnx_path='optimized_model.onnx',
    output_path='optimized_model.engine',
    precision='fp16',  # Use FP16 for faster inference
    max_batch_size=16  # Set maximum batch size for your use case
)