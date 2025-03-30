from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Check if CUDA is available
cuda_available = os.system('nvidia-smi > /dev/null 2>&1') == 0

# Define the extension
if cuda_available:
    ext_modules = [
        CUDAExtension(
            name='deform_attn_cuda',
            sources=[
                'deform_attn.cpp',
                'deform_attn_kernel.cu',
            ],
            extra_compile_args={
                'cxx': ['-g'],
                'nvcc': ['-O2']
            }
        )
    ]
else:
    print("CUDA not available. Building CPU-only version.")
    ext_modules = []

setup(
    name='deformable_attention',
    version='0.1.0',
    author='DEIMKit Team',
    author_email='info@deimkit.org',
    description='CUDA implementation of deformable attention for DEIMKit',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension}
) 