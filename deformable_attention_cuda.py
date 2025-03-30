"""
CUDA implementation for deformable attention operations.
This module provides CUDA kernels and Python bindings for efficient deformable attention.
"""

import os
import math
from typing import Tuple, List

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.utils.cpp_extension import load

# Find the current directory to locate C++/CUDA source files
current_dir = os.path.dirname(os.path.abspath(__file__))

# Try to load CUDA extension, with graceful fallback to CPU implementation
try:
    # JIT compile the CUDA extension
    deform_attn_cuda = load(
        name="deform_attn_cuda",
        sources=[
            os.path.join(current_dir, "cuda/deform_attn.cpp"),
            os.path.join(current_dir, "cuda/deform_attn_kernel.cu"),
        ],
        verbose=True
    )
    
    CUDA_AVAILABLE = True
    
except Exception as e:
    print(f"Warning: Failed to load CUDA extension: {e}")
    print("Falling back to CPU implementation")
    CUDA_AVAILABLE = False

# CUDA Function
class DeformableAttentionFunction(Function):
    """
    CUDA implementation of deformable attention forward and backward functions.
    If CUDA is not available, falls back to CPU implementation.
    """
    
    @staticmethod
    def forward(ctx, value, sampling_locations, attention_weights, im2col_step=64):
        """Forward pass for deformable attention."""
        
        if not value.is_cuda and CUDA_AVAILABLE:
            print("Warning: Input is not on CUDA device but CUDA implementation is available")
        
        if value.is_cuda and CUDA_AVAILABLE:
            # Use CUDA implementation if available and inputs are on CUDA device
            output = deform_attn_cuda.deform_attn_forward(
                value, sampling_locations, attention_weights, im2col_step
            )
            
            # Save for backward
            ctx.save_for_backward(value, sampling_locations, attention_weights)
            ctx.im2col_step = im2col_step
            
            return output
        else:
            # Fallback to CPU implementation
            return cpu_forward_deformable_attention(
                value, sampling_locations, attention_weights
            )
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass for deformable attention."""
        
        if grad_output.is_cuda and CUDA_AVAILABLE:
            # Use CUDA implementation
            value, sampling_locations, attention_weights = ctx.saved_tensors
            im2col_step = ctx.im2col_step
            
            grad_value, grad_sampling_loc, grad_attn_weight = deform_attn_cuda.deform_attn_backward(
                value, sampling_locations, attention_weights, grad_output, im2col_step
            )
            
            return grad_value, grad_sampling_loc, grad_attn_weight, None
        else:
            # Fallback to CPU implementation
            value, sampling_locations, attention_weights = ctx.saved_tensors
            grad_value, grad_sampling_loc, grad_attn_weight = cpu_backward_deformable_attention(
                value, sampling_locations, attention_weights, grad_output
            )
            
            return grad_value, grad_sampling_loc, grad_attn_weight, None

def deformable_attention(value, sampling_locations, attention_weights):
    """
    Deformable attention operation.
    
    Args:
        value (Tensor): The value tensor [N, H*W, num_heads, C/num_heads].
        sampling_locations (Tensor): The sampling locations [N, Lq, num_heads, num_levels, num_points, 2].
        attention_weights (Tensor): The attention weights [N, Lq, num_heads, num_levels, num_points].
    
    Returns:
        Tensor: Output tensor.
    """
    return DeformableAttentionFunction.apply(value, sampling_locations, attention_weights)

def cpu_forward_deformable_attention(value, sampling_locations, attention_weights):
    """
    CPU implementation of deformable attention forward pass.
    This is slower but serves as a fallback.
    
    Args:
        value (Tensor): The value tensor [N, H*W, num_heads, C/num_heads].
        sampling_locations (Tensor): The sampling locations [N, Lq, num_heads, num_levels, num_points, 2].
        attention_weights (Tensor): The attention weights [N, Lq, num_heads, num_levels, num_points].
    
    Returns:
        Tensor: Output tensor.
    """
    # Get shapes
    N, S, num_heads, C = value.shape
    _, Lq, _, num_levels, num_points, _ = sampling_locations.shape
    
    # Initialize output
    output = torch.zeros(N, Lq, num_heads, C, device=value.device, dtype=value.dtype)
    
    # CPU implementation (slow but functional)
    for b in range(N):  # batch dim
        for q in range(Lq):  # query dim
            for h in range(num_heads):  # head dim
                for l in range(num_levels):  # level dim
                    for p in range(num_points):  # point dim
                        # Get sampling location
                        loc_w, loc_h = sampling_locations[b, q, h, l, p, 0], sampling_locations[b, q, h, l, p, 1]
                        
                        # Convert to grid coordinates
                        h_im = int(math.floor(loc_h))
                        w_im = int(math.floor(loc_w))
                        
                        # Ensure within bounds
                        if h_im >= 0 and w_im >= 0 and h_im < int(math.sqrt(S)) and w_im < int(math.sqrt(S)):
                            s_idx = h_im * int(math.sqrt(S)) + w_im
                            weight = attention_weights[b, q, h, l, p]
                            
                            # Add weighted contribution
                            output[b, q, h] += weight * value[b, s_idx, h]
    
    return output

def cpu_backward_deformable_attention(value, sampling_locations, attention_weights, grad_output):
    """
    CPU implementation of deformable attention backward pass.
    This is a simplified version that only computes approximate gradients.
    
    Args:
        value (Tensor): The value tensor.
        sampling_locations (Tensor): The sampling locations.
        attention_weights (Tensor): The attention weights.
        grad_output (Tensor): Gradient from output.
    
    Returns:
        Tuple[Tensor]: Gradients for value, sampling_locations, and attention_weights.
    """
    # Initialize gradients
    grad_value = torch.zeros_like(value)
    grad_sampling_loc = torch.zeros_like(sampling_locations)
    grad_attn_weight = torch.zeros_like(attention_weights)
    
    # Get shapes
    N, S, num_heads, C = value.shape
    _, Lq, _, num_levels, num_points, _ = sampling_locations.shape
    
    # Simple approximation - distribute gradient based on weights
    for b in range(N):
        for q in range(Lq):
            for h in range(num_heads):
                for l in range(num_levels):
                    for p in range(num_points):
                        # Get sampling location
                        loc_w, loc_h = sampling_locations[b, q, h, l, p, 0], sampling_locations[b, q, h, l, p, 1]
                        
                        # Convert to grid coordinates
                        h_im = int(math.floor(loc_h))
                        w_im = int(math.floor(loc_w))
                        
                        # Ensure within bounds
                        if h_im >= 0 and w_im >= 0 and h_im < int(math.sqrt(S)) and w_im < int(math.sqrt(S)):
                            s_idx = h_im * int(math.sqrt(S)) + w_im
                            weight = attention_weights[b, q, h, l, p]
                            
                            # Gradients
                            grad_value[b, s_idx, h] += weight * grad_output[b, q, h]
                            grad_attn_weight[b, q, h, l, p] = torch.sum(value[b, s_idx, h] * grad_output[b, q, h])
    
    return grad_value, grad_sampling_loc, grad_attn_weight

class DeformableAttention(nn.Module):
    """
    Deformable attention module that can be used as a drop-in replacement.
    
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        num_levels (int): Number of feature levels.
        num_points (int): Number of sampling points per level.
    """
    
    def __init__(self, dim, num_heads=8, num_levels=4, num_points=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        
        self.sampling_offsets = nn.Linear(dim, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(dim, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(dim, dim)
        self.output_proj = nn.Linear(dim, dim)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize weights and biases."""
        # Initialize sampling offsets
        nn.init.constant_(self.sampling_offsets.weight, 0.)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init.view(self.num_heads, 1, 1, 2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        
        # Initialize attention weights
        nn.init.constant_(self.attention_weights.weight, 0.)
        nn.init.constant_(self.attention_weights.bias, 0.)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.)
        nn.init.constant_(self.output_proj.bias, 0.)
    
    def forward(self, query, value, query_pos=None, reference_points=None, spatial_shapes=None):
        """
        Forward function for deformable attention.
        
        Args:
            query (Tensor): Query tensor [N, L, C].
            value (Tensor): Value tensor [N, S, C].
            query_pos (Tensor, optional): Query position encoding.
            reference_points (Tensor): Reference points [N, L, num_levels, 2].
            spatial_shapes (Tensor): Spatial shapes of value features [num_levels, 2].
            
        Returns:
            Tensor: Output tensor [N, L, C].
        """
        N, Lq, _ = query.shape
        N, S, _ = value.shape
        
        # Project value
        value = self.value_proj(value).view(N, S, self.num_heads, self.dim // self.num_heads)
        
        # Add position encoding to query if provided
        if query_pos is not None:
            query = query + query_pos
        
        # Compute sampling locations
        sampling_offsets = self.sampling_offsets(query).view(
            N, Lq, self.num_heads, self.num_levels, self.num_points, 2
        )
        
        # Compute attention weights
        attention_weights = self.attention_weights(query).view(
            N, Lq, self.num_heads, self.num_levels * self.num_points
        ).softmax(-1).view(N, Lq, self.num_heads, self.num_levels, self.num_points)
        
        # If reference points not provided, use center of the image
        if reference_points is None:
            H = W = int(math.sqrt(S))
            device = query.device
            reference_points = torch.zeros(N, Lq, self.num_levels, 2, device=device)
            for i in range(self.num_levels):
                reference_points[:, :, i, 0] = W // 2
                reference_points[:, :, i, 1] = H // 2
        
        # Compute absolute sampling locations
        sampling_locations = reference_points.unsqueeze(2).unsqueeze(4) + sampling_offsets
        
        # Apply deformable attention
        output = deformable_attention(value, sampling_locations, attention_weights)
        
        # Project output
        output = self.output_proj(output.view(N, Lq, self.dim))
        
        return output

def optimize_deformable_attention_in_model(model: nn.Module) -> nn.Module:
    """
    Replace standard deformable attention implementations with our optimized CUDA version.
    
    Args:
        model: The PyTorch model to optimize
        
    Returns:
        Optimized model
    """
    import re
    
    # Check if CUDA is available
    if not CUDA_AVAILABLE:
        print("CUDA extension not available, skipping optimization")
        return model
    
    # Track replacements
    replacements = 0
    
    # Find and replace deformable attention modules
    for name, module in model.named_modules():
        # Look for modules that might be deformable attention
        if (isinstance(module, nn.Module) and 
            ('deform' in name.lower() or 
             'deformable' in name.lower() or 
             'attention' in name.lower())):
            
            # Check if it has the typical attributes of deformable attention
            has_sampling = hasattr(module, 'sampling_offsets')
            has_attention = hasattr(module, 'attention_weights')
            
            if has_sampling and has_attention:
                # Get parameters
                dim = getattr(module, 'dim', module.sampling_offsets.weight.size(0))
                num_heads = getattr(module, 'num_heads', 8)
                num_levels = getattr(module, 'num_levels', 4)
                num_points = getattr(module, 'num_points', 4)
                
                # Create replacement
                optimized_module = DeformableAttention(
                    dim=dim,
                    num_heads=num_heads,
                    num_levels=num_levels,
                    num_points=num_points
                )
                
                # Copy weights if possible
                if hasattr(module, 'sampling_offsets'):
                    optimized_module.sampling_offsets.weight.data.copy_(module.sampling_offsets.weight.data)
                    optimized_module.sampling_offsets.bias.data.copy_(module.sampling_offsets.bias.data)
                    
                if hasattr(module, 'attention_weights'):
                    optimized_module.attention_weights.weight.data.copy_(module.attention_weights.weight.data)
                    optimized_module.attention_weights.bias.data.copy_(module.attention_weights.bias.data)
                    
                if hasattr(module, 'value_proj'):
                    optimized_module.value_proj.weight.data.copy_(module.value_proj.weight.data)
                    optimized_module.value_proj.bias.data.copy_(module.value_proj.bias.data)
                    
                if hasattr(module, 'output_proj'):
                    optimized_module.output_proj.weight.data.copy_(module.output_proj.weight.data)
                    optimized_module.output_proj.bias.data.copy_(module.output_proj.bias.data)
                
                # Replace module
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                
                if parent_name:
                    parent = model
                    for part in parent_name.split('.'):
                        parent = getattr(parent, part)
                    setattr(parent, child_name, optimized_module)
                else:
                    setattr(model, child_name, optimized_module)
                
                replacements += 1
                
    print(f"Replaced {replacements} deformable attention modules with CUDA-optimized versions")
    return model 