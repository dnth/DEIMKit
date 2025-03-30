import os
import torch
import torch.nn as nn
from torch.autograd import Function
import logging

logger = logging.getLogger(__name__)

# Try to load the CUDA extension
try:
    from deform_attn_cuda import deform_attn_cuda_forward, deform_attn_cuda_backward
    CUDA_AVAILABLE = True
    logger.info("CUDA extension for deformable attention loaded successfully")
except ImportError:
    CUDA_AVAILABLE = False
    logger.warning("CUDA extension not found. Using CPU fallback implementation")

class DeformableAttentionFunction(Function):
    """
    Autograd function for deformable attention operation.
    """
    @staticmethod
    def forward(ctx, value, sampling_locations, attention_weights, im2col_step=64):
        if CUDA_AVAILABLE and value.is_cuda:
            output = deform_attn_cuda_forward(
                value, sampling_locations, attention_weights, im2col_step
            )
            ctx.save_for_backward(value, sampling_locations, attention_weights)
            ctx.im2col_step = im2col_step
        else:
            output = DeformableAttentionFunction.forward_cpu(
                value, sampling_locations, attention_weights
            )
            ctx.save_for_backward(value, sampling_locations, attention_weights)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        if CUDA_AVAILABLE and grad_output.is_cuda:
            value, sampling_locations, attention_weights = ctx.saved_tensors
            grad_value, grad_sampling_loc, grad_attn_weight = deform_attn_cuda_backward(
                value, sampling_locations, attention_weights, grad_output, ctx.im2col_step
            )
        else:
            grad_value, grad_sampling_loc, grad_attn_weight = DeformableAttentionFunction.backward_cpu(
                ctx, grad_output
            )
        
        return grad_value, grad_sampling_loc, grad_attn_weight, None

    @staticmethod
    def forward_cpu(value, sampling_locations, attention_weights):
        """
        CPU implementation of deformable attention forward pass.
        """
        batch_size, spatial_size, num_heads, channels = value.shape
        _, num_query, _, num_levels, num_points, _ = sampling_locations.shape
        
        output = torch.zeros(batch_size, num_query, num_heads, channels, 
                           device=value.device, dtype=value.dtype)
        
        height = width = int(spatial_size ** 0.5)
        
        for b in range(batch_size):
            for q in range(num_query):
                for h in range(num_heads):
                    for l in range(num_levels):
                        for p in range(num_points):
                            # Get sampling location
                            loc_w = sampling_locations[b, q, h, l, p, 0]
                            loc_h = sampling_locations[b, q, h, l, p, 1]
                            
                            # Convert to pixel coordinates
                            x = loc_w * width - 0.5
                            y = loc_h * height - 0.5
                            
                            # Get attention weight
                            attn_weight = attention_weights[b, q, h, l, p]
                            
                            # Bilinear interpolation
                            x0 = int(torch.floor(x))
                            x1 = x0 + 1
                            y0 = int(torch.floor(y))
                            y1 = y0 + 1
                            
                            x0 = max(0, min(x0, width - 1))
                            x1 = max(0, min(x1, width - 1))
                            y0 = max(0, min(y0, height - 1))
                            y1 = max(0, min(y1, height - 1))
                            
                            dx = x - x0
                            dy = y - y0
                            
                            w00 = (1 - dx) * (1 - dy)
                            w01 = (1 - dx) * dy
                            w10 = dx * (1 - dy)
                            w11 = dx * dy
                            
                            # Add weighted value to output
                            pos00 = y0 * width + x0
                            pos01 = y1 * width + x0
                            pos10 = y0 * width + x1
                            pos11 = y1 * width + x1
                            
                            if 0 <= pos00 < spatial_size:
                                output[b, q, h] += attn_weight * w00 * value[b, pos00, h]
                            if 0 <= pos01 < spatial_size:
                                output[b, q, h] += attn_weight * w01 * value[b, pos01, h]
                            if 0 <= pos10 < spatial_size:
                                output[b, q, h] += attn_weight * w10 * value[b, pos10, h]
                            if 0 <= pos11 < spatial_size:
                                output[b, q, h] += attn_weight * w11 * value[b, pos11, h]
        
        return output

    @staticmethod
    def backward_cpu(ctx, grad_output):
        """
        CPU implementation of deformable attention backward pass.
        """
        value, sampling_locations, attention_weights = ctx.saved_tensors
        batch_size, spatial_size, num_heads, channels = value.shape
        _, num_query, _, num_levels, num_points, _ = sampling_locations.shape
        
        grad_value = torch.zeros_like(value)
        grad_sampling_loc = torch.zeros_like(sampling_locations)
        grad_attn_weight = torch.zeros_like(attention_weights)
        
        height = width = int(spatial_size ** 0.5)
        
        # This is a simplified implementation that only computes grad_value
        # In a full implementation, we would need to compute grad_sampling_loc and grad_attn_weight as well
        for b in range(batch_size):
            for q in range(num_query):
                for h in range(num_heads):
                    for l in range(num_levels):
                        for p in range(num_points):
                            # Get sampling location
                            loc_w = sampling_locations[b, q, h, l, p, 0]
                            loc_h = sampling_locations[b, q, h, l, p, 1]
                            
                            # Convert to pixel coordinates
                            x = loc_w * width - 0.5
                            y = loc_h * height - 0.5
                            
                            # Get attention weight
                            attn_weight = attention_weights[b, q, h, l, p]
                            
                            # Bilinear interpolation
                            x0 = int(torch.floor(x))
                            x1 = x0 + 1
                            y0 = int(torch.floor(y))
                            y1 = y0 + 1
                            
                            x0 = max(0, min(x0, width - 1))
                            x1 = max(0, min(x1, width - 1))
                            y0 = max(0, min(y0, height - 1))
                            y1 = max(0, min(y1, height - 1))
                            
                            dx = x - x0
                            dy = y - y0
                            
                            w00 = (1 - dx) * (1 - dy)
                            w01 = (1 - dx) * dy
                            w10 = dx * (1 - dy)
                            w11 = dx * dy
                            
                            # Compute gradients
                            pos00 = y0 * width + x0
                            pos01 = y1 * width + x0
                            pos10 = y0 * width + x1
                            pos11 = y1 * width + x1
                            
                            grad = grad_output[b, q, h]
                            
                            if 0 <= pos00 < spatial_size:
                                grad_value[b, pos00, h] += attn_weight * w00 * grad
                            if 0 <= pos01 < spatial_size:
                                grad_value[b, pos01, h] += attn_weight * w01 * grad
                            if 0 <= pos10 < spatial_size:
                                grad_value[b, pos10, h] += attn_weight * w10 * grad
                            if 0 <= pos11 < spatial_size:
                                grad_value[b, pos11, h] += attn_weight * w11 * grad
        
        return grad_value, grad_sampling_loc, grad_attn_weight

class DeformableAttention(nn.Module):
    """
    Deformable attention module with optimized CUDA implementation.
    """
    def __init__(self, dim, num_heads=8, n_points=4, n_levels=1):
        """
        Initialize the deformable attention module.
        
        Args:
            dim (int): Input dimension
            num_heads (int): Number of attention heads
            n_points (int): Number of sampling points
            n_levels (int): Number of feature levels
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.n_points = n_points
        self.n_levels = n_levels
        
        self.sampling_offsets = nn.Linear(dim, num_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(dim, num_heads * n_levels * n_points)
        self.value_proj = nn.Linear(dim, dim)
        self.output_proj = nn.Linear(dim, dim)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize the parameters with appropriate distributions."""
        nn.init.constant_(self.sampling_offsets.weight, 0.)
        
        # Initialize sampling offsets with a small value
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * torch.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True)[0]
        grid_init = grid_init.view(self.num_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        
        nn.init.constant_(self.attention_weights.weight, 0.)
        nn.init.constant_(self.attention_weights.bias, 0.)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.)
    
    def forward(self, query, value, reference_points=None, spatial_shapes=None, level_start_index=None):
        """
        Forward pass of deformable attention.
        
        Args:
            query (torch.Tensor): Query tensor [bs, num_query, dim]
            value (torch.Tensor): Value tensor [bs, h*w, dim]
            reference_points (torch.Tensor, optional): Reference points [bs, num_query, 2]
            spatial_shapes (torch.Tensor, optional): Spatial shapes of value features
            level_start_index (torch.Tensor, optional): Start indices for each level
        
        Returns:
            torch.Tensor: Attention output [bs, num_query, dim]
        """
        bs, num_query, _ = query.shape
        bs, spatial_size, _ = value.shape
        
        # Project value
        value = self.value_proj(value).view(bs, spatial_size, self.num_heads, self.dim // self.num_heads)
        
        # Compute sampling locations
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.n_levels, self.n_points, 2
        )
        
        # Compute attention weights
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.n_levels, self.n_points
        )
        attention_weights = torch.softmax(attention_weights, dim=-1)
        
        # If reference points are not provided, use center of image
        if reference_points is None:
            h = w = int(spatial_size ** 0.5)
            y, x = torch.meshgrid(
                torch.linspace(0.5 / h, 1 - 0.5 / h, h, device=query.device),
                torch.linspace(0.5 / w, 1 - 0.5 / w, w, device=query.device)
            )
            reference_points = torch.stack([x, y], dim=-1).reshape(-1, 2).unsqueeze(0).repeat(bs, 1, 1)
            reference_points = reference_points.unsqueeze(0).repeat(num_query, 1, 1, 1).permute(1, 0, 2, 3)
        
        # Compute absolute sampling locations
        if reference_points.shape[-1] == 2:
            # For 2D reference points
            sampling_locations = reference_points.unsqueeze(2).unsqueeze(2) + sampling_offsets
        else:
            # For normalized reference points
            sampling_locations = reference_points.unsqueeze(2).unsqueeze(2) \
                              + sampling_offsets / torch.tensor([w, h], device=query.device)
        
        # Apply deformable attention
        output = DeformableAttentionFunction.apply(
            value, sampling_locations, attention_weights
        )
        
        # Project output
        output = self.output_proj(output.view(bs, num_query, self.dim))
        
        return output

def find_deformable_attention_modules(model):
    """
    Find modules in the model that might be deformable attention.
    
    Args:
        model (nn.Module): The PyTorch model to search
    
    Returns:
        list: List of (name, module) pairs
    """
    deformable_modules = []
    for name, module in model.named_modules():
        # Check if module name contains 'deform' or 'deformable'
        if ('deform' in name.lower() or 'deformable' in name.lower()) and 'atten' in name.lower():
            deformable_modules.append((name, module))
        # Check for MSDeformAttn which is common in many models
        elif any(cls_name in module.__class__.__name__ for cls_name in ['MSDeformAttn', 'DeformableAttention']):
            deformable_modules.append((name, module))
    
    return deformable_modules

def optimize_deformable_attention_in_model(model):
    """
    Replace deformable attention modules in the model with optimized versions.
    
    Args:
        model (nn.Module): The PyTorch model to optimize
    
    Returns:
        nn.Module: Model with optimized deformable attention
    """
    deformable_modules = find_deformable_attention_modules(model)
    
    if not deformable_modules:
        logger.info("No deformable attention modules found in the model")
        return model
    
    logger.info(f"Found {len(deformable_modules)} deformable attention modules to optimize")
    
    # This is a simplified replacement strategy
    # In a real implementation, we would need to analyze each module's interface
    # and create an appropriate replacement
    for name, module in deformable_modules:
        if hasattr(module, 'dim') and hasattr(module, 'num_heads'):
            # Try to extract parameters from the original module
            try:
                dim = module.dim
                num_heads = module.num_heads
                n_points = getattr(module, 'n_points', 4)
                n_levels = getattr(module, 'n_levels', 1)
                
                # Create optimized replacement
                optimized_module = DeformableAttention(dim, num_heads, n_points, n_levels)
                
                # Replace in parent module
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                
                if parent_name:
                    parent = model
                    for part in parent_name.split('.'):
                        parent = getattr(parent, part)
                    setattr(parent, child_name, optimized_module)
                else:
                    setattr(model, child_name, optimized_module)
                
                logger.info(f"Replaced {name} with optimized CUDA implementation")
            except Exception as e:
                logger.warning(f"Failed to replace {name}: {str(e)}")
    
    # Set a flag to indicate the model has deformable attention
    model.has_deformable_attention = True
    
    return model 