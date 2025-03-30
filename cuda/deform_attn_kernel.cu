#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <vector>
#include <cmath>

template <typename scalar_t>
__device__ scalar_t bilinear_interpolate(
    const scalar_t *input,
    const int height,
    const int width,
    scalar_t h,
    scalar_t w)
{

    if (h <= -1 || height <= h || w <= -1 || width <= w)
    {
        return 0;
    }

    int h_low = floor(h);
    int w_low = floor(w);
    int h_high = h_low + 1;
    int w_high = w_low + 1;

    scalar_t lh = h - h_low;
    scalar_t lw = w - w_low;
    scalar_t hh = 1 - lh;
    scalar_t hw = 1 - lw;

    scalar_t v1 = 0;
    if (h_low >= 0 && w_low >= 0)
        v1 = input[h_low * width + w_low];

    scalar_t v2 = 0;
    if (h_low >= 0 && w_high <= width - 1)
        v2 = input[h_low * width + w_high];

    scalar_t v3 = 0;
    if (h_high <= height - 1 && w_low >= 0)
        v3 = input[h_high * width + w_low];

    scalar_t v4 = 0;
    if (h_high <= height - 1 && w_high <= width - 1)
        v4 = input[h_high * width + w_high];

    scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

    scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
    return val;
}

template <typename scalar_t>
__global__ void deformable_attention_kernel(
    const int n,
    const scalar_t *value,
    const scalar_t *sampling_locations,
    const scalar_t *attention_weights,
    const int batch_size,
    const int spatial_size,
    const int num_heads,
    const int channels,
    const int num_levels,
    const int num_query,
    const int num_point,
    scalar_t *output)
{

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n)
        return;

    // Compute indices
    int data_idx = index;
    int channel_idx = data_idx % channels;
    data_idx /= channels;
    int head_idx = data_idx % num_heads;
    data_idx /= num_heads;
    int query_idx = data_idx % num_query;
    data_idx /= num_query;
    int batch_idx = data_idx;

    // Compute outputs
    scalar_t out_temp = 0;
    int height = sqrt(spatial_size);
    int width = height;

    for (int level_idx = 0; level_idx < num_levels; ++level_idx)
    {
        for (int point_idx = 0; point_idx < num_point; ++point_idx)
        {
            // Get sampling location and attention weight
            int sampling_index = batch_idx * num_query * num_heads * num_levels * num_point * 2 +
                                 query_idx * num_heads * num_levels * num_point * 2 +
                                 head_idx * num_levels * num_point * 2 +
                                 level_idx * num_point * 2 +
                                 point_idx * 2;

            scalar_t loc_w = sampling_locations[sampling_index];
            scalar_t loc_h = sampling_locations[sampling_index + 1];

            // Convert to pixel coordinates
            scalar_t h = loc_h * height - 0.5;
            scalar_t w = loc_w * width - 0.5;

            // Get attention weight
            int attn_index = batch_idx * num_query * num_heads * num_levels * num_point +
                             query_idx * num_heads * num_levels * num_point +
                             head_idx * num_levels * num_point +
                             level_idx * num_point +
                             point_idx;
            scalar_t attn_w = attention_weights[attn_index];

            // Iterate over value feature map
            for (int i = 0; i < spatial_size; ++i)
            {
                int value_h = i / width;
                int value_w = i % width;

                // Compute interpolation weight
                scalar_t weight = attn_w * bilinear_interpolate<scalar_t>(
                                               value + (batch_idx * spatial_size * num_heads * channels +
                                                        i * num_heads * channels +
                                                        head_idx * channels),
                                               height, width, h, w);

                out_temp += weight;
            }
        }
    }

    // Write output
    output[batch_idx * num_query * num_heads * channels +
           query_idx * num_heads * channels +
           head_idx * channels +
           channel_idx] = out_temp;
}

// Forward implementation
torch::Tensor deform_attn_cuda_forward(
    const torch::Tensor &value,
    const torch::Tensor &sampling_locations,
    const torch::Tensor &attention_weights,
    const int im2col_step)
{

    // Check dimensions
    AT_ASSERTM(value.dim() == 4, "value must have dimension 4");
    AT_ASSERTM(sampling_locations.dim() == 6, "sampling_locations must have dimension 6");
    AT_ASSERTM(attention_weights.dim() == 5, "attention_weights must have dimension 5");

    // Get sizes
    const int batch_size = value.size(0);
    const int spatial_size = value.size(1);
    const int num_heads = value.size(2);
    const int channels = value.size(3);

    const int num_query = sampling_locations.size(1);
    const int num_levels = sampling_locations.size(3);
    const int num_point = sampling_locations.size(4);

    // Create output tensor
    auto output = torch::zeros({batch_size, num_query, num_heads, channels}, value.options());

    // Compute elements per thread
    const int total_elements = batch_size * num_query * num_heads * channels;
    const int num_threads = 1024;
    const int num_blocks = (total_elements + num_threads - 1) / num_threads;

    // Launch kernel
    AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "deform_attn_forward_cuda", ([&]
                                                                                 { deformable_attention_kernel<scalar_t><<<num_blocks, num_threads>>>(
                                                                                       total_elements,
                                                                                       value.data_ptr<scalar_t>(),
                                                                                       sampling_locations.data_ptr<scalar_t>(),
                                                                                       attention_weights.data_ptr<scalar_t>(),
                                                                                       batch_size,
                                                                                       spatial_size,
                                                                                       num_heads,
                                                                                       channels,
                                                                                       num_levels,
                                                                                       num_query,
                                                                                       num_point,
                                                                                       output.data_ptr<scalar_t>()); }));

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Error in deform_attn_cuda_forward: %s\n", cudaGetErrorString(err));
    }

    return output;
}

// Backward kernel for value gradients
template <typename scalar_t>
__global__ void deformable_attention_backward_value_kernel(
    const int n,
    const scalar_t *grad_output,
    const scalar_t *sampling_locations,
    const scalar_t *attention_weights,
    const int batch_size,
    const int spatial_size,
    const int num_heads,
    const int channels,
    const int num_levels,
    const int num_query,
    const int num_point,
    scalar_t *grad_value)
{

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n)
        return;

    // Compute indices
    int data_idx = index;
    int channel_idx = data_idx % channels;
    data_idx /= channels;
    int head_idx = data_idx % num_heads;
    data_idx /= num_heads;
    int spatial_idx = data_idx % spatial_size;
    data_idx /= spatial_size;
    int batch_idx = data_idx;

    int spatial_h = spatial_idx / int(sqrt(spatial_size));
    int spatial_w = spatial_idx % int(sqrt(spatial_size));

    // Initialize gradient
    scalar_t grad_value_temp = 0;

    // Height and width of feature map
    int height = sqrt(spatial_size);
    int width = height;

    // Iterate through all query positions
    for (int query_idx = 0; query_idx < num_query; ++query_idx)
    {
        for (int level_idx = 0; level_idx < num_levels; ++level_idx)
        {
            for (int point_idx = 0; point_idx < num_point; ++point_idx)
            {
                // Get sampling location and attention weight
                int sampling_index = batch_idx * num_query * num_heads * num_levels * num_point * 2 +
                                     query_idx * num_heads * num_levels * num_point * 2 +
                                     head_idx * num_levels * num_point * 2 +
                                     level_idx * num_point * 2 +
                                     point_idx * 2;

                scalar_t loc_w = sampling_locations[sampling_index];
                scalar_t loc_h = sampling_locations[sampling_index + 1];

                // Convert to pixel coordinates
                scalar_t h = loc_h * height - 0.5;
                scalar_t w = loc_w * width - 0.5;

                // Check if point contributes to this spatial location
                if (abs(h - spatial_h) <= 1 && abs(w - spatial_w) <= 1)
                {
                    // Get attention weight
                    int attn_index = batch_idx * num_query * num_heads * num_levels * num_point +
                                     query_idx * num_heads * num_levels * num_point +
                                     head_idx * num_levels * num_point +
                                     level_idx * num_point +
                                     point_idx;
                    scalar_t attn_w = attention_weights[attn_index];

                    // Get gradient
                    int grad_idx = batch_idx * num_query * num_heads * channels +
                                   query_idx * num_heads * channels +
                                   head_idx * channels +
                                   channel_idx;
                    scalar_t grad = grad_output[grad_idx];

                    // Compute bilinear weights
                    int h_low = floor(h);
                    int w_low = floor(w);
                    int h_high = h_low + 1;
                    int w_high = w_low + 1;

                    scalar_t lh = h - h_low;
                    scalar_t lw = w - w_low;
                    scalar_t hh = 1 - lh;
                    scalar_t hw = 1 - lw;

                    // Check if this pixel contributes
                    bool contributes = false;
                    scalar_t weight = 0;

                    if (h_low == spatial_h && w_low == spatial_w)
                    {
                        weight = hh * hw;
                        contributes = true;
                    }
                    else if (h_low == spatial_h && w_high == spatial_w)
                    {
                        weight = hh * lw;
                        contributes = true;
                    }
                    else if (h_high == spatial_h && w_low == spatial_w)
                    {
                        weight = lh * hw;
                        contributes = true;
                    }
                    else if (h_high == spatial_h && w_high == spatial_w)
                    {
                        weight = lh * lw;
                        contributes = true;
                    }

                    if (contributes)
                    {
                        grad_value_temp += grad * attn_w * weight;
                    }
                }
            }
        }
    }

    // Write output
    grad_value[batch_idx * spatial_size * num_heads * channels +
               spatial_idx * num_heads * channels +
               head_idx * channels +
               channel_idx] = grad_value_temp;
}

// Backward implementation
std::vector<torch::Tensor> deform_attn_cuda_backward(
    const torch::Tensor &value,
    const torch::Tensor &sampling_locations,
    const torch::Tensor &attention_weights,
    const torch::Tensor &grad_output,
    const int im2col_step)
{

    // Get sizes
    const int batch_size = value.size(0);
    const int spatial_size = value.size(1);
    const int num_heads = value.size(2);
    const int channels = value.size(3);

    const int num_query = sampling_locations.size(1);
    const int num_levels = sampling_locations.size(3);
    const int num_point = sampling_locations.size(4);

    // Create output tensors
    auto grad_value = torch::zeros_like(value);
    auto grad_sampling_loc = torch::zeros_like(sampling_locations);
    auto grad_attn_weight = torch::zeros_like(attention_weights);

    // Compute elements per thread for value gradients
    const int total_value_elements = batch_size * spatial_size * num_heads * channels;
    const int num_threads = 1024;
    const int num_blocks_value = (total_value_elements + num_threads - 1) / num_threads;

    // Launch kernel for value gradients
    AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "deform_attn_backward_value_cuda", ([&]
                                                                                        { deformable_attention_backward_value_kernel<scalar_t><<<num_blocks_value, num_threads>>>(
                                                                                              total_value_elements,
                                                                                              grad_output.data_ptr<scalar_t>(),
                                                                                              sampling_locations.data_ptr<scalar_t>(),
                                                                                              attention_weights.data_ptr<scalar_t>(),
                                                                                              batch_size,
                                                                                              spatial_size,
                                                                                              num_heads,
                                                                                              channels,
                                                                                              num_levels,
                                                                                              num_query,
                                                                                              num_point,
                                                                                              grad_value.data_ptr<scalar_t>()); }));

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Error in deform_attn_cuda_backward: %s\n", cudaGetErrorString(err));
    }

    // Note: For simplicity, we're not implementing gradient for sampling_locations and attention_weights
    // In a real implementation, you would add kernels for these gradients as well

    return {grad_value, grad_sampling_loc, grad_attn_weight};
}