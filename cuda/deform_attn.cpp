#include <torch/extension.h>
#include <ATen/ATen.h>

#include <vector>

// CUDA forward declaration
torch::Tensor deform_attn_cuda_forward(
    const torch::Tensor &value,
    const torch::Tensor &sampling_locations,
    const torch::Tensor &attention_weights,
    const int im2col_step);

// CUDA backward declaration
std::vector<torch::Tensor> deform_attn_cuda_backward(
    const torch::Tensor &value,
    const torch::Tensor &sampling_locations,
    const torch::Tensor &attention_weights,
    const torch::Tensor &grad_output,
    const int im2col_step);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

torch::Tensor deform_attn_forward(
    const torch::Tensor &value,
    const torch::Tensor &sampling_locations,
    const torch::Tensor &attention_weights,
    const int im2col_step)
{

    CHECK_INPUT(value);
    CHECK_INPUT(sampling_locations);
    CHECK_INPUT(attention_weights);

    return deform_attn_cuda_forward(
        value, sampling_locations, attention_weights, im2col_step);
}

std::vector<torch::Tensor> deform_attn_backward(
    const torch::Tensor &value,
    const torch::Tensor &sampling_locations,
    const torch::Tensor &attention_weights,
    const torch::Tensor &grad_output,
    const int im2col_step)
{

    CHECK_INPUT(value);
    CHECK_INPUT(sampling_locations);
    CHECK_INPUT(attention_weights);
    CHECK_INPUT(grad_output);

    return deform_attn_cuda_backward(
        value, sampling_locations, attention_weights, grad_output, im2col_step);
}

// Binding to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("deform_attn_forward", &deform_attn_forward, "Deformable Attention forward (CUDA)",
          py::arg("value"), py::arg("sampling_locations"), py::arg("attention_weights"),
          py::arg("im2col_step") = 64);
    m.def("deform_attn_backward", &deform_attn_backward, "Deformable Attention backward (CUDA)",
          py::arg("value"), py::arg("sampling_locations"), py::arg("attention_weights"),
          py::arg("grad_output"), py::arg("im2col_step") = 64);
}