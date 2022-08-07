#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> fused_bilinear2x2_cuda_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor grids);

std::vector<torch::Tensor> fused_bilinear2x2_cuda_backward(
    torch::Tensor input_buffer,
    torch::Tensor weights,
    torch::Tensor grids,
    torch::Tensor grad_output);

// std::vector<torch::Tensor> fused_bilinear2x2_cuda_forward_v1(
//     torch::Tensor input,
//     torch::Tensor weights,
//     torch::Tensor grids);

// std::vector<torch::Tensor> fused_bilinear2x2_cuda_backward_v1(
//     torch::Tensor input_buffer,
//     torch::Tensor weights,
//     torch::Tensor grids,
//     torch::Tensor grad_output);

// std::vector<torch::Tensor> fused_bilinear2x2_cuda_forward_v2(
//     torch::Tensor input,
//     torch::Tensor weights,
//     torch::Tensor grids);

// std::vector<torch::Tensor> fused_bilinear2x2_cuda_backward_v2(
//     torch::Tensor input_buffer,
//     torch::Tensor weights,
//     torch::Tensor grids,
//     torch::Tensor grad_output);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


std::vector<torch::Tensor> bilinear2x2_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor grids){
  CHECK_INPUT(input);
  CHECK_INPUT(weights);
  CHECK_INPUT(grids);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  
  return fused_bilinear2x2_cuda_forward(input, weights, grids);
  }

std::vector<torch::Tensor> bilinear2x2_backward(
    torch::Tensor input_buffer,
    torch::Tensor weights,
    torch::Tensor grids,
    torch::Tensor grad_output){
  CHECK_INPUT(input_buffer);
  CHECK_INPUT(weights);
  CHECK_INPUT(grids);
  CHECK_INPUT(grad_output);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input_buffer));
  

  return fused_bilinear2x2_cuda_backward(input_buffer, weights, grids, grad_output);
}

// std::vector<torch::Tensor> bilinear2x2_forward_v1(
//     torch::Tensor input,
//     torch::Tensor weights,
//     torch::Tensor grids){
//   CHECK_INPUT(input);
//   CHECK_INPUT(weights);
//   CHECK_INPUT(grids);
//   return fused_bilinear2x2_cuda_forward_v1(input, weights, grids);
//   }

// std::vector<torch::Tensor> bilinear2x2_backward_v1(
//     torch::Tensor input_buffer,
//     torch::Tensor weights,
//     torch::Tensor grids,
//     torch::Tensor grad_output){
//   CHECK_INPUT(input_buffer);
//   CHECK_INPUT(weights);
//   CHECK_INPUT(grids);
//   CHECK_INPUT(grad_output);

//   return fused_bilinear2x2_cuda_backward_v1(input_buffer, weights, grids, grad_output);
// }


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bilinear2x2_forward", &bilinear2x2_forward, "Bi-Linear 2x2 forward (CUDA)");
  m.def("bilinear2x2_backward", &bilinear2x2_backward, "Bi-Linear 2x2 backward (CUDA)");
  // m.def("bilinear2x2_forward_v1", &bilinear2x2_forward_v1, "Bi-Linear 2x2 forward (CUDA)");
  // m.def("bilinear2x2_backward_v1", &bilinear2x2_backward_v1, "Bi-Linear 2x2 backward (CUDA)");
  // m.def("bilinear2x2_forward_v2", &bilinear2x2_forward_v2, "Bi-Linear 2x2 forward (CUDA)");
  // m.def("bilinear2x2_backward_v2", &bilinear2x2_backward_v2, "Bi-Linear 2x2 backward (CUDA)");
}