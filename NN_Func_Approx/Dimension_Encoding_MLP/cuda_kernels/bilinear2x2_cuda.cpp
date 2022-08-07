#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> bilinear2x2_cuda_forward(
    torch::Tensor input,
    torch::Tensor weights);

std::vector<torch::Tensor> bilinear2x2_cuda_backward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor grad_output);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


std::vector<torch::Tensor> bilinear2x2_forward(
    torch::Tensor input,
    torch::Tensor weights){
  CHECK_INPUT(input);
  CHECK_INPUT(weights);
  return bilinear2x2_cuda_forward(input, weights);
  }

std::vector<torch::Tensor> bilinear2x2_backward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor grad_output){
  CHECK_INPUT(input);
  CHECK_INPUT(weights);
  CHECK_INPUT(grad_output);

  return bilinear2x2_cuda_backward(input, weights, grad_output);
}


//////////////////////////////////////////////////
//////////////////////////////////////////////////
//////////////////////////////////////////////////
//////////////////////////////////////////////////


std::vector<torch::Tensor> bilinear2x1_cuda_forward(
    torch::Tensor input,
    torch::Tensor weights);

std::vector<torch::Tensor> bilinear2x1_cuda_backward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor grad_output);


    std::vector<torch::Tensor> bilinear2x1_forward(
    torch::Tensor input,
    torch::Tensor weights){
  CHECK_INPUT(input);
  CHECK_INPUT(weights);
  return bilinear2x1_cuda_forward(input, weights);
  }

std::vector<torch::Tensor> bilinear2x1_backward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor grad_output){
  CHECK_INPUT(input);
  CHECK_INPUT(weights);
  CHECK_INPUT(grad_output);

  return bilinear2x1_cuda_backward(input, weights, grad_output);
}


/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &bilinear2x2_forward, "Bi-Linear 2x2 forward (CUDA)");
  m.def("backward", &bilinear2x2_backward, "Bi-Linear 2x2 backward (CUDA)");
  m.def("forward_2x1", &bilinear2x1_forward, "Bi-Linear 2x1 forward (CUDA)");
  m.def("backward_2x1", &bilinear2x1_backward, "Bi-Linear 2x1 backward (CUDA)");
}
