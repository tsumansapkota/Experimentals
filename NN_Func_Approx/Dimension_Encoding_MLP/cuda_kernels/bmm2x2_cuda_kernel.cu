#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define BLOCK_DIM 16

/*
template <typename scalar_t>
__global__ void bmm2x2_cuda_forward_kernel(
    const scalar_t* __restrict__ mat_1, 
    const scalar_t* __restrict__ mat_2, 
    scalar_t* __restrict__ mat_3, 
    size_t b)
{
    // Each thread computes one batch of 2x2 matmul.
    size_t i4 = blockIdx.x * blockDim.x + threadIdx.x;

    if (i4 >= b){
        return;
    }
    i4 = i4*4 ;
    mat_3[i4] = mat_1[i4] * mat_2[i4] + mat_1[i4+1]*mat_2[i4+2];
    mat_3[i4+1] = mat_1[i4] * mat_2[i4+1] + mat_1[i4+1]*mat_2[i4+3];
    mat_3[i4+2] = mat_1[i4+2] * mat_2[i4] + mat_1[i4+3]*mat_2[i4+2];
    mat_3[i4+3] = mat_1[i4+2] * mat_2[i4+1] + mat_1[i4+3]*mat_2[i4+3];

    return;
}

template <typename scalar_t> // mat1 is X, mat2 is W -> Y = X.W
__global__ void bmm2x2_cuda_backward_kernel(
    const scalar_t* __restrict__ mat_1, 
    const scalar_t* __restrict__ mat_2,
    const scalar_t* __restrict__ d_out, 
    scalar_t* __restrict__ d_mat1, 
    scalar_t* __restrict__ d_mat2, 
    size_t b)
{
    // dmat1 is dX, dmat2 is dW, d_out is dY
    // Each thread computes one batch of 2x2 matmul.
    size_t i4 = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i4 >= b){
        return;
    }
    i4 = i4*4 ;

    /// computing dX = dY.(W^t)
    d_mat1[i4] = d_out[i4] * mat_2[i4] + d_out[i4+1]*mat_2[i4+1];
    d_mat1[i4+1] = d_out[i4] * mat_2[i4+2] + d_out[i4+1]*mat_2[i4+3];
    d_mat1[i4+2] = d_out[i4+2] * mat_2[i4] + d_out[i4+3]*mat_2[i4+1];
    d_mat1[i4+3] = d_out[i4+2] * mat_2[i4+2] + d_out[i4+3]*mat_2[i4+3];

    /// computing dW = dX^t.dY
    d_mat2[i4] = mat_1[i4] * d_out[i4] + mat_1[i4+2]*d_out[i4+2];
    d_mat2[i4+1] = mat_1[i4] * d_out[i4+1] + mat_1[i4+2]*d_out[i4+3];
    d_mat2[i4+2] = mat_1[i4+1] * d_out[i4] + mat_1[i4+3]*d_out[i4+2];
    d_mat2[i4+3] = mat_1[i4+1] * d_out[i4+1] + mat_1[i4+3]*d_out[i4+3];
    return;
}
*/

/* this one is working perfectly
template <typename scalar_t>
__global__ void bmm2x2_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> input, 
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> weight, 
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> output, 
    size_t s0, size_t s1)
{
    // Each thread computes one batch of 2x2 matmul.
    size_t i0 = blockIdx.x * blockDim.x + threadIdx.x; // input_dim//2 
    size_t i1 = blockIdx.y * blockDim.y + threadIdx.y; // batch_size
    if ((i0 >= s0) || (i1 >= s1)){
        return;
    }
    
    output[i0][i1][0] = input[i0][i1][0] * weight[i0][0][0] + 
                          input[i0][i1][1] * weight[i0][1][0];
    output[i0][i1][1] = input[i0][i1][0] * weight[i0][0][1] + 
                          input[i0][i1][1] * weight[i0][1][1];

    return;
}

template <typename scalar_t> // mat1 is X, mat2 is W -> Y = X.W
__global__ void bmm2x2_cuda_backward_kernel(
    const scalar_t* __restrict__ mat_1, 
    const scalar_t* __restrict__ mat_2,
    const scalar_t* __restrict__ d_out, 
    scalar_t* __restrict__ d_mat1, 
    scalar_t* __restrict__ d_mat2, 
    size_t b)
{
    // dmat1 is dX, dmat2 is dW, d_out is dY
    // Each thread computes one batch of 2x2 matmul.
    size_t i4 = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i4 >= b){
        return;
    }
    i4 = i4*4 ;

    /// computing dX = dY.(W^t)
    d_mat1[i4] = d_out[i4] * mat_2[i4] + d_out[i4+1]*mat_2[i4+1];
    d_mat1[i4+1] = d_out[i4] * mat_2[i4+2] + d_out[i4+1]*mat_2[i4+3];
    d_mat1[i4+2] = d_out[i4+2] * mat_2[i4] + d_out[i4+3]*mat_2[i4+1];
    d_mat1[i4+3] = d_out[i4+2] * mat_2[i4+2] + d_out[i4+3]*mat_2[i4+3];

    /// computing dW = dX^t.dY
    d_mat2[i4] = mat_1[i4] * d_out[i4] + mat_1[i4+2]*d_out[i4+2];
    d_mat2[i4+1] = mat_1[i4] * d_out[i4+1] + mat_1[i4+2]*d_out[i4+3];
    d_mat2[i4+2] = mat_1[i4+1] * d_out[i4] + mat_1[i4+3]*d_out[i4+2];
    d_mat2[i4+3] = mat_1[i4+1] * d_out[i4+1] + mat_1[i4+3]*d_out[i4+3];
    return;
}


//////////////////////////////////////////////////

std::vector<torch::Tensor> bmm2x2_cuda_forward(
    torch::Tensor input,
    torch::Tensor weights) {

  const auto s0 = input.size(0);
  const auto s1 = input.size(1);

  // std::cout<<"Batch Size "<<batch_size<<" Input Size "<<input.size(1)<<","<<input.size(2)<<std::endl;
  dim3 threads_per_block(BLOCK_DIM, BLOCK_DIM);
  // spreading batch across multiple blocks and thread
  dim3 blocks_per_grid(1, 1);
  blocks_per_grid.x = std::ceil(static_cast<double>(s0) /
                                static_cast<double>(threads_per_block.x));
  blocks_per_grid.y = std::ceil(static_cast<double>(s1) /
                                static_cast<double>(threads_per_block.y));

  // size_t threads_per_block = BLOCK_DIM*BLOCK_DIM;
  // size_t blocks_per_grid = std::ceil(static_cast<double>(batch_size) /
  //                                 static_cast<double>(threads_per_block));

  // const int threads_per_block = 1024; // default is 1024
  // const dim3 blocks_per_grid((batch_size + threads - 1) / threads, batch_size);

  auto output = torch::zeros_like(input);

  AT_DISPATCH_FLOATING_TYPES(input.type(), "bmm2x2_forward_cuda", ([&] {
    bmm2x2_cuda_forward_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
        input.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        weights.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        output.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        s0, s1);
  }));

  return {output};
}

std::vector<torch::Tensor> bmm2x2_cuda_backward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor grad_output) {

  const auto batch_size = input.size(0);

  std::cout<<"Batch Size"<<batch_size<<std::endl;

  // dim3 threads_per_block(BLOCK_DIM, BLOCK_DIM);
  // // spreading batch across multiple blocks and thread
  // dim3 blocks_per_grid(1, 1);
  // blocks_per_grid.x = std::ceil(static_cast<double>(p) /
  //                               static_cast<double>(threads_per_block.x));
  // blocks_per_grid.y = std::ceil(static_cast<double>(m) /
  //                               static_cast<double>(threads_per_block.y));


  size_t threads_per_block = BLOCK_DIM*BLOCK_DIM;
  size_t blocks_per_grid = std::ceil(static_cast<double>(batch_size) /
                                  static_cast<double>(threads_per_block));
  // const int threads = 1024;
  // const dim3 blocks((state_size + threads - 1) / threads, batch_size);

  auto del_input = torch::zeros_like(input);
  auto del_weights = torch::zeros_like(input);

  AT_DISPATCH_FLOATING_TYPES(input.type(), "bmm2x2_backward_cuda", ([&] {
    bmm2x2_cuda_backward_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
        input.data<scalar_t>(),
        weights.data<scalar_t>(),
        grad_output.data<scalar_t>(),
        del_input.data<scalar_t>(),
        del_weights.data<scalar_t>(),
        batch_size);
  }));

  return {del_input, del_weights};
}
*/


template <typename scalar_t>
__global__ void bmm2x2_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> input, 
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> weight, 
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> output, 
    size_t s0, size_t s1)
{
    // Each thread computes one batch of 2x2 matmul.
    size_t i0 = blockIdx.x * blockDim.x + threadIdx.x; // batch_size
    size_t i1 = blockIdx.y * blockDim.y + threadIdx.y; // input_dim//2
    if ((i0 >= s0) || (i1 >= s1)){
        return;
    }
    
    output[i0][i1][0] = input[i0][i1][0] * weight[i1][0][0] + 
                          input[i0][i1][1] * weight[i1][1][0];
    output[i0][i1][1] = input[i0][i1][0] * weight[i1][0][1] + 
                          input[i0][i1][1] * weight[i1][1][1];

    return;
}

/// here, we expect the tensor not to be transposed, but does same bmm
std::vector<torch::Tensor> bmm2x2_cuda_forward(
    torch::Tensor input,
    torch::Tensor weights) {

  const auto s0 = input.size(0);
  const auto s1 = input.size(1);

  // std::cout<<"Batch Size "<<batch_size<<" Input Size "<<input.size(1)<<","<<input.size(2)<<std::endl;
  dim3 threads_per_block(BLOCK_DIM, BLOCK_DIM);
  // spreading batch across multiple blocks and thread
  dim3 blocks_per_grid(1, 1);
  blocks_per_grid.x = std::ceil(static_cast<double>(s0) /
                                static_cast<double>(threads_per_block.x));
  blocks_per_grid.y = std::ceil(static_cast<double>(s1) /
                                static_cast<double>(threads_per_block.y));

  // size_t threads_per_block = BLOCK_DIM*BLOCK_DIM;
  // size_t blocks_per_grid = std::ceil(static_cast<double>(batch_size) /
  //                                 static_cast<double>(threads_per_block));

  // const int threads_per_block = 1024; // default is 1024
  // const dim3 blocks_per_grid((batch_size + threads - 1) / threads, batch_size);

  auto output = torch::zeros_like(input);

  AT_DISPATCH_FLOATING_TYPES(input.type(), "bmm2x2_forward_cuda", ([&] {
    bmm2x2_cuda_forward_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
        input.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        weights.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        output.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        s0, s1);
  }));

  return {output};
}



template <typename scalar_t> // mat1 is X, mat2 is W -> Y = X.W
__global__ void bmm2x2_cuda_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> input, 
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> weight, 
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> del_output, 
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> del_input, 
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> del_weight, 
    size_t s0, size_t s1)
{
    // Each thread computes one batch of 2x2 matmul.
    size_t i0 = blockIdx.x * blockDim.x + threadIdx.x; // batch_size
    size_t i1 = blockIdx.y * blockDim.y + threadIdx.y; // input_dim//2
    if ((i0 >= s0) || (i1 >= s1)){
        return;
    }

    /// computing dX = dY.(W^t)
    del_input[i0][i1][0] = del_output[i0][i1][0] * weight[i1][0][0] + 
                          del_output[i0][i1][1] * weight[i1][0][1];
    del_input[i0][i1][1] = del_output[i0][i1][0] * weight[i1][1][0] + 
                          del_output[i0][i1][1] * weight[i1][1][1];


    // d_mat1[i4] = d_out[i4] * mat_2[i4] + d_out[i4+1]*mat_2[i4+1];
    // d_mat1[i4+1] = d_out[i4] * mat_2[i4+2] + d_out[i4+1]*mat_2[i4+3];
    // d_mat1[i4+2] = d_out[i4+2] * mat_2[i4] + d_out[i4+3]*mat_2[i4+1];
    // d_mat1[i4+3] = d_out[i4+2] * mat_2[i4+2] + d_out[i4+3]*mat_2[i4+3];

    /// computing dW = X^t.dY
    del_weight[i0][i1][0][0] = del_output[i0][i1][0]*input[i0][i1][0];
    del_weight[i0][i1][0][1] = del_output[i0][i1][1]*input[i0][i1][0];
    del_weight[i0][i1][1][0] = del_output[i0][i1][0]*input[i0][i1][1];
    del_weight[i0][i1][1][1] = del_output[i0][i1][1]*input[i0][i1][1];
    
    // d_mat2[i4] = mat_1[i4] * d_out[i4] + mat_1[i4+2]*d_out[i4+2];
    // d_mat2[i4+1] = mat_1[i4] * d_out[i4+1] + mat_1[i4+2]*d_out[i4+3];
    // d_mat2[i4+2] = mat_1[i4+1] * d_out[i4] + mat_1[i4+3]*d_out[i4+2];
    // d_mat2[i4+3] = mat_1[i4+1] * d_out[i4+1] + mat_1[i4+3]*d_out[i4+3];
    return;
}


std::vector<torch::Tensor> bmm2x2_cuda_backward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor grad_output) {

  const auto s0 = input.size(0);
  const auto s1 = input.size(1);

  dim3 threads_per_block(BLOCK_DIM, BLOCK_DIM);
  // spreading batch across multiple blocks and thread
  dim3 blocks_per_grid(1, 1);
  blocks_per_grid.x = std::ceil(static_cast<double>(s0) /
                                static_cast<double>(threads_per_block.x));
  blocks_per_grid.y = std::ceil(static_cast<double>(s1) /
                                static_cast<double>(threads_per_block.y));

  auto del_input = torch::zeros_like(input);
  auto del_weights = torch::empty({s0, s1, 2, 2}, input.device());

  AT_DISPATCH_FLOATING_TYPES(input.type(), "bmm2x2_backward_cuda", ([&] {
    bmm2x2_cuda_backward_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
        input.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        weights.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        grad_output.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        del_input.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        del_weights.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        s0, s1);
  }));

  return {del_input, torch::sum(del_weights, 0)};
}

////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////

/// this is for computing del_input and del_weight seperately
template <typename scalar_t> // mat1 is X, mat2 is W -> Y = X.W
__global__ void bmm2x2_cuda_backward_delinput_kernel( 
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> weight, 
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> del_output, 
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> del_input, 
    size_t s0, size_t s1)
{
    // Each thread computes one batch of 2x2 matmul.
    size_t i0 = blockIdx.x * blockDim.x + threadIdx.x; // batch_size
    size_t i1 = blockIdx.y * blockDim.y + threadIdx.y; // input_dim//2
    if ((i0 >= s0) || (i1 >= s1)){
        return;
    }

    /// computing dX = dY.(W^t)
    del_input[i0][i1][0] = del_output[i0][i1][0] * weight[i1][0][0] + 
                          del_output[i0][i1][1] * weight[i1][0][1];
    del_input[i0][i1][1] = del_output[i0][i1][0] * weight[i1][1][0] + 
                          del_output[i0][i1][1] * weight[i1][1][1];

    return;
}

template <typename scalar_t> // mat1 is X, mat2 is W -> Y = X.W
__global__ void bmm2x2_cuda_backward_delweight_kernel( 
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> input, 
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> del_output, 
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> del_weight, 
    size_t s0, size_t s1)
{
    // Each thread computes one batch of 2x2 matmul.
    size_t i1 = blockIdx.x * blockDim.x + threadIdx.x; // input_dim//2
    if (i1 >= s1){
        return;
    }
    scalar_t t0=0, t1=0, t2=0, t3=0;
    for (size_t k{0}; k < s0; ++k){
        t0 += del_output[k][i1][0] * input[k][i1][0];
        t1 += del_output[k][i1][1] * input[k][i1][0];
        t2 += del_output[k][i1][0] * input[k][i1][1];
        t3 += del_output[k][i1][1] * input[k][i1][1];
    }
    // scalar_t _s0 = static_cast<scalar_t>(s0);
    del_weight[i1][0][0] = t0;
    del_weight[i1][0][1] = t1;
    del_weight[i1][1][0] = t2;
    del_weight[i1][1][1] = t3;
    return;
}



std::vector<torch::Tensor> bmm2x2_cuda_backward_v2(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor grad_output) {

  const auto s0 = input.size(0);
  const auto s1 = input.size(1);

  dim3 threads_per_block(BLOCK_DIM, BLOCK_DIM);
  // spreading batch across multiple blocks and thread
  dim3 blocks_per_grid(1, 1);
  blocks_per_grid.x = std::ceil(static_cast<double>(s0) /
                                static_cast<double>(threads_per_block.x));
  blocks_per_grid.y = std::ceil(static_cast<double>(s1) /
                                static_cast<double>(threads_per_block.y));

  auto del_input = torch::zeros_like(input);
  auto del_weights = torch::empty_like(weights);

  AT_DISPATCH_FLOATING_TYPES(input.type(), "bmm2x2_backward_input_cuda", ([&] {
    bmm2x2_cuda_backward_delinput_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
        weights.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        grad_output.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        del_input.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        s0, s1);
  }));

  
  int threads = BLOCK_DIM*BLOCK_DIM;
  int blocks = std::ceil(static_cast<double>(s1) /
                        static_cast<double>(threads));
  AT_DISPATCH_FLOATING_TYPES(input.type(), "bmm2x2_backward_weight_cuda", ([&] {
    // scalar_t _s0 = static_cast<scalar_t>(s0);
    bmm2x2_cuda_backward_delweight_kernel<scalar_t><<<blocks, threads>>>(
        input.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        grad_output.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        del_weights.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        s0, s1);
  }));

  return {del_input, del_weights};
}









//////////////////////////////////////////////////////
//////////////////////////////////////////////////////
//////////////////////////////////////////////////////
//////////////////////////////////////////////////////
//////////////////////////////////////////////////////


///////////// THIS PORTION CONTAINS CODE FOR BMM 2X1 (HALVER) FOR 2X2 PORTION


template <typename scalar_t>
__global__ void bmm2x1_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> input, 
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> weight, 
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output, 
    size_t s0, size_t s1)
{
    // Each thread computes one batch of 2x1 matmul.
    size_t i0 = blockIdx.x * blockDim.x + threadIdx.x; // batch_size
    size_t i1 = blockIdx.y * blockDim.y + threadIdx.y; // input_dim//2
    if ((i0 >= s0) || (i1 >= s1)){
        return;
    }
    
    output[i0][i1] = input[i0][i1][0] * weight[i1][0] + 
                          input[i0][i1][1] * weight[i1][1];
    return;
}

/// here, we expect the tensor not to be transposed, but does same bmm
std::vector<torch::Tensor> bmm2x1_cuda_forward(
    torch::Tensor input,
    torch::Tensor weights) {


    /// input has shape -> batch size, n_grids, 2
    /// weight has shape -> n_grids, 2 ## this is n_grids, 2, 1 for halving the number of inputs.

  const auto s0 = input.size(0);
  const auto s1 = input.size(1);

  // std::cout<<"Batch Size "<<batch_size<<" Input Size "<<input.size(1)<<","<<input.size(2)<<std::endl;
  dim3 threads_per_block(BLOCK_DIM, BLOCK_DIM);
  // spreading batch across multiple blocks and thread
  dim3 blocks_per_grid(1, 1);
  blocks_per_grid.x = std::ceil(static_cast<double>(s0) /
                                static_cast<double>(threads_per_block.x));
  blocks_per_grid.y = std::ceil(static_cast<double>(s1) /
                                static_cast<double>(threads_per_block.y));

  // size_t threads_per_block = BLOCK_DIM*BLOCK_DIM;
  // size_t blocks_per_grid = std::ceil(static_cast<double>(batch_size) /
  //                                 static_cast<double>(threads_per_block));

  // const int threads_per_block = 1024; // default is 1024
  // const dim3 blocks_per_grid((batch_size + threads - 1) / threads, batch_size);

  auto output = torch::zeros({s0, s1}, input.device());
  /// output has shape Batch, n_group 

  AT_DISPATCH_FLOATING_TYPES(input.type(), "bmm2x1_forward_cuda", ([&] {
    bmm2x1_cuda_forward_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
        input.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        weights.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        output.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        s0, s1);
  }));

  return {output};
}



template <typename scalar_t> // mat1 is X, mat2 is W -> Y = X.W
__global__ void bmm2x1_cuda_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> input, 
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> weight, 
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> del_output, 
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> del_input, 
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> del_weight, 
    size_t s0, size_t s1)
{
    // Each thread computes one batch of 2x2 matmul.
    size_t i0 = blockIdx.x * blockDim.x + threadIdx.x; // batch_size
    size_t i1 = blockIdx.y * blockDim.y + threadIdx.y; // input_dim//2
    if ((i0 >= s0) || (i1 >= s1)){
        return;
    }

    /// computing dX = dY.(W^t)
    del_input[i0][i1][0] = del_output[i0][i1] * weight[i1][0];
    del_input[i0][i1][1] = del_output[i0][i1] * weight[i1][1];

    /// computing dW = X^t.dY
    del_weight[i0][i1][0] = del_output[i0][i1]*input[i0][i1][0];
    del_weight[i0][i1][1] = del_output[i0][i1]*input[i0][i1][1];

    return;
}


std::vector<torch::Tensor> bmm2x1_cuda_backward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor grad_output) {


    /// input has shape -> batch size, n_grids, 2
    /// weight has shape -> n_grids, 2

  const auto s0 = input.size(0);
  const auto s1 = input.size(1);

  dim3 threads_per_block(BLOCK_DIM, BLOCK_DIM);
  // spreading batch across multiple blocks and thread
  dim3 blocks_per_grid(1, 1);
  blocks_per_grid.x = std::ceil(static_cast<double>(s0) /
                                static_cast<double>(threads_per_block.x));
  blocks_per_grid.y = std::ceil(static_cast<double>(s1) /
                                static_cast<double>(threads_per_block.y));

  auto del_input = torch::zeros_like(input);
  auto del_weights = torch::empty({s0, s1, 2}, input.device());

  AT_DISPATCH_FLOATING_TYPES(input.type(), "bmm2x1_backward_cuda", ([&] {
    bmm2x1_cuda_backward_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
        input.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        weights.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        grad_output.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        del_input.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        del_weights.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        s0, s1);
  }));

  return {del_input, torch::sum(del_weights, 0)};
}