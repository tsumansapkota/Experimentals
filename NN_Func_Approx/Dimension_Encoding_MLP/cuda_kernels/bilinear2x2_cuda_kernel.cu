#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define BLOCK_DIM 16

template <typename scalar_t>
__device__ inline scalar_t clamp(scalar_t d, scalar_t min, scalar_t max) {
  const scalar_t t = d < min ? min : d;
  return t > max ? max : t;
}

template <typename scalar_t>
__global__ void bilinear2x2_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> input, 
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> weight, 
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> output, 
    size_t s0, size_t s1, size_t gx, size_t gy)
{
    // Each thread computes one batch of 2x2 operation.
    size_t bi = blockIdx.x * blockDim.x + threadIdx.x; // batch_size
    size_t gi = blockIdx.y * blockDim.y + threadIdx.y; // input_dim//2
    // bi == batch index ; gi == group index
    if ((bi >= s0) || (gi >= s1)){
        return;
    }

    /// inputs (x,y) are calculated for range 0,1 globally...
    scalar_t x = input[bi][gi][0]* (scalar_t)gx; // value on x grid
    scalar_t y = input[bi][gi][1]* (scalar_t)gy; // value on y grid

    int ix = clamp<int>((int)x, 0, gx-2);  // index of x grid
    int iy = clamp<int>((int)y, 0, gy-2);  // index of y grid

    x -= (scalar_t) ix; // true value of x,y for the given piece
    y -= (scalar_t) iy; // given piece is in range [0,1]

    scalar_t a00 = weight[gi][0][ix][iy];
    scalar_t a01 = weight[gi][0][ix][iy+1] - a00;
    scalar_t a10 = weight[gi][0][ix+1][iy] - a00;
    scalar_t a11 = weight[gi][0][ix+1][iy+1] - weight[gi][0][ix+1][iy] - a01;

    output[bi][gi][0] = a00 + x*a10 + y*a01 + x*y*a11 ;

    ////////// for second grid with same pair
    a00 = weight[gi][1][ix][iy];
    a01 = weight[gi][1][ix][iy+1] - a00;
    a10 = weight[gi][1][ix+1][iy] - a00;
    a11 = weight[gi][1][ix+1][iy+1] - weight[gi][1][ix+1][iy] - a01;

    output[bi][gi][1] = a00 + x*a10 + y*a01 + x*y*a11 ;

    return;
}

/// here, we expect the tensor not to be transposed, but does same bmm
std::vector<torch::Tensor> bilinear2x2_cuda_forward(
    torch::Tensor input,
    torch::Tensor weights) {

  const auto s0 = input.size(0);
  const auto s1 = input.size(1);

  const auto gx = weights.size(2);
  const auto gy = weights.size(3);

  // input has shape -> bs, group, 2
  // weights has shape -> group, 2, grid_x, grid_y

  // std::cout<<"Batch Size "<<batch_size<<" Input Size "<<input.size(1)<<","<<input.size(2)<<std::endl;
  dim3 threads_per_block(BLOCK_DIM, BLOCK_DIM);
  // spreading batch across multiple blocks and thread
  dim3 blocks_per_grid(1, 1);
  blocks_per_grid.x = std::ceil(static_cast<double>(s0) /
                                static_cast<double>(threads_per_block.x));
  blocks_per_grid.y = std::ceil(static_cast<double>(s1) /
                                static_cast<double>(threads_per_block.y));

  auto output = torch::zeros_like(input);

  AT_DISPATCH_FLOATING_TYPES(input.type(), "bilinear2x2_forward_cuda", ([&] {
    bilinear2x2_cuda_forward_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
        input.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        weights.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        output.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        s0, s1, gx, gy);
  }));

  return {output};
}



template <typename scalar_t> // mat1 is X, mat2 is W -> Y = X.W
__global__ void bilinear2x2_cuda_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> input, 
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> weight, 
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> del_output, 
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> del_input, 
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> del_weight, 
    size_t s0, size_t s1, size_t gx, size_t gy)
{
    // Each thread computes one batch of 2x2 operation.
    size_t bi = blockIdx.x * blockDim.x + threadIdx.x; // batch_size
    size_t gi = blockIdx.y * blockDim.y + threadIdx.y; // input_dim//2
    // bi == batch index ; gi == group index
    if ((bi >= s0) || (gi >= s1)){
        return;
    }

    /// inputs (x,y) are calculated for range 0,1 globally...
    scalar_t x = input[bi][gi][0]* (scalar_t)gx; // value on x grid
    scalar_t y = input[bi][gi][1]* (scalar_t)gy; // value on y grid

    int ix = clamp<int>((int)x, 0, gx-2);  // index of x grid
    int iy = clamp<int>((int)y, 0, gy-2);  // index of y grid

    x -= (scalar_t) ix; // true value of x,y for the given piece
    y -= (scalar_t) iy; // given piece is in range [0,1]

    ///////// for first pairwise
    scalar_t dy = del_output[bi][gi][0];

    scalar_t a00 = weight[gi][0][ix][iy];
    scalar_t a01 = weight[gi][0][ix][iy+1] - a00;
    scalar_t a10 = weight[gi][0][ix+1][iy] - a00;
    scalar_t a11 = weight[gi][0][ix+1][iy+1] - weight[gi][0][ix+1][iy] - a01;

    del_input[bi][gi][0] = dy*(a10+y*a11);
    del_input[bi][gi][1] = dy*(a01+x*a11);

    scalar_t da01 = dy*y;
    scalar_t da10 = dy*x;
    scalar_t da11 = da10*y;

    del_weight[bi][gi][0][ix+1][iy+1] = da11; 
    del_weight[bi][gi][0][ix][iy+1] = da01 - da11; 
    del_weight[bi][gi][0][ix+1][iy] = da10 - da11;
    del_weight[bi][gi][0][ix][iy] = dy - da01 - da10 + da11; 

    //////////// for second pairwise, doing the same

    dy = del_output[bi][gi][1];

    a00 = weight[gi][1][ix][iy];
    a01 = weight[gi][1][ix][iy+1] - a00;
    a10 = weight[gi][1][ix+1][iy] - a00;
    a11 = weight[gi][1][ix+1][iy+1] - weight[gi][1][ix+1][iy] - a01;

    del_input[bi][gi][0] += dy*(a10+y*a11);
    del_input[bi][gi][1] += dy*(a01+x*a11);
    del_input[bi][gi][0] *= (scalar_t)gx; // correcting for the initial multiplication to gx
    del_input[bi][gi][1] *= (scalar_t)gy;

    da01 = dy*y;
    da10 = dy*x;
    da11 = da10*y;

    del_weight[bi][gi][1][ix+1][iy+1] = da11; 
    del_weight[bi][gi][1][ix][iy+1] = da01 - da11; 
    del_weight[bi][gi][1][ix+1][iy] = da10 - da11;
    del_weight[bi][gi][1][ix][iy] = dy - da01 - da10 + da11; 

    return;
}


std::vector<torch::Tensor> bilinear2x2_cuda_backward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor grad_output) {

  const auto s0 = input.size(0);
  const auto s1 = input.size(1);

  const auto gx = weights.size(2);
  const auto gy = weights.size(3);


  dim3 threads_per_block(BLOCK_DIM, BLOCK_DIM);
  // spreading batch across multiple blocks and thread
  dim3 blocks_per_grid(1, 1);
  blocks_per_grid.x = std::ceil(static_cast<double>(s0) /
                                static_cast<double>(threads_per_block.x));
  blocks_per_grid.y = std::ceil(static_cast<double>(s1) /
                                static_cast<double>(threads_per_block.y));

  auto del_input = torch::zeros_like(input);
  auto del_weights = torch::zeros(
                    {s0, weights.size(0), weights.size(1), gx, gy},
                    input.device());

  AT_DISPATCH_FLOATING_TYPES(input.type(), "bilinear2x2_backward_cuda", ([&] {
    bilinear2x2_cuda_backward_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
        input.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        weights.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        grad_output.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        del_input.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        del_weights.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        s0, s1, gx, gy);
  }));

  return {del_input, torch::sum(del_weights, 0)};
}