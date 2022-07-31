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
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> input, 
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> weights,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grids, 
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> input_buffer, 
    size_t s0, size_t s1, size_t gx, size_t gy, size_t num_layers)
{
    // Each thread computes one batch of 2x2 operation.
    size_t bi = blockIdx.x * blockDim.x + threadIdx.x; // batch_size
    size_t gi = blockIdx.y * blockDim.y + threadIdx.y; // input_dim//2
    // bi == batch index ; gi == group index
    if ((bi >= s0) || (gi >= s1)){
        return;
    }
    
    scalar_t x, y, a00, a01, a10, a11;
    int ix, iy;
    size_t gap, gidx, gidy;

    for (size_t layer_i = 0; layer_i < num_layers; layer_i++)
    {
      gap = 1 << layer_i ;
      gidx = (gi%gap) + (gi / gap)*(1<<(layer_i+1)) ;
      gidy = gidx+gap;
      /// save inputs for backward propagation      
      x = input[bi][gidx];
      y = input[bi][gidy];

      input_buffer[layer_i][bi][gidx] = x;
      input_buffer[layer_i][bi][gidy] = y;

      /// using a00 as temp variable
      a00 = x * weights[layer_i][gi][0][0] + y * weights[layer_i][gi][1][0];
      y = x * weights[layer_i][gi][0][1] + y * weights[layer_i][gi][1][1];
      x = a00;

      /// inputs (x,y) are calculated for range 0,1 globally...
      x = x* (scalar_t)(gx-1); // value on x grid
      y = y* (scalar_t)(gy-1); // value on y grid

      ix = clamp<int>((int)x, 0, gx-2);  // index of x grid
      iy = clamp<int>((int)y, 0, gy-2);  // index of y grid

      x -= (scalar_t) ix; // true value of x,y for the given piece
      y -= (scalar_t) iy; // given piece is in range [0,1]

      a00 = grids[layer_i][gi][0][ix][iy];
      a01 = grids[layer_i][gi][0][ix][iy+1] - a00;
      a10 = grids[layer_i][gi][0][ix+1][iy] - a00;
      a11 = grids[layer_i][gi][0][ix+1][iy+1] - grids[layer_i][gi][0][ix+1][iy] - a01;

      input[bi][gidx] = a00 + x*a10 + y*a01 + x*y*a11 ;

      ////////// for second grid with same pair
      a00 = grids[layer_i][gi][1][ix][iy];
      a01 = grids[layer_i][gi][1][ix][iy+1] - a00;
      a10 = grids[layer_i][gi][1][ix+1][iy] - a00;
      a11 = grids[layer_i][gi][1][ix+1][iy+1] - grids[layer_i][gi][1][ix+1][iy] - a01;

      input[bi][gidy] = a00 + x*a10 + y*a01 + x*y*a11 ;

      __syncthreads();
    }
    
    return;
}

/// here, we expect the tensor not to be transposed, but does same bmm
std::vector<torch::Tensor> fused_bilinear2x2_cuda_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor grids) {

  const auto s0 = input.size(0);
  const auto s1 = grids.size(1);

  const auto num_layers = grids.size(0);

  const auto gx = grids.size(3);
  const auto gy = grids.size(4);

  // input has shape -> bs, group*2 == input_dim
  // grids has shape -> layer_idx, group, 2, grid_x, grid_y
  // weights has shape -> layer_idx, group, 2, 2

  // std::cout<<"Batch Size "<<batch_size<<" Input Size "<<input.size(1)<<","<<input.size(2)<<std::endl;
  dim3 threads_per_block(BLOCK_DIM, BLOCK_DIM);
  // spreading batch across multiple blocks and thread
  dim3 blocks_per_grid(1, 1);
  blocks_per_grid.x = std::ceil(static_cast<double>(s0) /
                                static_cast<double>(threads_per_block.x));
  blocks_per_grid.y = std::ceil(static_cast<double>(s1) /
                                static_cast<double>(threads_per_block.y));

  auto input_buffer = torch::zeros({num_layers, s0, s1*2}, input.device());

  AT_DISPATCH_FLOATING_TYPES(input.type(), "bilinear2x2_forward_cuda", ([&] {
    bilinear2x2_cuda_forward_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
        input.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        weights.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        grids.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        input_buffer.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        s0, s1, gx, gy, num_layers);
  }));

  return {input, input_buffer};
}





template <typename scalar_t>
__global__ void bilinear2x2_cuda_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> input_buffer,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> weights,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grids,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> grad_output,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> del_weights,
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> del_grids,
    size_t s0, size_t s1, size_t gx, size_t gy, size_t num_layers)
{
    // Each thread computes one batch of 2x2 operation.
    size_t bi = blockIdx.x * blockDim.x + threadIdx.x; // batch_size
    size_t gi = blockIdx.y * blockDim.y + threadIdx.y; // input_dim//2
    // bi == batch index ; gi == group index
    if ((bi >= s0) || (gi >= s1)){
        return;
    }

    scalar_t x, y, a00, a01, a10, a11, dy, da01, da10, da11, dinp_x, dinp_y;
    int ix, iy;
    size_t gap, gidx, gidy;

    for (int layer_i = num_layers-1; layer_i >= 0; layer_i--)
    {
      gap = 1 << layer_i ;
      gidx = (gi%gap) + (gi / gap)*(1<<(layer_i+1)) ;
      gidy = gidx+gap;
      /// save inputs for backward propagation      

      x = input_buffer[layer_i][bi][gidx] ;
      y = input_buffer[layer_i][bi][gidy] ;

      /// using a00 as temp variable
      a00 = x * weights[layer_i][gi][0][0] + y * weights[layer_i][gi][1][0];
      y = x * weights[layer_i][gi][0][1] + y * weights[layer_i][gi][1][1];
      x = a00;

      /////////////////////////////////////////

      /// inputs (x,y) are calculated for range 0,1 globally...
      x = x* (scalar_t)(gx-1); // value on x grid
      y = y* (scalar_t)(gy-1); // value on y grid

      ix = clamp<int>((int)x, 0, gx-2);  // index of x grid
      iy = clamp<int>((int)y, 0, gy-2);  // index of y grid

      x -= (scalar_t) ix; // true value of x,y for the given piece
      y -= (scalar_t) iy; // given piece is in range [0,1]

      a00 = grids[layer_i][gi][0][ix][iy];
      a01 = grids[layer_i][gi][0][ix][iy+1] - a00;
      a10 = grids[layer_i][gi][0][ix+1][iy] - a00;
      a11 = grids[layer_i][gi][0][ix+1][iy+1] - grids[layer_i][gi][0][ix+1][iy] - a01;

      // compute del input here
      dy = grad_output[bi][gidx];

      dinp_x = dy*(a10+y*a11);
      dinp_y = dy*(a01+x*a11);

      da01 = dy*y;
      da10 = dy*x;
      da11 = da10*y;

      del_grids[layer_i][bi][gi][0][ix+1][iy+1] = da11; 
      del_grids[layer_i][bi][gi][0][ix][iy+1] = da01 - da11; 
      del_grids[layer_i][bi][gi][0][ix+1][iy] = da10 - da11;
      del_grids[layer_i][bi][gi][0][ix][iy] = dy - da01 - da10 + da11; 

      //////////// for second pairwise, doing the same


      a00 = grids[layer_i][gi][1][ix][iy];
      a01 = grids[layer_i][gi][1][ix][iy+1] - a00;
      a10 = grids[layer_i][gi][1][ix+1][iy] - a00;
      a11 = grids[layer_i][gi][1][ix+1][iy+1] - grids[layer_i][gi][1][ix+1][iy] - a01;

      dy = grad_output[bi][gidy];

      // this is del output for 2x2 linear layer before bilinear
      dinp_x += dy*(a10+y*a11);
      dinp_y += dy*(a01+x*a11);
      dinp_x *= (scalar_t)(gx-1); // correcting for the initial multiplication to gx
      dinp_y *= (scalar_t)(gy-1);

      da01 = dy*y;
      da10 = dy*x;
      da11 = da10*y;

      del_grids[layer_i][bi][gi][1][ix+1][iy+1] = da11; 
      del_grids[layer_i][bi][gi][1][ix][iy+1] = da01 - da11; 
      del_grids[layer_i][bi][gi][1][ix+1][iy] = da10 - da11;
      del_grids[layer_i][bi][gi][1][ix][iy] = dy - da01 - da10 + da11; 

      /////////////////////////////////////////
      /////////////////////////////////////////
      /// for input weight layer
      del_weights[layer_i][bi][gi][0][0] = dinp_x*input_buffer[layer_i][bi][gidx];
      del_weights[layer_i][bi][gi][0][1] = dinp_y*input_buffer[layer_i][bi][gidy];
      del_weights[layer_i][bi][gi][1][0] = dinp_x*input_buffer[layer_i][bi][gidx];
      del_weights[layer_i][bi][gi][1][1] = dinp_y*input_buffer[layer_i][bi][gidy];
      
      /// this is actually del_input, however the variables are reused for next iteraton
      grad_output[bi][gidx] = dinp_x * weights[layer_i][bi][0][0] + 
                            dinp_y * weights[layer_i][bi][0][1];
      grad_output[bi][gidy] = dinp_x * weights[layer_i][bi][1][0] + 
                            dinp_y * weights[layer_i][bi][1][1];

      __syncthreads();
    }
    
    return;
}


std::vector<torch::Tensor> fused_bilinear2x2_cuda_backward(
    torch::Tensor input_buffer,
    torch::Tensor weights,
    torch::Tensor grids,
    torch::Tensor grad_output) {

  // input_buffer has shape -> layer_idx, bs, group*2 == input_dim
  // grids has shape -> layer_idx, group, 2, grid_x, grid_y
  // weights has shape -> layer_idx, group, 2, 2
  // grad_output has shape -> bs, group*2

  const auto s0 = input_buffer.size(1);
  const auto s1 = grids.size(1);

  const auto num_layers = grids.size(0);

  const auto gx = grids.size(3);
  const auto gy = grids.size(4);

  dim3 threads_per_block(BLOCK_DIM, BLOCK_DIM);
  // spreading batch across multiple blocks and thread
  dim3 blocks_per_grid(1, 1);
  blocks_per_grid.x = std::ceil(static_cast<double>(s0) /
                                static_cast<double>(threads_per_block.x));
  blocks_per_grid.y = std::ceil(static_cast<double>(s1) /
                                static_cast<double>(threads_per_block.y));

  auto del_weights = torch::zeros(
                    {num_layers, s0, s1, 2, 2},
                    weights.device());
  auto del_grids = torch::zeros(
                    {num_layers, s0, s1, grids.size(2), gx, gy},
                    grids.device()); // grids.size(2) == 2

  AT_DISPATCH_FLOATING_TYPES(input_buffer.type(), "bilinear2x2_backward_cuda", ([&] {
    bilinear2x2_cuda_backward_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
        input_buffer.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        weights.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        grids.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        grad_output.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        del_weights.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        del_grids.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
        s0, s1, gx, gy, num_layers);
  }));

  return {grad_output, torch::sum(del_weights, 1), torch::sum(del_grids, 1)};
}