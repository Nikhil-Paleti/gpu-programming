#define BLOCK_DIM 32
#include <cuda_runtime.h>

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {

    __shared__ float tile[BLOCK_DIM][BLOCK_DIM + 1];

    int x = blockIdx.x * BLOCK_DIM + threadIdx.x; 
    int y = blockIdx.y * BLOCK_DIM + threadIdx.y; 

    // Read from input
    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];
    }

    __syncthreads();

    int tx = blockIdx.y * BLOCK_DIM + threadIdx.x;
    int ty = blockIdx.x * BLOCK_DIM + threadIdx.y;
    
    if (tx < rows && ty < cols) {
        output[ty * rows + tx] = tile[threadIdx.x][threadIdx.y];
    }
}

extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}