#define BLOCK_DIM 32
#include <cuda_runtime.h>

__global__ void batched_transpose_2d_grid(const float* input, float* output, int rows, int cols, int batch_size) {

    int blocks_per_mat_y = (rows + BLOCK_DIM - 1) / BLOCK_DIM;

    int batch_id = blockIdx.y / blocks_per_mat_y;
    
    int by = blockIdx.y % blocks_per_mat_y;
    int bx = blockIdx.x;

    if (batch_id >= batch_size) return;

    int offset = batch_id * rows * cols;

    
    __shared__ float tile[BLOCK_DIM][BLOCK_DIM + 1];

    int x = bx * BLOCK_DIM + threadIdx.x; 
    int y = by * BLOCK_DIM + threadIdx.y; 

    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = input[offset + y * cols + x];
    }

    __syncthreads();


    int tx = by * BLOCK_DIM + threadIdx.x; 
    int ty = bx * BLOCK_DIM + threadIdx.y;

    if (tx < rows && ty < cols) {
        output[offset + ty * rows + tx] = tile[threadIdx.x][threadIdx.y];
    }
}

extern "C" void solve(const float* input, float* output, int rows, int cols, int batch_size) {
    dim3 threadsPerBlock(BLOCK_DIM, BLOCK_DIM);

    // calculate how many blocks needed for ONE matrix
    int grid_x = (cols + threadsPerBlock.x - 1) / threadsPerBlock.x;
    int grid_y_per_matrix = (rows + threadsPerBlock.y - 1) / threadsPerBlock.y;

    // Make the grid Y dimension N times bigger to hold all batches
    dim3 blocksPerGrid(grid_x, 
                       grid_y_per_matrix * batch_size); 

    batched_transpose_2d_grid<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols, batch_size);
    cudaDeviceSynchronize();
}