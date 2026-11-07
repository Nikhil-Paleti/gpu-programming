#include <cuda_runtime.h>
#define BLOCK_SIZE 1024

__global__ void kogge_stone(const float* input, float* output, float* aux, int N){
    __shared__ float smem[BLOCK_SIZE];
    int tid = threadIdx.x;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    
    if( x < N){
        smem[threadIdx.x] = input[x];
    }else{
        smem[threadIdx.x] = 0;
    }
    __syncthreads();

    for(int stride=1; stride<blockDim.x; stride*=2){
        float temp = 0.0;
        if(tid >= stride){
            temp = smem[tid - stride] + smem[tid];
        }
        __syncthreads();
        if(tid >= stride){
            smem[tid] = temp;
        }
        __syncthreads();
    }

    if(x < N){
        output[x] = smem[tid];
    }
    
    if (aux != nullptr && tid == blockDim.x - 1) {
        aux[blockIdx.x] = smem[tid];
    }
}

__global__ void add_offset_kernel(float* output, float* aux, int N){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if(x < N && blockIdx.x > 0){
        output[x] += aux[blockIdx.x-1];
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    dim3 block(1024);
    dim3 grid((N + BLOCK_SIZE - 1)/BLOCK_SIZE);
    float* aux_d = nullptr;
    if(grid.x > 1){
        cudaMalloc((void**)&aux_d, grid.x * sizeof(float));
    }
    kogge_stone<<<grid, block>>>(input, output, aux_d, N);
    cudaDeviceSynchronize();

    if(grid.x > 1){
        solve(aux_d, aux_d, grid.x);
        cudaDeviceSynchronize();
        
        add_offset_kernel<<<grid, block>>>(output, aux_d, N);
        cudaFree(aux_d);
    }
} 