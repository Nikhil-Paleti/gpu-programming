#include <cuda_runtime.h>
#define BLOCK_SIZE 1024

__global__ void brent_kung(const float* input, float* output, float* aux, int N){
    __shared__ float smem[BLOCK_SIZE];

    int tid = threadIdx.x;
    int threads = blockDim.x;
    int base = blockIdx.x * BLOCK_SIZE;
    
    int g0 = base + tid;
    smem[tid] = (g0 < N) ? input[g0] : 0.0f;

    int g1 = base + threads + tid;
    smem[tid + threads] = (g1 < N) ? input[g1] : 0.0f;
    __syncthreads();

    for(int stride=1; stride<=threads; stride*=2){
        int index = (tid + 1)*2*stride - 1;
        if(index < BLOCK_SIZE){
            smem[index] += smem[index - stride];
        }
        __syncthreads();
    }

    for(int stride=BLOCK_SIZE/4; stride>0; stride /= 2){
        int index = (tid + 1)*2*stride -1;
        if(index+stride < BLOCK_SIZE){
            smem[index + stride] += smem[index];
        }
        __syncthreads();
    }

    if (g0 < N) output[g0] = smem[tid];
    if (g1 < N) output[g1] = smem[tid + threads];

    if (aux != nullptr && tid == threads - 1) {
        aux[blockIdx.x] = smem[BLOCK_SIZE-1];
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
    dim3 block(BLOCK_SIZE / 2);
    dim3 grid((N + BLOCK_SIZE - 1)/BLOCK_SIZE);
    float* aux_d = nullptr;
    if(grid.x > 1){
        cudaMalloc((void**)&aux_d, grid.x * sizeof(float));
    }
    brent_kung<<<grid, block>>>(input, output, aux_d, N);
    cudaDeviceSynchronize();  

    if(grid.x > 1){
        solve(aux_d, aux_d, grid.x);
        cudaDeviceSynchronize();

        dim3 offBlock(BLOCK_SIZE);
        add_offset_kernel<<<grid, offBlock>>>(output, aux_d, N);

        cudaFree(aux_d);
    }
} 