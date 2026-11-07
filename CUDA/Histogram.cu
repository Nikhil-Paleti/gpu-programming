#include <cuda_runtime.h>
#include <stdio.h>

__global__ void histo_kernel(const int *input, int *histogram, int N, int num_bins){

    extern __shared__ int histo_s[];

    int x = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = threadIdx.x; i < num_bins; i += blockDim.x){
        histo_s[i] = 0;
    }
    __syncthreads();

    for (int i = x; i < N; i += gridDim.x * blockDim.x){
        int val = input[i];
        if (val >= 0 && val < num_bins) {
            atomicAdd(&(histo_s[val]), 1);
        }
    }
    __syncthreads();

    for(int i = threadIdx.x; i < num_bins; i += blockDim.x){
        if(histo_s[i] > 0) {
            atomicAdd(&(histogram[i]), histo_s[i]);
        }
    }
}

extern "C" void solve(const int* input, int* histogram, int N, int num_bins) {

    cudaMemset(histogram, 0, num_bins * sizeof(int));
    int threads_per_block = 256;
    int elements_processed_by_block = 4 * threads_per_block;

    dim3 threads(threads_per_block);
    dim3 block((N + elements_processed_by_block - 1) / elements_processed_by_block);
    
    size_t shared_mem_size = num_bins * sizeof(int);
    
    histo_kernel<<<block, threads, shared_mem_size>>>(input, histogram, N, num_bins);
}

// #include <cuda_runtime.h>
// #include <stdio.h>

// __global__ void histo_kernel(const int *input, int *histogram, int N, int num_bins){

//     extern __shared__ int histo_s[];

//     int x = blockIdx.x * blockDim.x + threadIdx.x;

//     for(int i = threadIdx.x; i < num_bins; i += blockDim.x){
//         histo_s[i] = 0;
//     }
//     __syncthreads();

//     int prev_block_idx = -1;
//     int acc = 0;
//     for (int i = x; i < N; i += gridDim.x * blockDim.x){
//         int val = input[i];
//         if (val >= 0 && val < num_bins) {
//             if (val == prev_block_idx){
//                 acc++;
//             }
//             else{
//                 if (acc > 0){
//                     atomicAdd(&(histo_s[prev_block_idx]), acc);
//                 }
//                 acc = 1;
//                 prev_block_idx = val;
//             }
//         }
//     }
//     if (acc > 0){
//         atomicAdd(&(histo_s[prev_block_idx]), acc);
//     }
//     __syncthreads();

//     for(int i = threadIdx.x; i < num_bins; i += blockDim.x){
//         if(histo_s[i] > 0) {
//             atomicAdd(&(histogram[i]), histo_s[i]);
//         }
//     }
// }

// extern "C" void solve(const int* input, int* histogram, int N, int num_bins) {

//     cudaMemset(histogram, 0, num_bins * sizeof(int));
//     int threads_per_block = 256;
//     int elements_processed_by_block = 4 * threads_per_block;

//     dim3 threads(threads_per_block);
//     dim3 block((N + elements_processed_by_block - 1) / elements_processed_by_block);
    
//     size_t shared_mem_size = num_bins * sizeof(int);
    
//     histo_kernel<<<block, threads, shared_mem_size>>>(input, histogram, N, num_bins);
// }

// #include <cuda_runtime.h>
// #include <stdio.h>

// __global__ void histo_kernel(const int *input, int *histogram, int N, int num_bins){

//     extern __shared__ int histo_s[];

//     int x = blockIdx.x * blockDim.x + threadIdx.x;

//     for(int i = threadIdx.x; i < num_bins; i += blockDim.x){
//         histo_s[i] = 0;
//     }
//     __syncthreads();

//     if (x < N){
//         int val = input[x];
//         if (val >= 0 && val < num_bins) {
//             atomicAdd(&(histo_s[val]), 1);
//         }
//     }
//     __syncthreads();

//     for(int i = threadIdx.x; i < num_bins; i += blockDim.x){
//         if(histo_s[i] > 0) {
//             atomicAdd(&(histogram[i]), histo_s[i]);
//         }
//     }
// }

// extern "C" void solve(const int* input, int* histogram, int N, int num_bins) {

//     cudaMemset(histogram, 0, num_bins * sizeof(int));

//     dim3 threads(1024);
//     dim3 block((N + 1024 - 1) / 1024);
    
//     size_t shared_mem_size = num_bins * sizeof(int);
    
//     histo_kernel<<<block, threads, shared_mem_size>>>(input, histogram, N, num_bins);
// }


// #include <cuda_runtime.h>

// __global__ void histo_kernel(const int *input, int *histogram, int N, int num_bins){
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
    
//     if (x < N){
//         int val = input[x];
//         if (val >= 0 && val < num_bins) {
//             atomicAdd(&(histogram[val]), 1);
//         }
//     }
// }

// extern "C" void solve(const int* input, int* histogram, int N, int num_bins) {

//     cudaMemset(histogram, 0, num_bins * sizeof(int));

//     dim3 threads(1024);
//     dim3 block((N + 1024 - 1) / 1024);
    
//     histo_kernel<<<block, threads>>>(input, histogram, N, num_bins);
// }