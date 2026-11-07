#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <algorithm>

// -------------------------------------------------------------------------
// TUNING PARAMETERS
// -------------------------------------------------------------------------
// Block Tile dimensions (Shared Memory size)
#define BM 128
#define BN 128
#define BK 8

// Thread Tile dimensions (Register size per thread)
// Each thread calculates a TM x TN sub-matrix of C
#define TM 8
#define TN 8

// Resulting Block Dimensions (128/8 = 16 threads in each dimension)
// BlockDim.x = BN/TN = 16
// BlockDim.y = BM/TM = 16
// Total threads = 256

void generateRandomMatrix(float *matrix, int rows, int cols){
    for(int i=0; i<rows*cols; i++){
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// -------------------------------------------------------------------------
// REGISTER TILED KERNEL
// -------------------------------------------------------------------------
__global__ void matmul_register_tiled(float *A, float *B, float *C, int M, int N, int K) {
    // 1. Allocate Shared Memory
    // These tiles cache Global Memory
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    // 2. Allocate Registers
    // accum: Accumulates the results for the TMxTN sub-tile
    // regM/regN: Cache values from Shared Memory during compute
    float accum[TM][TN] = {0.0f};
    float regM[TM];
    float regN[TN];

    // Calculate thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Global row and column for this thread block
    int rowStart = by * BM;
    int colStart = bx * BN;

    // Linear thread ID for cooperative loading
    int tid = ty * blockDim.x + tx; 
    int numThreads = blockDim.x * blockDim.y; // Should be 256

    // ---------------------------------------------------------------------
    // MAIN LOOP: Iterate over K dimension in chunks of BK
    // ---------------------------------------------------------------------
    for (int ph = 0; ph < (K + BK - 1) / BK; ++ph) {
        
        // --- LOADING PHASE: Global Memory -> Shared Memory ---
        // Usage of Strided Loops for cleaner indexing and robust handling of
        // cases where Tile Size is not a perfect multiple of Block Size.
        
        // 1. Load A row-major: As[BM][BK]
        // Iterate over the shared memory tile with stride = numThreads
        for (int i = tid; i < BM * BK; i += numThreads) {
            int row = i / BK;
            int col = i % BK;
            int globalRow = rowStart + row;
            int globalCol = ph * BK + col;

            if (globalRow < M && globalCol < K) {
                As[row][col] = A[globalRow * K + globalCol];
            } else {
                As[row][col] = 0.0f;
            }
        }

        // 2. Load B row-major: Bs[BK][BN]
        // Iterate over the shared memory tile with stride = numThreads
        for (int i = tid; i < BK * BN; i += numThreads) {
            int row = i / BN;
            int col = i % BN;
            int globalRow = ph * BK + row;
            int globalCol = colStart + col;

            if (globalRow < K && globalCol < N) {
                Bs[row][col] = B[globalRow * N + globalCol];
            } else {
                Bs[row][col] = 0.0f;
            }
        }

        // Wait for all threads to finish loading shared memory
        __syncthreads();

        // --- COMPUTE PHASE: Shared Memory -> Registers ---
        // Iterate within the tile (BK steps)
        for (int k = 0; k < BK; ++k) {
            
            // 1. Load required values from Shared Mem into Registers
            // We load the specific column from A and row from B required
            // for this thread's sub-tile
            for (int i = 0; i < TM; ++i) {
                regM[i] = As[ty * TM + i][k];
            }
            for (int j = 0; j < TN; ++j) {
                regN[j] = Bs[k][tx * TN + j];
            }

            // 2. Compute Outer Product (accumulate into registers)
            // This is purely register arithmetic (FAST)
            for (int i = 0; i < TM; ++i) {
                for (int j = 0; j < TN; ++j) {
                    accum[i][j] += regM[i] * regN[j];
                }
            }
        }
        
        // Wait before overwriting shared memory in next phase
        __syncthreads();
    }

    // ---------------------------------------------------------------------
    // WRITE BACK: Registers -> Global Memory
    // ---------------------------------------------------------------------
    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
            int globalRow = rowStart + ty * TM + i;
            int globalCol = colStart + tx * TN + j;
            
            if (globalRow < M && globalCol < N) {
                C[globalRow * N + globalCol] = accum[i][j];
            }
        }
    }
}

// Host wrapper
void matmul(float *A_h, float *B_h, float *C_h, int M, int N, int K){
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    float *A_d, *B_d, *C_d;
    cudaMalloc((void**)&A_d, size_A);
    cudaMalloc((void**)&B_d, size_B);
    cudaMalloc((void**)&C_d, size_C);
    
    cudaMemcpy(A_d, A_h, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size_B, cudaMemcpyHostToDevice);
    
    // Grid and Block setup
    // Block dimensions are determined by Tile Size / Thread Tile Size
    dim3 block(BN / TN, BM / TM); 
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    
    printf("Launching Kernel with Grid: (%d, %d), Block: (%d, %d)\n", 
           grid.x, grid.y, block.x, block.y);

    matmul_register_tiled<<<grid, block>>>(A_d, B_d, C_d, M, N, K);
    
    cudaMemcpy(C_h, C_d, size_C, cudaMemcpyDeviceToHost);
    
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main(){
    int M = 1024;
    int N = 1024;
    int K = 1024; 
    
    float *A_h = new float[M*K];
    float *B_h = new float[K*N];
    float *C_h = new float[M*N];
    
    srand(42);
    generateRandomMatrix(A_h, M, K);
    generateRandomMatrix(B_h, K, N);
    
    // Warmup
    matmul(A_h, B_h, C_h, M, N, K);
    
    // Actual Run
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    matmul(A_h, B_h, C_h, M, N, K);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("Time: %f ms\n", milliseconds);
    
    // Simple verification of corner element
    printf("C[0] = %f\n", C_h[0]);

    delete[] A_h;
    delete[] B_h;
    delete[] C_h;
    return 0;
}