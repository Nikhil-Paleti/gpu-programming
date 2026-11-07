// merge_tiled_demo.cu
#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include <random>
#include <cassert>

// ---------------- co-rank ----------------
__host__ __device__ __forceinline__
int co_rank(int k, const int* A, int m, const int* B, int n) {
    int i_low  = max(0, k - n);
    int i_high = min(k, m);

    while (true) {
        int i = (i_low + i_high) >> 1;
        int j = k - i;

        // took too many from A
        if (i > 0 && j < n && A[i - 1] > B[j]) {
            i_high = i - 1;
        }
        // took too few from A (>= keeps it stable: A before B on ties)
        else if (j > 0 && i < m && B[j - 1] >= A[i]) {
            i_low = i + 1;
        } else {
            return i;
        }
    }
}

// ---------------- sequential merge ----------------
__host__ __device__ __forceinline__
void merge_sequential(const int* A, int lenA,
                      const int* B, int lenB,
                      int* C) {
    int i = 0, j = 0, k = 0;
    while (i < lenA && j < lenB) {
        if (A[i] <= B[j]) C[k++] = A[i++];
        else              C[k++] = B[j++];
    }
    while (i < lenA) C[k++] = A[i++];
    while (j < lenB) C[k++] = B[j++];
}

// ---------------- shared-memory tiled merge kernel ----------------
__global__ void merge_tiled_kernel(const int* A, int m,
                                  const int* B, int n,
                                  int* C,
                                  int tile_size) {
    extern __shared__ int smem[]; // size = 2*tile_size ints
    int* A_S = smem;
    int* B_S = smem + tile_size;

    const int total = m + n;

    // block's output segment in C
    const int elemsPerBlock = (total + gridDim.x - 1) / gridDim.x;
    const int C_block_start = blockIdx.x * elemsPerBlock;
    if (C_block_start >= total) return;
    const int C_block_end   = min(C_block_start + elemsPerBlock, total);

    // global co-ranks for block boundaries
    __shared__ int A_bounds[2]; // (A_curr, A_next) for this block
    if (threadIdx.x == 0) {
        A_bounds[0] = co_rank(C_block_start, A, m, B, n);
        A_bounds[1] = co_rank(C_block_end,   A, m, B, n);
    }
    __syncthreads();

    const int A_block_start = A_bounds[0];
    const int A_block_end   = A_bounds[1];
    const int B_block_start = C_block_start - A_block_start;
    const int B_block_end   = C_block_end   - A_block_end;

    const int C_len = C_block_end - C_block_start;
    const int A_len = A_block_end - A_block_start;
    const int B_len = B_block_end - B_block_start;

    int C_done = 0;   // how many outputs produced for this block segment
    int A_done = 0;   // how many A elements consumed for this block segment
    int B_done = 0;   // how many B elements consumed for this block segment

    while (C_done < C_len) {
        // how many we will produce this iteration (last tile may be smaller)
        const int c_tile = min(tile_size, C_len - C_done);

        // how many inputs remain
        const int a_rem = A_len - A_done;
        const int b_rem = B_len - B_done;

        // load up to tile_size from A and B into shared memory
        // (we still *address* A_S/B_S up to tile_size, but only valid ranges are filled)
        for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
            if (i < a_rem) {
                A_S[i] = A[A_block_start + A_done + i];
            }
        }
        for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
            if (i < b_rem) {
                B_S[i] = B[B_block_start + B_done + i];
            }
        }
        __syncthreads();

        // effective tile lengths available in shared memory this iter
        const int a_tile = min(tile_size, a_rem);
        const int b_tile = min(tile_size, b_rem);

        // partition the c_tile outputs among threads
        const int chunk = (c_tile + blockDim.x - 1) / blockDim.x;
        int c_curr = threadIdx.x * chunk;
        int c_next = min(c_curr + chunk, c_tile);

        if (c_curr < c_next) {
            // co-ranks in the *shared* tiles
            int a_curr = co_rank(c_curr, A_S, a_tile, B_S, b_tile);
            int b_curr = c_curr - a_curr;

            int a_next = co_rank(c_next, A_S, a_tile, B_S, b_tile);
            int b_next = c_next - a_next;

            // each thread merges its own slice into global C
            merge_sequential(A_S + a_curr, a_next - a_curr,
                             B_S + b_curr, b_next - b_curr,
                             C + (C_block_start + C_done + c_curr));
        }

        __syncthreads();

        // Update how many A/B were consumed for producing c_tile outputs this iter.
        // co_rank(c_tile, ...) gives how many came from A in this tile.
        int a_used = co_rank(c_tile, A_S, a_tile, B_S, b_tile);
        int b_used = c_tile - a_used;

        A_done += a_used;
        B_done += b_used;
        C_done += c_tile;

        __syncthreads();
    }
}

// ---------------- simple CUDA error check ----------------
static void checkCuda(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(e) << "\n";
        std::exit(1);
    }
}

// ---------------- demo main ----------------
int main() {
    // Make two sorted arrays (with duplicates)
    int m = 1 << 12; // 4096
    int n = 1 << 12; // 4096

    std::mt19937 rng(123);
    std::uniform_int_distribution<int> dist(0, 5000);

    std::vector<int> hA(m), hB(n);
    for (int i = 0; i < m; ++i) hA[i] = dist(rng);
    for (int j = 0; j < n; ++j) hB[j] = dist(rng);
    std::sort(hA.begin(), hA.end());
    std::sort(hB.begin(), hB.end());

    // Device allocations
    int *dA = nullptr, *dB = nullptr, *dC = nullptr;
    checkCuda(cudaMalloc(&dA, m * sizeof(int)), "cudaMalloc dA");
    checkCuda(cudaMalloc(&dB, n * sizeof(int)), "cudaMalloc dB");
    checkCuda(cudaMalloc(&dC, (m + n) * sizeof(int)), "cudaMalloc dC");

    // Copy inputs
    checkCuda(cudaMemcpy(dA, hA.data(), m * sizeof(int), cudaMemcpyHostToDevice), "Memcpy A");
    checkCuda(cudaMemcpy(dB, hB.data(), n * sizeof(int), cudaMemcpyHostToDevice), "Memcpy B");

    // Launch params
    int threads   = 256;
    int blocks    = 120;   // demo
    int tile_size = 1024;  // ints per tile PER array in shared memory

    size_t shmemBytes = 2ULL * tile_size * sizeof(int);

    merge_tiled_kernel<<<blocks, threads, shmemBytes>>>(dA, m, dB, n, dC, tile_size);
    checkCuda(cudaGetLastError(), "Kernel launch");
    checkCuda(cudaDeviceSynchronize(), "Kernel sync");

    // Copy result back
    std::vector<int> hC(m + n);
    checkCuda(cudaMemcpy(hC.data(), dC, (m + n) * sizeof(int), cudaMemcpyDeviceToHost), "Memcpy C");

    // Verify vs std::merge
    std::vector<int> ref(m + n);
    std::merge(hA.begin(), hA.end(), hB.begin(), hB.end(), ref.begin());

    if (hC == ref) {
        std::cout << "Tiled merge correct! (m=" << m << ", n=" << n
                  << ", tile_size=" << tile_size << ")\n";
    } else {
        std::cerr << "Merge mismatch!\n";
        for (int i = 0; i < m + n; ++i) {
            if (hC[i] != ref[i]) {
                std::cerr << "First mismatch at i=" << i
                          << " gpu=" << hC[i] << " ref=" << ref[i] << "\n";
                break;
            }
        }
        return 1;
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return 0;
}