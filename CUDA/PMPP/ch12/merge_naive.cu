#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include <random>
#include <cassert>


__host__ __device__ __forceinline__
int co_rank(int k, const int* A, int m, const int* B, int n) {
    int i_low  = max(0, k - n);
    int i_high = min(k, m);

    while (true) {
        int i = (i_low + i_high) >> 1;
        int j = k - i;

        // Took too many from A: move i left
        if (i > 0 && j < n && A[i - 1] > B[j]) {
            i_high = i - 1;
        }
        // Took too few from A: move i right (>= keeps merge stable: A before B on ties)
        else if (j > 0 && i < m && B[j - 1] >= A[i]) {
            i_low = i + 1;
        }
        else {
            return i;
        }
    }
}

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

__global__ void merge_basic_kernel(const int* A, int m,
                                   const int* B, int n,
                                   int* C) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x;
    int total = m + n;

    // ceil(total / totalThreads)
    int elemsPerThread = (total + totalThreads - 1) / totalThreads;

    int k_curr = tid * elemsPerThread;
    if (k_curr >= total) return;

    int k_next = min(k_curr + elemsPerThread, total);

    int i_curr = co_rank(k_curr, A, m, B, n);
    int i_next = co_rank(k_next, A, m, B, n);

    int j_curr = k_curr - i_curr;
    int j_next = k_next - i_next;

    merge_sequential(A + i_curr, i_next - i_curr,
                     B + j_curr, j_next - j_curr,
                     C + k_curr);
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
    // Make two sorted arrays (with duplicates to test stability/ties)
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

    // Launch
    int threads = 256;
    int blocks  = 120;  // arbitrary; for demo. (Too many threads => elemsPerThread small => co_rank overhead bigger)
    merge_basic_kernel<<<blocks, threads>>>(dA, m, dB, n, dC);
    checkCuda(cudaGetLastError(), "Kernel launch");
    checkCuda(cudaDeviceSynchronize(), "Kernel sync");

    // Copy result back
    std::vector<int> hC(m + n);
    checkCuda(cudaMemcpy(hC.data(), dC, (m + n) * sizeof(int), cudaMemcpyDeviceToHost), "Memcpy C");

    // Verify vs std::merge
    std::vector<int> ref(m + n);
    std::merge(hA.begin(), hA.end(), hB.begin(), hB.end(), ref.begin());

    if (hC == ref) {
        std::cout << "Merge correct! (m=" << m << ", n=" << n << ")\n";
    } else {
        std::cerr << "Merge mismatch!\n";
        // Print a small mismatch region to debug
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