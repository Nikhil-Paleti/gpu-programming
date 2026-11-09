import torch 
import cutlass.cute as cute 
from cutlass.cute.runtime import from_dlpack 

@cute.kernel
def add_kernel(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
    tid_x, _, _ = cute.arch.thread_idx()
    bid_x, _, _ = cute.arch.block_idx()
    block_size_x, _, _ = cute.arch.block_dim()

    i = tid_x + bid_x * block_size_x
    rows, cols = a.shape 

    if (i < rows * cols):
        # row = i // cols 
        # col = i % cols 
        c[i] = a[i] + b[i]
        # c[row, col] = a[row, col] + b[row, col]

@cute.jit 
def launch_add(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
    threads_per_block = 256
    rows, cols = a.shape
    n = rows * cols 
    num_blocks = (n + threads_per_block - 1) // threads_per_block
    add_kernel(a, b, c).launch(
        grid = [num_blocks, 1, 1],
        block = [threads_per_block, 1, 1]
    )

def main():
    M, N = 16, 256 

    a = torch.randn(M, N, device="cuda", dtype=torch.float32)
    b = torch.randn(M, N, device="cuda", dtype=torch.float32)
    c = torch.empty_like(a)
    launch_add(
        from_dlpack(a),
        from_dlpack(b),
        from_dlpack(c)
    )
    torch.cuda.synchronize()
    expected = a+b 
    torch.testing.assert_close(expected, c)
    print("passed 2d add kernel")

if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Why a[i] and a[row, col] assign different elements to threads in CuTe
#
# Consider a contiguous, row-major tensor with shape (2, 4):
#
#                 col 0   col 1   col 2   col 3
#     row 0         A       B       C       D
#     row 1         E       F       G       H
#
# Its physical memory order is:
#     offset:       0   1   2   3   4   5   6   7
#     element:      A   B   C   D   E   F   G   H
#
# 1. Explicit row-major coordinates
#
#     row = i // cols
#     col = i % cols
#     c[row, col] = a[row, col] + b[row, col]
#
#     thread i:     0       1       2       3       4
#     coordinate:  (0,0)   (0,1)   (0,2)   (0,3)   (1,0)
#     element:      A       B       C       D       E
#     offset:       0       1       2       3       4
#
# Neighboring threads access neighboring columns and memory locations. This
# favors coalesced global-memory access for these row-major tensors.
#
# 2. A scalar index into a multidimensional CuTe tensor
#
#     c[i] = a[i] + b[i]
#
# Here i is a logical coordinate interpreted through the tensor layout, not
# a raw memory offset. CuTe expands the scalar coordinate with the first
# shape dimension varying fastest. For a flat shape (rows, cols), this is:
#
#     row = i % rows
#     col = i // rows
#     memory_offset = row * cols + col  # for this row-major layout
#
# For the same (2, 4) tensor:
#     thread i:     0       1       2       3       4
#     coordinate:  (0,0)   (1,0)   (0,1)   (1,1)   (0,2)
#     element:      A       E       B       F       C
#     offset:       0       4       1       5       2
#
# Both versions can compute the correct result: each valid element is visited
# once, and A, B, and C use matching coordinates. What changes is which thread
# owns each element and therefore the memory-access pattern.
#
# In this file's (16, 256) example, threads 0 through 15 access successive rows
# of column 0 with scalar indexing: their offsets are 0, 256, 512, ..., 3840
# floats. Explicit row-major coordinates instead give offsets 0, 1, 2, ..., 15.
# Correctness alone therefore does not establish efficient memory access.
#
# To use the explicit row-major mapping, replace the current c[i] assignment
# inside the bounds check with the row/col calculation and assignment above.
# A shape such as (37, 29) also exercises the extra-thread bounds check; the
# current 16 * 256 elements divide evenly into 256-thread blocks.
#
# CuTe coordinate conversion reference:
# https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute.html#cutlass.cute.idx2crd
# ---------------------------------------------------------------------------
