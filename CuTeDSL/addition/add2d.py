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


# CuTe's a[i] is a logical index, not a raw memory offset.
# For shape (rows, cols), it maps to (i % rows, i // rows).
# Explicit row-major indexing uses (i // cols, i % cols) instead.
#
# Example: shape (2, 4), with rows [A B C D] and [E F G H]:
#   a[i] visits:       A E B F C G D H
#   row-major visits: A B C D E F G H
#
# Both give correct sums because A, B, and C use matching coordinates.
# Explicit row-major indexing lets neighboring threads access neighboring
# memory locations, favoring coalesced access for these tensors.
