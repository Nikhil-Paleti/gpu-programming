import torch
import cutlass.cute as cute 
from cutlass.cute.runtime import from_dlpack 

@cute.kernel 
def add_kernel(a: cute.Tensor, b: cute.Tensor, c:cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    block_x, _, _ = cute.arch.block_dim()

    elements_per_thread = 4 
    rows, cols = a.shape 

    block_start = bidx * block_x * elements_per_thread 

    for v in range(elements_per_thread):
        i = block_start + tidx + v * block_x 

        if i < rows * cols: 
            row = i // cols 
            col = i % cols 
            c[row, col] = a[row, col] + b[row, col]

@cute.jit 
def launch_add(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
    threads_per_block = 256
    elements_per_thread = 4 

    rows, cols = a.shape 
    elements_per_block = threads_per_block * elements_per_thread

    num_blocks = (rows * cols + elements_per_block - 1) // elements_per_block

    add_kernel(a, b, c).launch(
        grid=(num_blocks, 1, 1),
        block=(threads_per_block, 1, 1)
    )

def main():
    rows, cols = 37, 29

    a = torch.randn(rows, cols, device="cuda", dtype=torch.float32)
    b = torch.randn(rows, cols, device="cuda", dtype=torch.float32)
    c = torch.empty_like(a) 

    launch_add(
        from_dlpack(a),
        from_dlpack(b),
        from_dlpack(c)
    )
    
    torch.cuda.synchronize()
    torch.testing.assert_close(c, a+b)
    print("passed validation")

if __name__ == "__main__":
    main()
