import torch 
import cutlass.cute as cute 
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def add_kernel(a: cute.Tensor, b:cute.Tensor, c: cute.Tensor):
    thread_x , _, _ = cute.arch.thread_idx()
    block_x, _, _ = cute.arch.block_idx()
    block_size_x, _, _ = cute.arch.block_dim()

    i = thread_x + block_x * block_size_x
    if (i < a.shape[0]):
        c[i] = a[i] + b[i]

@cute.jit 
def launch_add(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
    threads_per_block = 256
    n = a.shape[0]
    num_blocks = (n + threads_per_block - 1) // threads_per_block
    add_kernel(a,b,c).launch(
        grid=(num_blocks, 1, 1),
        block=(threads_per_block, 1, 1)
    )

def main():
    n: int = 1000

    a = torch.randn(n, device="cuda", dtype=torch.float32)
    b = torch.randn(n, device="cuda", dtype=torch.float32)
    c = torch.empty_like(a)

    launch_add(
        from_dlpack(a),
        from_dlpack(b),
        from_dlpack(c)
    )

    torch.cuda.synchronize()

    expected = a+b 
    torch.testing.assert_close(c, expected)
    print("Passed: GPU kernel matches PyTorch")

if __name__ == "__main__":
    main()
