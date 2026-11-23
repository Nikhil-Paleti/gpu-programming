import torch 
import cutlass.cute as cute 
from cutlass.cute.runtime import from_dlpack 

@cute.kernel
def add_kernel(
        gA: cute.Tensor,
        gB: cute.Tensor,
        gC: cute.Tensor,
        tv_layout: cute.Layout
):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()

    block_coord = ((None, None), bid)
    block_a = gA[block_coord]
    block_b = gB[block_coord]
    block_c = gC[block_coord]

    tv_a = cute.composition(block_a, tv_layout)
    tv_b = cute.composition(block_b, tv_layout)
    tv_c = cute.composition(block_c, tv_layout)

    thread_a = tv_a[(tid, None)]
    thread_b = tv_b[(tid, None)]
    thread_c = tv_c[(tid, None)]

    thread_c.store(thread_a.load() + thread_b.load())
        
@cute.jit 
def launch_add(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
    thread_layout = cute.make_layout(
        (4, 64), stride=(64, 1)
    )

    value_layout = cute.make_layout(
        (1, 4), stride=(4, 1)
    )

    tile_shape, tv_layout = cute.make_layout_tv(
        thread_layout, value_layout
    )

    gA = cute.zipped_divide(a, tile_shape)
    gB = cute.zipped_divide(b, tile_shape)
    gC = cute.zipped_divide(c, tile_shape)

    add_kernel(gA, gB, gC, tv_layout).launch(
        grid = (cute.size(gC, mode=[1]), 1, 1),
        block=(cute.size(tv_layout, mode=[0]), 1, 1)
    )

def main():
    rows, cols = 128, 512

    a = torch.randn(rows, cols, device="cuda", dtype=torch.float32)
    b = torch.randn_like(a)
    c = torch.empty_like(a)

    launch_add(
        from_dlpack(a, assumed_align=16),
        from_dlpack(b, assumed_align=16),
        from_dlpack(c, assumed_align=16),
    )

    torch.cuda.synchronize()
    torch.testing.assert_close(c, a + b)
    print("Passed: tiled addition matches PyTorch")


if __name__ == "__main__":
    main()
