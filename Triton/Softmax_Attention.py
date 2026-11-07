import torch
import triton
import triton.language as tl


@triton.jit
def softmax_attention(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    q_len, kv_len, head_dim,
    Q_stride_row, Q_stride_col,
    K_stride_row, K_stride_col,
    V_stride_row, V_stride_col,
    O_stride_row, O_stride_col,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # Program ids: which block of query rows / head-dim cols this program handles
    pid_q = tl.program_id(0)
    pid_h = tl.program_id(1)

    # Tiled indices
    q_rows = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)         # [BLOCK_Q]
    h_cols = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)         # [BLOCK_H]
    k_cols = tl.arange(0, BLOCK_K)                           # [BLOCK_K]

    # Bounds masks
    q_mask = q_rows < q_len
    h_mask = h_cols < head_dim

    # Output accumulator and running softmax state (online softmax)
    out_accum = tl.zeros((BLOCK_Q, BLOCK_H), dtype=tl.float32)
    sm_running_sum = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    sm_running_max = tl.full((BLOCK_Q,), float("-inf"), dtype=tl.float32)

    # Scale for dot(Q, K) / sqrt(d)
    scale = 1.0 / tl.sqrt(head_dim + 0.0)

    # Sweep over keys/values in BLOCK_K chunks
    for k_start in range(0, kv_len, BLOCK_K):
        k_block_cols = k_start + k_cols                      # [BLOCK_K]
        k_block_mask = k_block_cols < kv_len

        # Compute attention logits for this (Q-block x K-block)
        attn_scores_block = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)

        # Reduce over head_dim in BLOCK_H chunks
        for h_start in range(0, head_dim, BLOCK_H):
            h_block = h_start + tl.arange(0, BLOCK_H)        # [BLOCK_H]
            h_block_mask = h_block < head_dim

            # Q slice: shape [BLOCK_Q, BLOCK_H]
            Q_offsets = (
                q_rows[:, None] * Q_stride_row
                + h_block[None, :] * Q_stride_col
            )
            Q_mask = q_mask[:, None] & h_block_mask[None, :]
            Q_tile = tl.load(Q_ptr + Q_offsets, mask=Q_mask, other=0.0)

            # K slice: K is [head_dim, kv_len] → rows=head_dim, cols=kv_len
            K_offsets = (
                h_block[:, None] * K_stride_row
                + k_block_cols[None, :] * K_stride_col
            )
            K_mask = h_block_mask[:, None] & k_block_mask[None, :]
            K_tile = tl.load(K_ptr + K_offsets, mask=K_mask, other=0.0)

            # Accumulate Q · K for this head_dim tile
            attn_scores_block += tl.dot(Q_tile, K_tile)

        # Scale logits
        attn_scores_block = attn_scores_block * scale

        # Mask out-of-bounds rows/cols
        attn_mask = q_mask[:, None] & k_block_mask[None, :]
        attn_scores_block = tl.where(attn_mask, attn_scores_block, float("-inf"))

        # Online softmax update
        block_row_max = tl.max(attn_scores_block, axis=1)                # [BLOCK_Q]
        new_row_max = tl.maximum(block_row_max, sm_running_max)           # [BLOCK_Q]
        exp_scale = tl.exp(sm_running_max - new_row_max)                  # [BLOCK_Q]
        sm_running_max = new_row_max

        scores_shifted = attn_scores_block - new_row_max[:, None]
        probs_block = tl.exp(scores_shifted)                              # [BLOCK_Q, BLOCK_K]
        block_den = tl.sum(probs_block, axis=1)                           # [BLOCK_Q]

        sm_running_sum = tl.fma(sm_running_sum, exp_scale, block_den)     # [BLOCK_Q]

        # Load V slice: V is [kv_len, head_dim]
        V_offsets = (
            k_block_cols[:, None] * V_stride_row
            + h_cols[None, :] * V_stride_col
        )
        V_mask = k_block_mask[:, None] & h_mask[None, :]
        V_tile = tl.load(V_ptr + V_offsets, mask=V_mask, other=0.0)       # [BLOCK_K, BLOCK_H]

        # Accumulate probabilities * V, with online scaling
        weighted_v = tl.dot(probs_block, V_tile)                          # [BLOCK_Q, BLOCK_H]
        out_accum = tl.fma(out_accum, exp_scale[:, None], weighted_v)

    # Normalize by softmax denominator
    out_accum /= sm_running_sum[:, None]

    # Store output O: shape [q_len, head_dim]
    O_offsets = (
        q_rows[:, None] * O_stride_row
        + h_cols[None, :] * O_stride_col
    )
    O_mask = q_mask[:, None] & h_mask[None, :]
    tl.store(O_ptr + O_offsets, out_accum.to(tl.float16), mask=O_mask)


# Q, K, V, O are tensors on the GPU:
# Q: [q_len, head_dim]
# K: [head_dim, kv_len]
# V: [kv_len, head_dim]
# O: [q_len, head_dim]
def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, O: torch.Tensor,
          q_len: int, kv_len: int, head_dim: int):
    BLOCK_Q = 16
    BLOCK_H = 128
    BLOCK_K = 64

    grid = (triton.cdiv(q_len, BLOCK_Q), triton.cdiv(head_dim, BLOCK_H))

    Q_stride_row, Q_stride_col = Q.stride()
    K_stride_row, K_stride_col = K.stride()
    V_stride_row, V_stride_col = V.stride()
    O_stride_row, O_stride_col = O.stride()

    softmax_attention[grid](
        Q_ptr=Q, K_ptr=K, V_ptr=V, O_ptr=O,
        q_len=q_len, kv_len=kv_len, head_dim=head_dim,
        Q_stride_row=Q_stride_row, Q_stride_col=Q_stride_col,
        K_stride_row=K_stride_row, K_stride_col=K_stride_col,
        V_stride_row=V_stride_row, V_stride_col=V_stride_col,
        O_stride_row=O_stride_row, O_stride_col=O_stride_col,
        BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, BLOCK_H=BLOCK_H,
        num_warps=4
    )