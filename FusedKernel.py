import math
import torch
import triton
import triton.language as tl


# --------------------------------------------------------------------------
# Kernel 1: attention-only (pre-spiked Q, K, V inputs)
#
# SDSA v2 operation per the paper (Spike-driven Transformer, Yao et al.):
#   ktv = spike(sum_over_L(K ⊙ V))    →  [Cph] binary vector
#   Out = Q * ktv[None, :]              →  [L, Cph] broadcast multiply
#
# K ⊙ V is the Hadamard (element-wise) product, NOT K^T @ V.
# The sum over L counts per-channel co-occurrences, then the result is
# spiked to produce a binary gating vector that is broadcast-multiplied
# with every row of Q.
#
# One program per (batch, head).
# --------------------------------------------------------------------------

@triton.jit
def _sdsa2_forward_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    L: tl.constexpr,
    Cph: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_Cph: tl.constexpr,
    dtype: tl.constexpr,
):
    bh = tl.program_id(0)
    stride_bh = L * Cph

    k_base = K_ptr + bh * stride_bh
    v_base = V_ptr + bh * stride_bh
    q_base = Q_ptr + bh * stride_bh
    out_base = Out_ptr + bh * stride_bh

    cph_offs = tl.arange(0, BLOCK_Cph)
    ktv_acc = tl.zeros((BLOCK_Cph,), dtype=dtype)

    for l_start in tl.static_range(0, L, BLOCK_L):
        l_offs = l_start + tl.arange(0, BLOCK_L)
        mask = (l_offs[:, None] < L) & (cph_offs[None, :] < Cph)

        k_ptrs = k_base + l_offs[:, None] * Cph + cph_offs[None, :]
        v_ptrs = v_base + l_offs[:, None] * Cph + cph_offs[None, :]

        k_block = tl.load(k_ptrs, mask=mask, other=0.0).to(dtype)
        v_block = tl.load(v_ptrs, mask=mask, other=0.0).to(dtype)

        kv_prod = k_block * v_block
        ktv_acc += tl.sum(kv_prod, axis=0)

    ktv_spiked = (ktv_acc > 0.0).to(dtype)

    for q_start in tl.static_range(0, L, BLOCK_L):
        q_offs = q_start + tl.arange(0, BLOCK_L)
        mask = (q_offs[:, None] < L) & (cph_offs[None, :] < Cph)

        q_ptrs = q_base + q_offs[:, None] * Cph + cph_offs[None, :]
        q_block = tl.load(q_ptrs, mask=mask, other=0.0).to(dtype)

        out_tile = q_block * ktv_spiked[None, :]

        out_ptrs = out_base + q_offs[:, None] * Cph + cph_offs[None, :]
        tl.store(out_ptrs, out_tile, mask=mask)


# --------------------------------------------------------------------------
# Kernel 2: fused projection → spike → Hadamard attention
#
# One program per (batch, head). K and V projections are split into
# separate D-loops to keep only one [BLOCK_L, BLOCK_Cph] projection
# accumulator live at a time.  D-loops use tl.range (not static_range)
# to prevent full unrolling and register explosion.
# --------------------------------------------------------------------------

@triton.jit
def _sdsa2_fused_proj_spike_attn_kernel(
    S_ptr,
    Wq_ptr, Wk_ptr, Wv_ptr,
    Out_ptr,
    L: tl.constexpr,
    D: tl.constexpr,
    Cph: tl.constexpr,
    H: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_Cph: tl.constexpr,
    BLOCK_D: tl.constexpr,
    dtype: tl.constexpr,
):
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)

    s_base = S_ptr + batch_id * L * D
    head_offset = head_id * Cph
    out_base = Out_ptr + (batch_id * H + head_id) * L * Cph

    # ---- Phase 1: ktv_acc = sum_L(spike(S@Wk) ⊙ spike(S@Wv)) ----
    ktv_acc = tl.zeros((BLOCK_Cph,), dtype=dtype)

    for l_start in tl.static_range(0, L, BLOCK_L):

        # Step A: project K for this L-tile
        k_proj = tl.zeros((BLOCK_L, BLOCK_Cph), dtype=dtype)
        for d_start in tl.range(0, D, BLOCK_D):
            s_ptrs = tl.make_block_ptr(
                s_base,
                shape=(L, D), strides=(D, 1),
                offsets=(l_start, d_start),
                block_shape=(BLOCK_L, BLOCK_D), order=(1, 0),
            )
            s_tile = tl.load(s_ptrs, boundary_check=(0, 1), padding_option="zero")

            wk_ptrs = tl.make_block_ptr(
                Wk_ptr,
                shape=(D, H * Cph), strides=(H * Cph, 1),
                offsets=(d_start, head_offset),
                block_shape=(BLOCK_D, BLOCK_Cph), order=(1, 0),
            )
            wk_tile = tl.load(wk_ptrs, boundary_check=(0, 1), padding_option="zero")

            k_proj = tl.dot(s_tile, wk_tile, acc=k_proj,
                            input_precision="ieee", out_dtype=dtype)

        k_spike = (k_proj > 0.0).to(dtype)

        # Step B: project V for this L-tile, then combine with K
        v_proj = tl.zeros((BLOCK_L, BLOCK_Cph), dtype=dtype)
        for d_start in tl.range(0, D, BLOCK_D):
            s_ptrs = tl.make_block_ptr(
                s_base,
                shape=(L, D), strides=(D, 1),
                offsets=(l_start, d_start),
                block_shape=(BLOCK_L, BLOCK_D), order=(1, 0),
            )
            s_tile = tl.load(s_ptrs, boundary_check=(0, 1), padding_option="zero")

            wv_ptrs = tl.make_block_ptr(
                Wv_ptr,
                shape=(D, H * Cph), strides=(H * Cph, 1),
                offsets=(d_start, head_offset),
                block_shape=(BLOCK_D, BLOCK_Cph), order=(1, 0),
            )
            wv_tile = tl.load(wv_ptrs, boundary_check=(0, 1), padding_option="zero")

            v_proj = tl.dot(s_tile, wv_tile, acc=v_proj,
                            input_precision="ieee", out_dtype=dtype)

        v_spike = (v_proj > 0.0).to(dtype)

        # Hadamard product + reduce over L-tile
        ktv_acc += tl.sum(k_spike * v_spike, axis=0)

    # Spike the accumulated channel-wise counts
    ktv_spiked = (ktv_acc > 0.0).to(dtype)

    # ---- Phase 2: project Q, spike, broadcast-multiply ----
    for q_start in tl.static_range(0, L, BLOCK_L):
        q_proj = tl.zeros((BLOCK_L, BLOCK_Cph), dtype=dtype)

        for d_start in tl.range(0, D, BLOCK_D):
            s_ptrs = tl.make_block_ptr(
                s_base,
                shape=(L, D), strides=(D, 1),
                offsets=(q_start, d_start),
                block_shape=(BLOCK_L, BLOCK_D), order=(1, 0),
            )
            s_tile = tl.load(s_ptrs, boundary_check=(0, 1), padding_option="zero")

            wq_ptrs = tl.make_block_ptr(
                Wq_ptr,
                shape=(D, H * Cph), strides=(H * Cph, 1),
                offsets=(d_start, head_offset),
                block_shape=(BLOCK_D, BLOCK_Cph), order=(1, 0),
            )
            wq_tile = tl.load(wq_ptrs, boundary_check=(0, 1), padding_option="zero")

            q_proj = tl.dot(s_tile, wq_tile, acc=q_proj,
                            input_precision="ieee", out_dtype=dtype)

        q_spike = (q_proj > 0.0).to(dtype)
        out_tile = q_spike * ktv_spiked[None, :]

        out_ptrs = tl.make_block_ptr(
            out_base,
            shape=(L, Cph), strides=(Cph, 1),
            offsets=(q_start, 0),
            block_shape=(BLOCK_L, BLOCK_Cph), order=(1, 0),
        )
        tl.store(out_ptrs, out_tile, boundary_check=(0, 1))


# --------------------------------------------------------------------------
# Python wrappers
# --------------------------------------------------------------------------

def sdsa2_forward(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
    BLOCK_L: int = 64, BLOCK_Cph: int = 64,
) -> torch.Tensor:
    assert Q.shape == K.shape == V.shape
    assert Q.is_cuda and Q.is_contiguous()
    N, H, L, Cph = Q.shape

    out = torch.empty_like(Q)
    Q_flat = Q.reshape(N * H, L, Cph)
    K_flat = K.reshape(N * H, L, Cph)
    V_flat = V.reshape(N * H, L, Cph)
    out_flat = out.reshape(N * H, L, Cph)

    grid = (N * H,)

    _sdsa2_forward_kernel[grid](
        Q_flat, K_flat, V_flat, out_flat,
        L=L, Cph=Cph,
        BLOCK_L=BLOCK_L, BLOCK_Cph=BLOCK_Cph,
        dtype=tl.float32,
    )
    return out


def sdsa2_fused_forward(
    S: torch.Tensor,
    Wq: torch.Tensor, Wk: torch.Tensor, Wv: torch.Tensor,
    num_heads: int,
    BLOCK_L: int = 64, BLOCK_Cph: int = 64, BLOCK_D: int = 64,
) -> torch.Tensor:
    N, L, D = S.shape
    H = num_heads
    Cph = D // H

    assert S.is_cuda
    assert Wq.shape == Wk.shape == Wv.shape == (D, D)

    S = S.contiguous()
    Wq = Wq.contiguous()
    Wk = Wk.contiguous()
    Wv = Wv.contiguous()

    out = torch.empty(N, H, L, Cph, dtype=S.dtype, device=S.device)

    grid = (N, H)

    _sdsa2_fused_proj_spike_attn_kernel[grid](
        S,
        Wq, Wk, Wv,
        out,
        L=L, D=D, Cph=Cph, H=H,
        BLOCK_L=BLOCK_L, BLOCK_Cph=BLOCK_Cph, BLOCK_D=BLOCK_D,
        dtype=tl.float32,
    )
    return out


# --------------------------------------------------------------------------
# Reference implementations (pure PyTorch)
# --------------------------------------------------------------------------

def sdsa2_reference(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    SDSA v2 reference: Hadamard product reduced over sequence, then spike + broadcast.
    Q, K, V: [N, H, L, Cph] binary
    """
    ktv = (K * V).sum(dim=2)             # [N, H, Cph]
    ktv_spiked = (ktv > 0.0).float()     # [N, H, Cph] binary
    return Q * ktv_spiked.unsqueeze(2)   # [N, H, L, Cph] broadcast


def sdsa2_fused_reference(
    S: torch.Tensor,
    Wq: torch.Tensor, Wk: torch.Tensor, Wv: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    """
    Fused reference: project, spike, Hadamard attend.
    S: [N, L, D], Wq/Wk/Wv: [D, D]
    """
    N, L, D = S.shape
    H = num_heads
    Cph = D // H

    Q = (S @ Wq).reshape(N, L, H, Cph).permute(0, 2, 1, 3)
    K = (S @ Wk).reshape(N, L, H, Cph).permute(0, 2, 1, 3)
    V = (S @ Wv).reshape(N, L, H, Cph).permute(0, 2, 1, 3)

    Q = (Q > 0.0).float()
    K = (K > 0.0).float()
    V = (V > 0.0).float()

    return sdsa2_reference(Q, K, V)
