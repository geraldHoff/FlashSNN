import torch
import triton
import triton.language as tl


# Forward kernel: out = Q @ (K^T @ V),  tiled over L for both Q and K/V
@triton.jit
def _sdsa2_forward_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    Out_ptr,
    L: tl.constexpr,
    Cph: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_Cph: tl.constexpr,    # >= Cph, next power of 2
    # Compute dtype
    dtype: tl.constexpr,
):

    # Program IDs
    bh = tl.program_id(0)          # batch * num_heads index (flattened)
    q_block_id = tl.program_id(2)  # which BLOCK_L tile of Q/output
    q_start = q_block_id * BLOCK_L

    # Base pointers for this (batch, head) pair
    # Each tensor is [batch_heads, L, Cph] row-major, so stride is L*Cph per batch_head
    stride_bh = L * Cph
    q_base = Q_ptr + bh * stride_bh
    k_base = K_ptr + bh * stride_bh
    v_base = V_ptr + bh * stride_bh
    out_base = Out_ptr + bh * stride_bh

    KtV = tl.zeros((BLOCK_Cph, BLOCK_Cph), dtype=dtype)

    for l_start in tl.static_range(0, L, BLOCK_L):
        k_ptrs = tl.make_block_ptr(
            k_base,
            shape=(Cph, L),
            strides=(1, Cph),
            offsets=(0, l_start),
            block_shape=(BLOCK_Cph, BLOCK_L),
            order=(1, 0),
        )
        k_block = tl.load(k_ptrs, boundary_check=(0, 1), padding_option="zero")

        v_ptrs = tl.make_block_ptr(
            v_base,
            shape=(L, Cph),
            strides=(Cph, 1),
            offsets=(l_start, 0),
            block_shape=(BLOCK_L, BLOCK_Cph),
            order=(1, 0),
        )
        v_block = tl.load(v_ptrs, boundary_check=(0, 1), padding_option="zero")

        # Accumulate: KtV += K_block^T @ V_block = [BLOCK_Cph, BLOCK_L] @ [BLOCK_L, BLOCK_Cph]
        KtV = tl.dot(
            k_block,
            v_block,
            acc=KtV,
            input_precision="ieee",
            out_dtype=dtype,
        )

    q_ptrs = tl.make_block_ptr(
        q_base,
        shape=(L, Cph),
        strides=(Cph, 1),
        offsets=(q_start, 0),
        block_shape=(BLOCK_L, BLOCK_Cph),
        order=(1, 0),
    )
    q_block = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")
    out_tile = tl.dot(
        q_block,    # [BLOCK_L, BLOCK_Cph]
        KtV,        # [BLOCK_Cph, BLOCK_Cph]
        input_precision="ieee",
        out_dtype=dtype,
    )

    out_ptrs = tl.make_block_ptr(
        out_base,
        shape=(L, Cph),
        strides=(Cph, 1),
        offsets=(q_start, 0),
        block_shape=(BLOCK_L, BLOCK_Cph),
        order=(1, 0),
    )
    tl.store(out_ptrs, out_tile, boundary_check=(0, 1))

@triton.jit
def _sdsa2_fused_proj_spike_attn_kernel(
    S_ptr,
    Wq_ptr, Wk_ptr, Wv_ptr,
    Vmem_ptr,
    Out_ptr,
    threshold,
    L: tl.constexpr,
    D: tl.constexpr,
    Cph: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_Cph: tl.constexpr,
    BLOCK_D: tl.constexpr,
    dtype: tl.constexpr,
):

    bh = tl.program_id(0)
    q_block_id = tl.program_id(2)
    q_start = q_block_id * BLOCK_L

    stride_s = L * D
    stride_out = L * Cph
    stride_vmem = 3 * Cph

    s_base = S_ptr + bh * stride_s
    out_base = Out_ptr + bh * stride_out
    vmem_base = Vmem_ptr + bh * stride_vmem

    KtV = tl.zeros((BLOCK_Cph, BLOCK_Cph), dtype=dtype)

    for l_start in tl.static_range(0, L, BLOCK_L):
        k_proj = tl.zeros((BLOCK_L, BLOCK_Cph), dtype=dtype)
        v_proj = tl.zeros((BLOCK_L, BLOCK_Cph), dtype=dtype)

        for d_start in tl.static_range(0, D, BLOCK_D):
            s_ptrs = tl.make_block_ptr(
                s_base,
                shape=(L, D),
                strides=(D, 1),
                offsets=(l_start, d_start),
                block_shape=(BLOCK_L, BLOCK_D),
                order=(1, 0),
            )
            s_tile = tl.load(s_ptrs, boundary_check=(0, 1), padding_option="zero")

            wk_ptrs = tl.make_block_ptr(
                Wk_ptr,
                shape=(D, Cph),
                strides=(Cph, 1),
                offsets=(d_start, 0),
                block_shape=(BLOCK_D, BLOCK_Cph),
                order=(1, 0),
            )
            wk_tile = tl.load(wk_ptrs, boundary_check=(0, 1), padding_option="zero")

            wv_ptrs = tl.make_block_ptr(
                Wv_ptr,
                shape=(D, Cph),
                strides=(Cph, 1),
                offsets=(d_start, 0),
                block_shape=(BLOCK_D, BLOCK_Cph),
                order=(1, 0),
            )
            wv_tile = tl.load(wv_ptrs, boundary_check=(0, 1), padding_option="zero")

            # Accumulate projections
            k_proj = tl.dot(s_tile, wk_tile, acc=k_proj,
                            input_precision="ieee", out_dtype=dtype)
            v_proj = tl.dot(s_tile, wv_tile, acc=v_proj,
                            input_precision="ieee", out_dtype=dtype)

        #LiF
        k_spike = (k_proj > 0.0).to(dtype)   # [BLOCK_L, BLOCK_Cph]
        v_spike = (v_proj > 0.0).to(dtype)   # [BLOCK_L, BLOCK_Cph]

        # Accumulate KtV += K_spike^T @ V_spike
        # K_spike^T: [BLOCK_Cph, BLOCK_L], V_spike: [BLOCK_L, BLOCK_Cph]
        KtV = tl.dot(
            tl.trans(k_spike),  # [BLOCK_Cph, BLOCK_L]
            v_spike,            # [BLOCK_L, BLOCK_Cph]
            acc=KtV,
            input_precision="ieee",
            out_dtype=dtype,
        )

    q_proj = tl.zeros((BLOCK_L, BLOCK_Cph), dtype=dtype)

    for d_start in tl.static_range(0, D, BLOCK_D):
        s_ptrs = tl.make_block_ptr(
            s_base,
            shape=(L, D),
            strides=(D, 1),
            offsets=(q_start, d_start),
            block_shape=(BLOCK_L, BLOCK_D),
            order=(1, 0),
        )
        s_tile = tl.load(s_ptrs, boundary_check=(0, 1), padding_option="zero")

        wq_ptrs = tl.make_block_ptr(
            Wq_ptr,
            shape=(D, Cph),
            strides=(Cph, 1),
            offsets=(d_start, 0),
            block_shape=(BLOCK_D, BLOCK_Cph),
            order=(1, 0),
        )
        wq_tile = tl.load(wq_ptrs, boundary_check=(0, 1), padding_option="zero")

        q_proj = tl.dot(s_tile, wq_tile, acc=q_proj,
                        input_precision="ieee", out_dtype=dtype)

    # LIF spike for Q
    q_spike = (q_proj > 0.0).to(dtype)  # [BLOCK_L, BLOCK_Cph]

    # Output = Q_spike @ KtV
    out_tile = tl.dot(
        q_spike,    # [BLOCK_L, BLOCK_Cph]
        KtV,        # [BLOCK_Cph, BLOCK_Cph]
        input_precision="ieee",
        out_dtype=dtype,
    )

    # Store output
    out_ptrs = tl.make_block_ptr(
        out_base,
        shape=(L, Cph),
        strides=(Cph, 1),
        offsets=(q_start, 0),
        block_shape=(BLOCK_L, BLOCK_Cph),
        order=(1, 0),
    )
    tl.store(out_ptrs, out_tile, boundary_check=(0, 1))


# Python wrapper functions
def sdsa2_forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                  BLOCK_L: int = 64, BLOCK_Cph: int = 64) -> torch.Tensor:
    import math

    assert Q.shape == K.shape == V.shape
    N, H, L, Cph = Q.shape
    assert Q.is_cuda and Q.is_contiguous()

    out = torch.empty_like(Q)

    # Flatten N and H into single batch dim
    Q_flat = Q.reshape(N * H, L, Cph).contiguous()
    K_flat = K.reshape(N * H, L, Cph).contiguous()
    V_flat = V.reshape(N * H, L, Cph).contiguous()
    out_flat = out.reshape(N * H, L, Cph)

    grid = (N * H, 1, math.ceil(L / BLOCK_L))

    _sdsa2_forward_kernel[grid](
        Q_flat, K_flat, V_flat, out_flat,
        L=L,
        Cph=Cph,
        BLOCK_L=BLOCK_L,
        BLOCK_Cph=BLOCK_Cph,
        dtype=tl.float32,
    )
    return out_flat.reshape(N, H, L, Cph)


def sdsa2_fused_forward(S: torch.Tensor,
                        Wq: torch.Tensor, Wk: torch.Tensor, Wv: torch.Tensor,
                        num_heads: int,
                        threshold: float = 0.0,
                        BLOCK_L: int = 64, BLOCK_Cph: int = 64,
                        BLOCK_D: int = 64) -> torch.Tensor:
    import math

    N, L, D = S.shape
    Cph = D // num_heads
    H = num_heads

    out = torch.empty(N, H, L, Cph, dtype=S.dtype, device=S.device)
    Vmem = torch.zeros(N * H, 3, Cph, dtype=S.dtype, device=S.device)

    for h in range(H):
        # Slice weights for this head
        wq_h = Wq[:, h * Cph:(h + 1) * Cph].contiguous()  # [D, Cph]
        wk_h = Wk[:, h * Cph:(h + 1) * Cph].contiguous()
        wv_h = Wv[:, h * Cph:(h + 1) * Cph].contiguous()

        out_h = out[:, h, :, :].contiguous()  # [N, L, Cph]

        grid = (N, 1, math.ceil(L / BLOCK_L))

        _sdsa2_fused_proj_spike_attn_kernel[grid](
            S.contiguous(),
            wq_h, wk_h, wv_h,
            Vmem[:, :, :],
            out_h,
            threshold,
            L=L, D=D, Cph=Cph,
            BLOCK_L=BLOCK_L,
            BLOCK_Cph=BLOCK_Cph,
            BLOCK_D=BLOCK_D,
            dtype=tl.float32,
        )
        out[:, h, :, :] = out_h

    return out
