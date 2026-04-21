import math
import torch
import triton
import triton.language as tl


#The grid is (N*H,) instead of (N*H, 1, ceil(L/BLOCK_L))
#Q-tile loop moved inside the kernel
#single program per (batch, head) computes KtV once into registers, 
#then sequentially loops over the Q tiles to emit output. 
#The KtV matrix lives in registers the entire time and is never recomputed
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
 
    # Phase 1: accumulate KtV = K^T @ V  [Cph, Cph] — computed ONCE
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
 
        KtV = tl.dot(k_block, v_block, acc=KtV,
                      input_precision="ieee", out_dtype=dtype)
 
    # Phase 2: stream ALL Q tiles and emit Out = Q @ KtV
    for q_start in tl.static_range(0, L, BLOCK_L):
        q_ptrs = tl.make_block_ptr(
            q_base,
            shape=(L, Cph),
            strides=(Cph, 1),
            offsets=(q_start, 0),
            block_shape=(BLOCK_L, BLOCK_Cph),
            order=(1, 0),
        )
        q_block = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")
 
        out_tile = tl.dot(q_block, KtV,
                          input_precision="ieee", out_dtype=dtype)
 
        out_ptrs = tl.make_block_ptr(
            out_base,
            shape=(L, Cph),
            strides=(Cph, 1),
            offsets=(q_start, 0),
            block_shape=(BLOCK_L, BLOCK_Cph),
            order=(1, 0),
        )
        tl.store(out_ptrs, out_tile, boundary_check=(0, 1))
 
 
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
 
 
def sdsa2_reference(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    N, H, L, Cph = Q.shape
    Q_flat = Q.flatten(0, 1)
    K_flat = K.flatten(0, 1)
    V_flat = V.flatten(0, 1)
    KtV = torch.bmm(K_flat.transpose(1, 2), V_flat)
    out_flat = torch.bmm(Q_flat, KtV)
    return out_flat.reshape(N, H, L, Cph)

