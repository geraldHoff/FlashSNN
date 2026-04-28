import math
import torch
import triton
import triton.language as tl



@triton.jit
def _sdsa2_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    U_ktv_ptr,                          # membrane state [N*H, Cph]
    tau,                                # leak factor (scalar)
    V_th,                               # firing threshold (scalar)
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
 
    # Membrane base: each (batch, head) owns a [Cph] vector
    u_base = U_ktv_ptr + bh * Cph
 
    cph_offs = tl.arange(0, BLOCK_Cph)
    cph_mask = cph_offs < Cph
 
    # ---- Phase 1: ktv_acc = Σ_L(K ⊙ V) ----
    # Accumulate per-channel co-occurrence counts across all L-tiles.
    # The result is a [BLOCK_Cph] vector
 
    ktv_acc = tl.zeros((BLOCK_Cph,), dtype=dtype)
 
    for l_start in tl.static_range(0, L, BLOCK_L):
        l_offs = l_start + tl.arange(0, BLOCK_L)
        mask = (l_offs[:, None] < L) & (cph_offs[None, :] < Cph)
 
        k_ptrs = k_base + l_offs[:, None] * Cph + cph_offs[None, :]
        v_ptrs = v_base + l_offs[:, None] * Cph + cph_offs[None, :]
 
        k_tile = tl.load(k_ptrs, mask=mask, other=0.0).to(dtype)
        v_tile = tl.load(v_ptrs, mask=mask, other=0.0).to(dtype)
 
        # Hadamard product + reduce over L dimension | column wise dot product reduction
        kv_prod = k_tile * v_tile                   # [BLOCK_L, BLOCK_Cph]
        ktv_acc += tl.sum(kv_prod, axis=0)          # [BLOCK_Cph]
 
    # ---- LIF update on the [Cph] aggregation vector ----
    # Single [Cph] load from HBM → registers, update, single store back.
    # The spiked result stays in registers for Phase 2.
 
    u_ptrs = u_base + cph_offs
    u_mem = tl.load(u_ptrs, mask=cph_mask, other=0.0).to(dtype)
 
    # Leaky integrate
    u_mem = tau * u_mem + ktv_acc
 
    # Fire
    spike = (u_mem > V_th).to(dtype)
 
    # Hard reset
    u_mem = u_mem * (1.0 - spike)
 
    # Persist updated membrane to HBM
    tl.store(u_ptrs, u_mem, mask=cph_mask)
 
    # ---- Phase 2: out = Q * spike (broadcast multiply) ----
    # spike is [BLOCK_Cph], broadcast across every row of each Q tile.
    # element-wise multiplication.
 
    for q_start in tl.static_range(0, L, BLOCK_L):
        q_offs = q_start + tl.arange(0, BLOCK_L)
        mask = (q_offs[:, None] < L) & (cph_offs[None, :] < Cph)
 
        q_ptrs = q_base + q_offs[:, None] * Cph + cph_offs[None, :]
        q_tile = tl.load(q_ptrs, mask=mask, other=0.0).to(dtype)
 
        out_tile = q_tile * spike[None, :]          # broadcast [BLOCK_L, BLOCK_Cph]
 
        out_ptrs = out_base + q_offs[:, None] * Cph + cph_offs[None, :]
        tl.store(out_ptrs, out_tile, mask=mask)
 
 
# --------------------------------------------------------------------------
# Python wrapper
# --------------------------------------------------------------------------
 
def sdsa2_forward(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    u_ktv: torch.Tensor | None = None,
    tau: float = 0.25,
    V_th: float = 0.5,
    BLOCK_L: int = 64,
    BLOCK_Cph: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    SDSA2 w/ LIF
 
    Args:
        Q, K, V:  [N, H, L, Cph]  pre-spiked binary inputs.
        u_ktv:    [N, H, Cph] membrane state, or None to zero-init.
                  Mutated in-place; pass the returned tensor back on the
                  next timestep.
        tau:      Membrane leak factor (0 = no memory, 1 = perfect memory).
        V_th:     Firing threshold.
 
    Returns:
        out:      [N, H, L, Cph]  attention output.
        u_ktv:    [N, H, Cph]     updated membrane (same object, mutated).
    """
    assert Q.shape == K.shape == V.shape
    assert Q.is_cuda and Q.is_contiguous()
    N, H, L, Cph = Q.shape
 
    # Allocate or validate membrane state, [N, H, Cph]
    if u_ktv is None:
        u_ktv = torch.zeros(N, H, Cph, dtype=Q.dtype, device=Q.device)
    else:
        assert u_ktv.shape == (N, H, Cph), (
            f"Expected membrane shape {(N, H, Cph)}, got {u_ktv.shape}"
        )
        u_ktv = u_ktv.contiguous()
 
    out = torch.empty_like(Q)
 
    # Flatten batch and head dims for the kernel grid
    Q_flat = Q.reshape(N * H, L, Cph)
    K_flat = K.reshape(N * H, L, Cph)
    V_flat = V.reshape(N * H, L, Cph)
    out_flat = out.reshape(N * H, L, Cph)
    u_flat = u_ktv.reshape(N * H, Cph)
 
    grid = (N * H,)
 
    _sdsa2_kernel[grid](
        Q_flat, K_flat, V_flat, out_flat,
        u_flat,
        tau, V_th,
        L=L, Cph=Cph,
        BLOCK_L=BLOCK_L, BLOCK_Cph=BLOCK_Cph,
        dtype=tl.float32,
    )
    return out, u_ktv

import torch
 
 
# --------------------------------------------------------------------------
# Naive SDSA v2 baseline (pure PyTorch, no Triton)
#
#uses einsum to express each operation semantically:
#
#   1. ktv = einsum('nhlc,nhlc->nhc')  vector-wise dot product [N,H,Cph]
#   2. u = τ·u_prev + ktv              leaky integrate         [N,H,Cph]
#   3. spike = (u > V_th)              fire                    [N,H,Cph]
#   4. u = u · (1 - spike)             hard reset              [N,H,Cph]
#   5. out = einsum('nhlc,nhc->nhlc')  broadcast multiply      [N,H,L,Cph]
#
# vector-wise dot product + broadcast.  Each einsum is the intended operation 
# without decomposing into separate element-wise + reduction steps.
# Doesn't tile or do inter-operation fusion.
#
# The LIF membrane is [N, H, Cph].
# --------------------------------------------------------------------------
 
 
def sdsa2_lif_hadamard_baseline(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    u_ktv: torch.Tensor | None = None,
    tau: float = 0.25,
    V_th: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Paper-faithful SDSA v2 + LIF.
 
    Args:
        Q, K, V:  [N, H, L, Cph]  pre-spiked binary inputs.
        u_ktv:    [N, H, Cph] membrane state, or None to zero-init.
                  NOTE: this is a vector, not a matrix — the paper's
                  Hadamard-reduce produces a [Cph] aggregation, not
                  [Cph, Cph].
        tau:      Membrane leak factor.
        V_th:     Firing threshold.
 
    Returns:
        out:      [N, H, L, Cph]  attention output.
        u_ktv:    [N, H, Cph]     updated membrane.
    """
    N, H, L, Cph = Q.shape
 
    if u_ktv is None:
        u_ktv = torch.zeros(N, H, Cph, dtype=Q.dtype, device=Q.device)
    else:
        u_ktv = u_ktv.clone()
 
    # Step 1: vector-wise dot product  Σ_L(K ⊙ V)  →  [N, H, Cph]
    #         contracts over L per channel in a single fused op
    ktv = torch.einsum('nhlc,nhlc->nhc', K, V)
 
    # Step 2: leaky integrate
    u_ktv = tau * u_ktv + ktv
 
    # Step 3: fire
    spike = (u_ktv > V_th).to(Q.dtype)
 
    # Step 4: hard reset
    u_ktv = u_ktv * (1.0 - spike)
 
    # Step 5: broadcast element-wise multiply  Q ⊗ spike  →  [N, H, L, Cph]
    #         spike [N,H,Cph] broadcast across L positions of Q [N,H,L,Cph]
    out = torch.einsum('nhlc,nhc->nhlc', Q, spike)
 
    return out, u_ktv

