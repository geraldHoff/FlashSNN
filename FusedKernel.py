import math
import torch
import triton
import triton.language as tl


# --------------------------------------------------------------------------
# Kernel 1: attention with LIF membrane on KtV
#
# Computes KtV = K^T @ V  →  [Cph, Cph] per (batch, head),
# then applies a Leaky Integrate-and-Fire neuron with persistent
# membrane potential across timesteps:
#
#     u = τ · u_prev + KtV
#     spike = (u > V_th)
#     u = u · (1 - spike)          (hard reset)
#
# The spiked KtV (binary) is then used as:  Out = Q @ spike(KtV)
#
# Memory layout for the membrane:
#   - Shape [N, H, Cph, Cph], stored in HBM, persisted across launches.
#   - Each program (one per batch×head) loads its [Cph, Cph] slice once
#     before the LIF update and stores it once after.  The membrane sits
#     in registers between load and store — no extra global traffic
#     during the L-loop or Q-loop.
#
# One program per (batch, head).
# --------------------------------------------------------------------------
 
@triton.jit
def _sdsa2_lif_forward_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    U_ktv_ptr,                          # membrane state [N*H, Cph, Cph]
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
 
    # Membrane base: each (batch, head) owns a [Cph, Cph] block
    u_base = U_ktv_ptr + bh * Cph * Cph
 
    # ---- Phase 1: accumulate KtV = K^T @ V over all L-tiles ----
 
    KtV = tl.zeros((BLOCK_Cph, BLOCK_Cph), dtype=dtype)
 
    for l_start in tl.static_range(0, L, BLOCK_L):
        k_ptrs = tl.make_block_ptr(
            k_base,
            shape=(Cph, L), strides=(1, Cph),
            offsets=(0, l_start),
            block_shape=(BLOCK_Cph, BLOCK_L), order=(1, 0),
        )
        k_block = tl.load(k_ptrs, boundary_check=(0, 1), padding_option="zero")
 
        v_ptrs = tl.make_block_ptr(
            v_base,
            shape=(L, Cph), strides=(Cph, 1),
            offsets=(l_start, 0),
            block_shape=(BLOCK_L, BLOCK_Cph), order=(1, 0),
        )
        v_block = tl.load(v_ptrs, boundary_check=(0, 1), padding_option="zero")
 
        KtV = tl.dot(k_block, v_block, acc=KtV,
                      input_precision="ieee", out_dtype=dtype)
 
    # ---- LIF update on the accumulated KtV ----
    #
    # Single load from HBM → registers, update, single store back.
    # The spiked result stays in registers for Phase 2.
 
    u_ptrs = tl.make_block_ptr(
        u_base,
        shape=(Cph, Cph), strides=(Cph, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_Cph, BLOCK_Cph), order=(1, 0),
    )
    u_mem = tl.load(u_ptrs, boundary_check=(0, 1), padding_option="zero").to(dtype)
 
    # Leaky integrate: decay old potential, add new input
    u_mem = tau * u_mem + KtV
 
    # Fire: threshold comparison → binary spike matrix
    ktv_spike = (u_mem > V_th).to(dtype)
 
    # Hard reset: zero out membrane where a spike occurred
    u_mem = u_mem * (1.0 - ktv_spike)
 
    # Persist updated membrane to HBM for the next timestep
    tl.store(u_ptrs, u_mem, boundary_check=(0, 1))
 
    # ---- Phase 2: Out = Q @ spike(KtV) for each L-tile ----
 
    for q_start in tl.static_range(0, L, BLOCK_L):
        q_ptrs = tl.make_block_ptr(
            q_base,
            shape=(L, Cph), strides=(Cph, 1),
            offsets=(q_start, 0),
            block_shape=(BLOCK_L, BLOCK_Cph), order=(1, 0),
        )
        q_block = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")
 
        out_tile = tl.dot(q_block, ktv_spike,
                          input_precision="ieee", out_dtype=dtype)
 
        out_ptrs = tl.make_block_ptr(
            out_base,
            shape=(L, Cph), strides=(Cph, 1),
            offsets=(q_start, 0),
            block_shape=(BLOCK_L, BLOCK_Cph), order=(1, 0),
        )
        tl.store(out_ptrs, out_tile, boundary_check=(0, 1))
 
 
# --------------------------------------------------------------------------
# Python wrapper
# --------------------------------------------------------------------------
 
def sdsa2_lif_forward(
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
    SDSA v2 forward with LIF membrane on the KtV aggregation.
 
    Args:
        Q, K, V:  [N, H, L, Cph]  pre-spiked binary inputs.
        u_ktv:    [N, H, Cph, Cph] membrane state, or None to zero-init.
                  Mutated in-place; pass the returned tensor back on the
                  next timestep.
        tau:      Membrane leak factor (0 = no memory, 1 = perfect memory).
        V_th:     Firing threshold.
 
    Returns:
        out:      [N, H, L, Cph]   attention output.
        u_ktv:    [N, H, Cph, Cph] updated membrane (same object, mutated).
    """
    assert Q.shape == K.shape == V.shape
    assert Q.is_cuda and Q.is_contiguous()
    N, H, L, Cph = Q.shape
 
    # Allocate or validate membrane state
    if u_ktv is None:
        u_ktv = torch.zeros(N, H, Cph, Cph, dtype=Q.dtype, device=Q.device)
    else:
        assert u_ktv.shape == (N, H, Cph, Cph), (
            f"Expected membrane shape {(N, H, Cph, Cph)}, got {u_ktv.shape}"
        )
        u_ktv = u_ktv.contiguous()
 
    out = torch.empty_like(Q)
 
    # Flatten batch and head dims for the kernel grid
    Q_flat = Q.reshape(N * H, L, Cph)
    K_flat = K.reshape(N * H, L, Cph)
    V_flat = V.reshape(N * H, L, Cph)
    out_flat = out.reshape(N * H, L, Cph)
    u_flat = u_ktv.reshape(N * H, Cph, Cph)
 
    grid = (N * H,)
 
    _sdsa2_lif_forward_kernel[grid](
        Q_flat, K_flat, V_flat, out_flat,
        u_flat,
        tau, V_th,
        L=L, Cph=Cph,
        BLOCK_L=BLOCK_L, BLOCK_Cph=BLOCK_Cph,
        dtype=tl.float32,
    )
    return out, u_ktv



# --------------------------------------------------------------------------
# Reference implementation (pure PyTorch, for validation)
# --------------------------------------------------------------------------
 
def sdsa2_lif_reference(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    u_ktv: torch.Tensor | None = None,
    tau: float = 0.25,
    V_th: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reference LIF-gated SDSA v2.
    Q, K, V: [N, H, L, Cph] binary.
    u_ktv:   [N, H, Cph, Cph] membrane or None.
    """
    N, H, L, Cph = Q.shape
 
    if u_ktv is None:
        u_ktv = torch.zeros(N, H, Cph, Cph, dtype=Q.dtype, device=Q.device)
    else:
        u_ktv = u_ktv.clone()       # don't mutate caller's tensor
 
    # K^T @ V:  [N, H, Cph, L] @ [N, H, L, Cph] → [N, H, Cph, Cph]
    KtV = torch.einsum("nhlc,nhld->nhcd", K, V)
 
    # LIF update
    u_ktv = tau * u_ktv + KtV
    spike = (u_ktv > V_th).float()
    u_ktv = u_ktv * (1.0 - spike)      # hard reset
 
    # Out = Q @ spike(KtV):  [N, H, L, Cph] @ [N, H, Cph, Cph]
    out = torch.einsum("nhlc,nhcd->nhld", Q, spike)
 
    return out, u_ktv
