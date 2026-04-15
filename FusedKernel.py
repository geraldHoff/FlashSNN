# Given, S, w_k, w_q, w_v s.t.
# S, spiking feature map
# and w_k, w_q, w_v
# Compute attetnion according to SDSA2

#SDSA2
#Binary Spike Input S
#Linear projections of each weight matrix
#Spiking layer
#Dot products between vectors of K and V
#Broadcst element-wise multiplication between that and Q

import triton

@triton.jit
def _sdsa2_forward_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    L: tl.constexpr,      # Sequence length
    Cph: tl.constexpr,    # Head Dim, 48
    BLOCK_L: tl.constexpr,
    BLOCK_Cph: tl.constexpr,
    dtype: tl.constexpr,
):
    # Program IDs
    tn = tl.program_id(0)
    head = tl.program_id(1)
    l_idx = tl.program_id(2) * BLOCK_L

    # Pointer offsets
    offsets_l = l_idx + tl.arange(0, BLOCK_L)
    offsets_c = tl.arange(0, BLOCK_Cph)

    # Load Q block [BLOCK_L, BLOCK_Cph]
    # Q already binarized (0.0 or 1.0)
    q_ptrs = q_ptr + (offsets_l[:, None] * Cph + offsets_c[None, :])
    q = tl.load(q_ptrs, mask=(offsets_l[:, None] < L) & (offsets_c[None, :] < Cph), other=0.0)

    # Accumulator for K^T * V
    kv_acc = tl.zeros((BLOCK_Cph, BLOCK_Cph), dtype=tl.float32)

    # Iterate over Sequence Length to compute K^T * V
    for k_idx in range(0, L, BLOCK_L):
        k_offsets = k_idx + tl.arange(0, BLOCK_L)
        
        # Load K and V
        k_ptrs = k_ptr + (k_offsets[None, :] * Cph + offsets_c[:, None]) # Transposed K
        v_ptrs = v_ptr + (k_offsets[:, None] * Cph + offsets_c[None, :])
        
        k = tl.load(k_ptrs, mask=(k_offsets[None, :] < L) & (offsets_c[:, None] < Cph), other=0.0)
        v = tl.load(v_ptrs, mask=(k_offsets[:, None] < L) & (offsets_c[None, :] < Cph), other=0.0)
        
        # Dot product: (Cph, BLOCK_L) @ (BLOCK_L, Cph) -> (Cph, Cph)
        kv_acc += tl.dot(k, v)

    # Broadcasted element-wise multiplication
    # Output[i, j] = Q[i, j] * Sum(K[:, i] * V[:, j])
    res = q * kv_acc 

    out_ptrs = out_ptr + (offsets_l[:, None] * Cph + offsets_c[None, :])
    tl.store(out_ptrs, res.to(dtype), mask=(offsets_l[:, None] < L) & (offsets_c[None, :] < Cph))