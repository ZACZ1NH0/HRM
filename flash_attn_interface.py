import torch
import torch.nn.functional as F

# Shim thay thế flash_attn_func bằng SDPA (PyTorch)
# Kỳ vọng đầu vào: q, k, v shape [B, H, S, D]
# Trả về: tensor [B, H, S, D] (không tuple)
def flash_attn_func(q, k, v, causal=False, **kwargs):
    assert q.dim() == k.dim() == v.dim() == 4, "Expect q,k,v as [B,H,S,D]"
    B, H, S, D = q.shape
    # gộp batch & heads để dùng SDPA: [B*H, S, D]
    q_ = q.transpose(0,1).reshape(H*B, S, D)
    k_ = k.transpose(0,1).reshape(H*B, S, D)
    v_ = v.transpose(0,1).reshape(H*B, S, D)

    # PyTorch >=2.0 có is_causal; set cả attn_mask để tương thích
    attn_mask = None
    if causal:
        # upper-triangular True trên phần bị che
        attn_mask = torch.ones((S, S), dtype=torch.bool, device=q_.device).triu(1)

    out = F.scaled_dot_product_attention(
        q_, k_, v_,
        attn_mask=attn_mask,
        is_causal=causal
    )  # [B*H, S, D]

    out = out.reshape(H, B, S, D).transpose(0,1).contiguous()  # [B,H,S,D]
    return out