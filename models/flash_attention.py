"""Flash-attention helpers for BERT-style (batch, heads, seq, dim) tensors."""

import torch

try:
    from flash_attn import flash_attn_func
    from flash_attn.bert_padding import unpad_input, pad_input
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
    _HAS_FLASH_ATTN = True
except ImportError:
    flash_attn_func = None
    unpad_input = None
    pad_input = None
    flash_attn_varlen_qkvpacked_func = None
    _HAS_FLASH_ATTN = False


def has_flash_attention():
    return _HAS_FLASH_ATTN


def _token_valid_mask(extended_attention_mask):
    """Recover per-token validity (B, S) from BERT extended additive mask."""
    if extended_attention_mask is None:
        return None
    if extended_attention_mask.dim() != 4:
        return None
    return extended_attention_mask[:, 0, 0, :] > -5000


def _is_causal_mask(extended_attention_mask):
    if extended_attention_mask is None:
        return False
    return (
        extended_attention_mask.dim() == 4
        and extended_attention_mask.size(2) > 1
        and extended_attention_mask.size(3) > 1
    )


def _cast_flash_inputs(*tensors):
    orig_dtype = tensors[0].dtype
    if orig_dtype in (torch.float16, torch.bfloat16):
        return tensors, orig_dtype
    return tuple(t.half() for t in tensors), orig_dtype


def _restore_dtype(tensor, orig_dtype):
    if tensor.dtype != orig_dtype:
        return tensor.to(orig_dtype)
    return tensor


def flash_self_attention(q, k, v, extended_attention_mask, dropout_p=0.0):
    """
    Flash self-attention for q/k/v shaped (B, num_heads, seq_len, head_dim).
    Returns output with the same shape, or None to fall back to standard attention.
    """
    if not _HAS_FLASH_ATTN:
        return None

    B, H, S, D = q.shape
    q_ = q.transpose(1, 2).contiguous()
    k_ = k.transpose(1, 2).contiguous()
    v_ = v.transpose(1, 2).contiguous()
    causal = _is_causal_mask(extended_attention_mask)

    valid = _token_valid_mask(extended_attention_mask)
    if valid is None or valid.all():
        (q_, k_, v_), orig_dtype = _cast_flash_inputs(q_, k_, v_)
        out = flash_attn_func(q_, k_, v_, dropout_p=dropout_p, causal=causal)
        return _restore_dtype(out, orig_dtype).transpose(1, 2)

    attn_mask = valid.long()
    qkv = torch.stack([q_, k_, v_], dim=2)
    qkv_flat = qkv.reshape(B, S, 3 * H * D)
    qkv_unpad, indices, cu_seqlens, max_s = unpad_input(qkv_flat, attn_mask)
    qkv_unpad = qkv_unpad.view(-1, 3, H, D)
    if qkv_unpad.dtype not in (torch.float16, torch.bfloat16):
        qkv_unpad = qkv_unpad.half()
        orig_dtype = q.dtype
    else:
        orig_dtype = qkv_unpad.dtype
    out_unpad = flash_attn_varlen_qkvpacked_func(
        qkv_unpad, cu_seqlens, max_s, dropout_p=dropout_p, causal=causal
    )
    out_flat = pad_input(out_unpad.reshape(-1, H * D), indices, B, S)
    out = out_flat.view(B, S, H, D).transpose(1, 2)
    return _restore_dtype(out, q.dtype)


def flash_cross_attention(q, k, v, extended_q_mask, extended_kv_mask, dropout_p=0.0):
    """
    Flash cross-attention for q (B,H,Sq,D), k/v (B,H,Sk,D).
    Returns output (B,H,Sq,D) or None to fall back to standard attention.
    """
    if not _HAS_FLASH_ATTN:
        return None

    q_valid = _token_valid_mask(extended_q_mask)
    kv_valid = _token_valid_mask(extended_kv_mask)
    if q_valid is not None and not q_valid.all():
        return None
    if kv_valid is not None and not kv_valid.all():
        return None

    q_ = q.transpose(1, 2).contiguous()
    k_ = k.transpose(1, 2).contiguous()
    v_ = v.transpose(1, 2).contiguous()
    (q_, k_, v_), orig_dtype = _cast_flash_inputs(q_, k_, v_)
    out = flash_attn_func(q_, k_, v_, dropout_p=dropout_p, causal=False)
    return _restore_dtype(out, orig_dtype).transpose(1, 2)
