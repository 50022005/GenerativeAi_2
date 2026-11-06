"""Educational attention, explicitly tiled; not a fused FlashAttention kernel."""
import math
import torch

def reference(q, k, v, causal=True, query_offset=0):
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    if causal:
        mask = torch.arange(k.shape[-2], device=q.device)[None, :] > (
            torch.arange(q.shape[-2], device=q.device)[:, None] + query_offset)
        scores = scores.masked_fill(mask, -torch.inf)
    maximum = scores.amax(-1, keepdim=True)
    weights = torch.exp(scores - maximum)
    weights = weights / weights.sum(-1, keepdim=True)
    return weights @ v

def tiled(q, k, v, causal=True, query_offset=0, query_tile=32, key_tile=32):
    output = torch.empty((*q.shape[:-1], v.shape[-1]), dtype=q.dtype, device=q.device)
    for qs in range(0, q.shape[-2], query_tile):
        block = q[..., qs:qs+query_tile, :]
        shape = (*block.shape[:-1], 1)
        m = torch.full(shape, -torch.inf, dtype=q.dtype, device=q.device)
        l = torch.zeros(shape, dtype=q.dtype, device=q.device)
        acc = torch.zeros((*block.shape[:-1], v.shape[-1]), dtype=q.dtype, device=q.device)
        for ks in range(0, k.shape[-2], key_tile):
            kb, vb = k[..., ks:ks+key_tile, :], v[..., ks:ks+key_tile, :]
            scores = block @ kb.transpose(-1, -2) / math.sqrt(q.shape[-1])
            if causal:
                mask = torch.arange(ks, ks+kb.shape[-2], device=q.device)[None, :] > (
                    torch.arange(qs, qs+block.shape[-2], device=q.device)[:, None] + query_offset)
                scores = scores.masked_fill(mask, -torch.inf)
            new_m = torch.maximum(m, scores.amax(-1, keepdim=True))
            # Handle all-masked tiles: their exponential contribution is exactly zero.
            safe_m = torch.where(torch.isfinite(new_m), new_m, torch.zeros_like(new_m))
            scale = torch.exp(m-safe_m)
            p = torch.exp(scores-safe_m)
            acc = acc * scale + p @ vb
            l = l * scale + p.sum(-1, keepdim=True)
            m = new_m
        output[..., qs:qs+block.shape[-2], :] = acc / l.clamp_min(torch.finfo(q.dtype).tiny)
    return output
