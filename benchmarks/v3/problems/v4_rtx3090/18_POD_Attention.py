"""POD-Attention: fused prefill + decode in one kernel for mixed-batch serving.

Real production pattern: an inference server batches requests in different
phases -- some are prefilling (long Q, long KV), others are decoding (Q=1,
long KV). Running two separate kernels wastes SM occupancy because the decode
tail has too few rows to fill the GPU. POD-Attention fuses both into a single
persistent kernel that co-schedules work.

Reference: FlashInfer's POD-Attention (2024) -- a serving-time optimization
that is absent from most kernel benchmarks.

Compute pattern: causal scaled-dot-product attention, one kernel, two request
phases in the same batch.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


HARDWARE_REQUIRED = ['RTX3090', 'H100', 'B200']
OP_TYPE = "attention"
SUPPORTED_PRECISIONS = ["bf16", "fp16"]

# Framework gate: Triton banned (can't express persistent fused scheduling
# naturally). Forces CUDA or CUTLASS.
FRAMEWORK_GATE = "no_triton"


class Model(nn.Module):
    """Batched causal attention with mixed prefill/decode request lengths.

    Inputs are packed: queries for all requests concatenated, KV similarly,
    with per-request length metadata. The model handles both phases in one
    forward call so a fused kernel can co-schedule them.
    """

    def __init__(self):
        super().__init__()

    def forward(self, q_packed, k_packed, v_packed, q_lens, kv_lens, scale):
        """
        Args:
            q_packed:  (sum_q, H, D)   -- queries from all requests, packed
            k_packed:  (sum_kv, H_kv, D)
            v_packed:  (sum_kv, H_kv, D)
            q_lens:    (B,) int64      -- per-request query lengths (mix of 1s and longer)
            kv_lens:   (B,) int64      -- per-request KV lengths
            scale:     float

        Returns:
            out_packed: (sum_q, H, D)
        """
        B = q_lens.shape[0]
        H = q_packed.shape[1]
        D = q_packed.shape[2]
        H_kv = k_packed.shape[1]
        group = H // H_kv

        outs = []
        q_off = 0
        kv_off = 0
        for i in range(B):
            qL = int(q_lens[i].item())
            kL = int(kv_lens[i].item())
            q = q_packed[q_off:q_off + qL]           # (qL, H, D)
            k = k_packed[kv_off:kv_off + kL]         # (kL, H_kv, D)
            v = v_packed[kv_off:kv_off + kL]
            # Expand GQA -> MHA
            k = k.repeat_interleave(group, dim=1)    # (kL, H, D)
            v = v.repeat_interleave(group, dim=1)
            # (H, qL, D) x (H, D, kL) -> (H, qL, kL)
            q_t = q.transpose(0, 1)
            k_t = k.transpose(0, 1)
            v_t = v.transpose(0, 1)
            scores = torch.einsum('hqd,hkd->hqk', q_t, k_t) * scale
            # Causal mask: query j attends keys [0, kL - qL + j]
            offset = kL - qL
            causal = torch.arange(kL, device=q.device).unsqueeze(0) <= (torch.arange(qL, device=q.device).unsqueeze(1) + offset)
            scores = scores.masked_fill(~causal, float('-inf'))
            attn = F.softmax(scores, dim=-1)
            out = torch.einsum('hqk,hkd->hqd', attn, v_t)  # (H, qL, D)
            outs.append(out.transpose(0, 1).contiguous())  # (qL, H, D)
            q_off += qL
            kv_off += kL
        return torch.cat(outs, dim=0)


# Shape anchor: 8-way mixed batch, 4 prefill requests (qL=512) + 4 decode (qL=1)
# head_dim=128, H=32, H_kv=8 (GQA ratio 4)
B = 8
H = 32
H_KV = 8
D = 128
KV_LEN_EACH = 2048
PREFILL_Q = 512
DECODE_Q = 1
SCALE = 1.0 / (D ** 0.5)


def get_inputs():
    q_lens = torch.tensor([PREFILL_Q] * (B // 2) + [DECODE_Q] * (B // 2), dtype=torch.int64)
    kv_lens = torch.tensor([KV_LEN_EACH] * B, dtype=torch.int64)
    sum_q = int(q_lens.sum().item())
    sum_kv = int(kv_lens.sum().item())
    torch.manual_seed(0)
    q = torch.randn(sum_q, H, D, dtype=torch.bfloat16)
    k = torch.randn(sum_kv, H_KV, D, dtype=torch.bfloat16)
    v = torch.randn(sum_kv, H_KV, D, dtype=torch.bfloat16)
    return [q, k, v, q_lens, kv_lens, SCALE]


def get_init_inputs():
    return []
