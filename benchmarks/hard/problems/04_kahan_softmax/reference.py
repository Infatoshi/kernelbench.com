"""Naive softmax over the last dim, computed in fp64 for ground-truth.

The reference deliberately runs in float64 so that fp16 / fp32 accumulation
drift in agent solutions is exposed by the tight tolerance in problem.yaml.
The agent's job is to produce an fp32 softmax whose values match this
double-precision reference within atol=rtol=1e-5 — this requires either
fp32 accumulation or compensated (Kahan) summation when vocab is large.
"""
import torch
import torch.nn as nn

OP_TYPE = "softmax"
SUPPORTED_PRECISIONS = ["fp32"]
HARDWARE_REQUIRED = ["RTX_PRO_6000", "H100", "B200"]


class Model(nn.Module):
    """y = softmax(x, dim=-1) computed in fp64 then returned as fp32.

    No learned parameters — softmax is parameter-free. We still expose an
    empty state_dict so the harness's strict load_state_dict matches.
    """

    def __init__(self, batch: int, vocab: int):
        super().__init__()
        self.batch = batch
        self.vocab = vocab

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Promote to fp64 for the ground-truth pathway. Even with double
        # precision we still subtract the row-max for stability.
        x64 = x.to(torch.float64)
        m = x64.amax(dim=-1, keepdim=True)
        e = torch.exp(x64 - m)
        s = e.sum(dim=-1, keepdim=True)
        return (e / s).to(torch.float32)


# Default shape; overridden per-iteration by check.py / benchmark.py.
BATCH = 8
VOCAB = 32768


def get_inputs():
    # Mix of moderate-magnitude logits. The shapes module supplies an
    # extreme-magnitude variant separately to stress numerical stability.
    x = torch.randn(BATCH, VOCAB, dtype=torch.float32) * 4.0
    return [x]


def get_init_inputs():
    return [BATCH, VOCAB]
