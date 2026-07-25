#!/usr/bin/env python3
"""vLLM entrypoint for LFM2.5-2.6B-Agent-NVFP4A16.

vLLM's Lfm2Model declares stacked-weight rules ".w1" -> (".w13", 0) and
".w3" -> (".w13", 1) so it can fuse an unfused HF checkpoint's gate/up
projections. The NVFP4A16 checkpoint ships them ALREADY fused, so its parameter
names contain ".w13." -- and the substring rewrite fires on the ".w1" inside
".w13", producing ".w133.":

    ValueError: There is no module or parameter named
    'layers.0.feed_forward.w133' in Lfm2Model

Dropping the two rules is correct here precisely because the checkpoint is
pre-fused; qkv stacking stays, since q/k/v are still separate tensors.

Do NOT serve the bf16 checkpoint through this file -- there the w1/w3 rules are
load-bearing and removing them breaks the MLP.

Usage (same args as `vllm`):
    <vllm-env-python> serve_nvfp4.py serve <model_path> --served-model-name ...
"""
import sys

from vllm.model_executor.models.lfm2 import Lfm2Model
from vllm.model_executor.models.utils import WeightsMapper

_orig = Lfm2Model.hf_to_vllm_mapper
_stacked = {
    k: v for k, v in _orig.orig_to_new_stacked.items() if k not in (".w1", ".w3")
}
if len(_stacked) == len(_orig.orig_to_new_stacked):
    print(
        "serve_nvfp4: WARNING: no .w1/.w3 stacked rules found; vLLM may have "
        "fixed this upstream. Verify the patch is still needed.",
        file=sys.stderr,
    )
Lfm2Model.hf_to_vllm_mapper = WeightsMapper(
    orig_to_new_renamings=_orig.orig_to_new_renamings,
    orig_to_new_regex=_orig.orig_to_new_regex,
    orig_to_new_substr=_orig.orig_to_new_substr,
    orig_to_new_stacked=_stacked,
    orig_to_new_prefix=_orig.orig_to_new_prefix,
    orig_to_new_suffix=_orig.orig_to_new_suffix,
)

from vllm.entrypoints.cli.main import main  # noqa: E402

# The patch above stays at module level on purpose: vLLM starts EngineCore with
# the "spawn" method, and the child re-imports this file as __mp_main__, so the
# mapper must be fixed on import, not inside main(). main() itself must be
# guarded, or that same re-import would try to start a second server.
if __name__ == "__main__":
    sys.exit(main())
