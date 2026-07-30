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

# Mutate the mapper's stacked-rule dict in place rather than constructing a new
# WeightsMapper: its keyword fields get renamed across vLLM versions
# (orig_to_new_renamings vs orig_to_new_renaming), and the dict itself is what
# weight loading consults.
_stacked = Lfm2Model.hf_to_vllm_mapper.orig_to_new_stacked
_dropped = [k for k in (".w1", ".w3") if _stacked.pop(k, None) is not None]
if not _dropped:
    print(
        "serve_nvfp4: WARNING: no .w1/.w3 stacked rules found; vLLM may have "
        "fixed this upstream. Verify the patch is still needed.",
        file=sys.stderr,
    )

from vllm.entrypoints.cli.main import main  # noqa: E402

# The patch above stays at module level on purpose: vLLM starts EngineCore with
# the "spawn" method, and the child re-imports this file as __mp_main__, so the
# mapper must be fixed on import, not inside main(). main() itself must be
# guarded, or that same re-import would try to start a second server.
if __name__ == "__main__":
    sys.exit(main())
