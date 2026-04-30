import Link from "next/link"

export default function KernelBenchHardPost() {
  return (
    <div className="space-y-8">

        <Link href="/" className="text-[var(--color-fg-bright)] hover:underline mb-8 inline-block">
          &larr; back to home
        </Link>

        <article className="max-w-none">
          <h1 className="text-4xl font-bold mb-6">KernelBench-Hard: Seven Problems, Twelve Frontier Models, Two Rubric Leaks</h1>

          <p className="text-[var(--color-fg-muted)] mb-12">
            A focused successor to KernelBench v3. One GPU, fewer but harder problems, real coding-agent CLIs as the harness, and an explicit decision to publish the rubric leaks rather than iterate until perfect.
          </p>

          <h2 className="text-2xl font-bold mt-12 mb-6">Results at a Glance</h2>

          <p className="leading-relaxed mb-6">
            12 models across 7 problems on a single RTX PRO 6000 Blackwell Workstation (sm_120, 96 GB GDDR7, 1.8 TB/s peak DRAM bandwidth). Cells are <code>peak_fraction</code> — the fraction of the relevant tensor-core or memory-bandwidth ceiling the kernel actually achieved. <code>FAIL</code> means a solution was written but missed the correctness gate; <code>ERR</code> means no solution was produced. <code>★</code> marks cells that have a per-run annotation in the source repo.
          </p>

          <div className="overflow-x-auto mb-6 text-sm">
            <table className="w-full">
              <thead>
                <tr className="border-b border-[var(--color-border)]">
                  <th className="text-left py-2 pr-4">Model</th>
                  <th className="text-left py-2 pr-2">01 fp8</th>
                  <th className="text-left py-2 pr-2">02 kda</th>
                  <th className="text-left py-2 pr-2">03 paged</th>
                  <th className="text-left py-2 pr-2">04 kahan</th>
                  <th className="text-left py-2 pr-2">05 topk</th>
                  <th className="text-left py-2 pr-2">06 moe</th>
                  <th className="text-left py-2 pr-2">07 w4a16</th>
                  <th className="text-left py-2">PASS</th>
                </tr>
              </thead>
              <tbody className="text-[var(--color-fg-muted)] font-mono text-xs">
                <tr className="border-b border-[var(--color-border)]/50"><td className="py-2 pr-4 font-semibold text-[var(--color-fg)]">gpt-5.5 [xhigh]</td><td className="py-2 pr-2">0.423 ★</td><td className="py-2 pr-2">0.032</td><td className="py-2 pr-2">0.497</td><td className="py-2 pr-2">0.363 ★</td><td className="py-2 pr-2">0.042</td><td className="py-2 pr-2">0.251</td><td className="py-2 pr-2">0.159</td><td className="py-2 text-[var(--color-fg)]">7/7</td></tr>
                <tr className="border-b border-[var(--color-border)]/50"><td className="py-2 pr-4 font-semibold text-[var(--color-fg)]">claude-opus-4-7 [max]</td><td className="py-2 pr-2">0.534 ★</td><td className="py-2 pr-2">PASS</td><td className="py-2 pr-2">0.602 ★</td><td className="py-2 pr-2">0.317 ★</td><td className="py-2 pr-2">0.020</td><td className="py-2 pr-2">FAIL</td><td className="py-2 pr-2">0.184</td><td className="py-2 text-[var(--color-fg)]">6/7</td></tr>
                <tr className="border-b border-[var(--color-border)]/50"><td className="py-2 pr-4 font-semibold text-[var(--color-fg)]">kimi-k2.6</td><td className="py-2 pr-2">FAIL</td><td className="py-2 pr-2">0.022</td><td className="py-2 pr-2">0.432</td><td className="py-2 pr-2">0.118 ★</td><td className="py-2 pr-2">0.014</td><td className="py-2 pr-2">0.161</td><td className="py-2 pr-2">0.220</td><td className="py-2 text-[var(--color-fg)]">6/7</td></tr>
                <tr className="border-b border-[var(--color-border)]/50"><td className="py-2 pr-4">mimo-v2.5-pro</td><td className="py-2 pr-2">0.434 ★</td><td className="py-2 pr-2">FAIL</td><td className="py-2 pr-2">ERR</td><td className="py-2 pr-2">0.121 ★</td><td className="py-2 pr-2">0.017</td><td className="py-2 pr-2">0.211</td><td className="py-2 pr-2">0.137</td><td className="py-2">5/7</td></tr>
                <tr className="border-b border-[var(--color-border)]/50"><td className="py-2 pr-4">qwen3.6-max-preview</td><td className="py-2 pr-2">0.429 ★</td><td className="py-2 pr-2">0.011</td><td className="py-2 pr-2">ERR</td><td className="py-2 pr-2">0.077</td><td className="py-2 pr-2">FAIL</td><td className="py-2 pr-2">0.004</td><td className="py-2 pr-2">0.110</td><td className="py-2">5/7</td></tr>
                <tr className="border-b border-[var(--color-border)]/50"><td className="py-2 pr-4">deepseek-v4-flash</td><td className="py-2 pr-2">FAIL</td><td className="py-2 pr-2">0.009</td><td className="py-2 pr-2">0.167</td><td className="py-2 pr-2">0.138 ★</td><td className="py-2 pr-2">FAIL</td><td className="py-2 pr-2">0.083</td><td className="py-2 pr-2">0.134</td><td className="py-2">5/7</td></tr>
                <tr className="border-b border-[var(--color-border)]/50"><td className="py-2 pr-4">deepseek-v4-pro</td><td className="py-2 pr-2">FAIL</td><td className="py-2 pr-2">FAIL</td><td className="py-2 pr-2">0.027</td><td className="py-2 pr-2">0.101 ★</td><td className="py-2 pr-2">0.011</td><td className="py-2 pr-2">0.108</td><td className="py-2 pr-2">0.125</td><td className="py-2">5/7</td></tr>
                <tr className="border-b border-[var(--color-border)]/50"><td className="py-2 pr-4">qwen3.6-plus</td><td className="py-2 pr-2">0.431 ★</td><td className="py-2 pr-2">ERR</td><td className="py-2 pr-2">0.022</td><td className="py-2 pr-2">ERR</td><td className="py-2 pr-2">FAIL</td><td className="py-2 pr-2">0.040</td><td className="py-2 pr-2">0.125</td><td className="py-2">4/7</td></tr>
                <tr className="border-b border-[var(--color-border)]/50"><td className="py-2 pr-4">glm-5.1</td><td className="py-2 pr-2">FAIL</td><td className="py-2 pr-2">0.005</td><td className="py-2 pr-2">ERR</td><td className="py-2 pr-2">0.125 ★</td><td className="py-2 pr-2">ERR</td><td className="py-2 pr-2">0.238</td><td className="py-2 pr-2">0.180</td><td className="py-2">4/7</td></tr>
                <tr className="border-b border-[var(--color-border)]/50"><td className="py-2 pr-4">minimax-m2.7</td><td className="py-2 pr-2">ERR</td><td className="py-2 pr-2">ERR</td><td className="py-2 pr-2">FAIL</td><td className="py-2 pr-2">0.034</td><td className="py-2 pr-2">FAIL</td><td className="py-2 pr-2">0.076</td><td className="py-2 pr-2">0.030</td><td className="py-2">3/7</td></tr>
                <tr className="border-b border-[var(--color-border)]/50"><td className="py-2 pr-4">qwen3.6-27b</td><td className="py-2 pr-2">ERR</td><td className="py-2 pr-2">FAIL</td><td className="py-2 pr-2">FAIL</td><td className="py-2 pr-2">ERR</td><td className="py-2 pr-2">FAIL</td><td className="py-2 pr-2">0.082</td><td className="py-2 pr-2">ERR</td><td className="py-2">1/7</td></tr>
                <tr><td className="py-2 pr-4">qwen3.6-35b-a3b</td><td className="py-2 pr-2">ERR</td><td className="py-2 pr-2">ERR</td><td className="py-2 pr-2">ERR</td><td className="py-2 pr-2">ERR</td><td className="py-2 pr-2">ERR</td><td className="py-2 pr-2">ERR</td><td className="py-2 pr-2">ERR</td><td className="py-2">0/7</td></tr>
              </tbody>
            </table>
          </div>

          <p className="leading-relaxed mb-6">
            <strong>GPT-5.5 at extra-high reasoning is the only model that solved every problem.</strong> Claude Opus 4.7 max ate one FAIL on sonic_moe but earned the highest peak fraction on the deck — 0.602 on paged attention, with a real Triton FlashDecoding-style kernel. Kimi K2.6 was the surprise of the run: 6/7 PASS at a much lower API cost than the top tier, including the only PASS where it took the deck-leading peak (0.220 on w4a16). Qwen 3.6 35B-A3B never got a single tool call through — its only OpenRouter providers don&apos;t advertise tool-use, so the agent harness couldn&apos;t reach it. That&apos;s an honest <code>0/7</code>: not a capability failure, an infrastructure ceiling.
          </p>

          <h3 className="text-xl font-bold mt-8 mb-4">Per-problem ceilings</h3>

          <div className="overflow-x-auto mb-6 text-sm">
            <table className="w-full">
              <thead>
                <tr className="border-b border-[var(--color-border)]">
                  <th className="text-left py-2 pr-4">Problem</th>
                  <th className="text-left py-2 pr-4">Best peak</th>
                  <th className="text-left py-2 pr-4">Best model</th>
                  <th className="text-left py-2">N correct</th>
                </tr>
              </thead>
              <tbody className="text-[var(--color-fg-muted)]">
                <tr className="border-b border-[var(--color-border)]/50"><td className="py-2 pr-4 font-medium">01 fp8_gemm</td><td className="py-2 pr-4 font-mono">0.534</td><td className="py-2 pr-4">claude-opus-4-7 [max]</td><td className="py-2">5/12</td></tr>
                <tr className="border-b border-[var(--color-border)]/50"><td className="py-2 pr-4 font-medium">02 kda_cutlass</td><td className="py-2 pr-4 font-mono">0.032</td><td className="py-2 pr-4">gpt-5.5 [xhigh]</td><td className="py-2">6/12</td></tr>
                <tr className="border-b border-[var(--color-border)]/50"><td className="py-2 pr-4 font-medium">03 paged_attention</td><td className="py-2 pr-4 font-mono">0.602</td><td className="py-2 pr-4">claude-opus-4-7 [max]</td><td className="py-2">6/12</td></tr>
                <tr className="border-b border-[var(--color-border)]/50"><td className="py-2 pr-4 font-medium">04 kahan_softmax</td><td className="py-2 pr-4 font-mono">0.363</td><td className="py-2 pr-4">gpt-5.5 [xhigh]</td><td className="py-2">9/12</td></tr>
                <tr className="border-b border-[var(--color-border)]/50"><td className="py-2 pr-4 font-medium">05 topk_bitonic</td><td className="py-2 pr-4 font-mono">0.042</td><td className="py-2 pr-4">gpt-5.5 [xhigh]</td><td className="py-2">5/12</td></tr>
                <tr className="border-b border-[var(--color-border)]/50"><td className="py-2 pr-4 font-medium">06 sonic_moe_swiglu</td><td className="py-2 pr-4 font-mono">0.251</td><td className="py-2 pr-4">gpt-5.5 [xhigh]</td><td className="py-2">10/12</td></tr>
                <tr><td className="py-2 pr-4 font-medium">07 w4a16_gemm</td><td className="py-2 pr-4 font-mono">0.220</td><td className="py-2 pr-4">kimi-k2.6</td><td className="py-2">10/12</td></tr>
              </tbody>
            </table>
          </div>

          <p className="leading-relaxed mb-6">
            Two of the seven problems have peaks above 0.30 (paged attention 0.602, FP8 GEMM 0.534, Kahan softmax 0.363). The rest cap below — 02 KDA Cutlass and 05 TopK Bitonic don&apos;t even break 5%. Either the references on those two are unusually well-tuned or the autonomous-agent gap is biggest there. Both, probably.
          </p>

          <h2 className="text-2xl font-bold mt-12 mb-6">The Rubric Leaks</h2>

          <p className="leading-relaxed mb-6">
            Two cells in the table promise something the benchmark doesn&apos;t actually measure. They&apos;re marked <code>★</code> for a reason.
          </p>

          <h3 className="text-xl font-bold mt-8 mb-4">01 fp8_gemm: every high-peak solution is a bf16 GEMM in disguise</h3>

          <p className="leading-relaxed mb-6">
            All five solutions that scored above <code>peak_fraction = 0.4</code> on the FP8 GEMM problem do the same thing: cast the fp8 inputs to bf16 inside the kernel and run a bf16 GEMM. Both Opus 4.7 max and GPT-5.5 xhigh explicitly pin to <code>cutlass::arch::Sm80</code> — Ampere CUTLASS, not the SM120 Blackwell FP8 tensor cores the problem name implies.
          </p>

          <p className="leading-relaxed mb-6">
            Opus&apos;s source comment is unusually direct about the choice:
          </p>

          <pre className="bg-[rgba(20,83,45,0.15)] p-4 rounded-lg overflow-x-auto mb-6 text-sm">
            <code>{`"""SM120 (Blackwell consumer) FP8 e4m3 GEMM via CUTLASS 2.x BF16 GEMM.

Strategy
--------
The reference computes  y = x.to(bf16) @ w_bf16.T, with x being fp8_e4m3fn input
and w stored as bf16. Quantizing w to fp8 introduces a per-element error of
~5% relative; over K~4096 random products that yields max-abs noise around
~0.5 — far above the 0.01 default bf16 atol/rtol used by check.py.

So we follow the codex baseline (BF16 GEMM internally) but extend it to ALL
shapes via:
  * K-padding to a multiple of 8 (handles K=4127)
  * a skinny tile config for M<=64 (handles the M=32 decode shape)
  * larger tiles + 4-stage pipeline for the bulk compute-bound shapes
"""`}</code>
          </pre>

          <p className="leading-relaxed mb-6">
            The reasoning is correct on its own terms — fp8 multiply does introduce noise that exceeds tight bf16 tolerances — but the prompt actually allows a 0.15 absolute/relative tolerance, which is loose enough for real fp8 tensor-core math to pass. The model misjudged the rubric and took the safer path. Then four other models took the same path. Same shortcut, five different model families.
          </p>

          <p className="leading-relaxed mb-6">
            <strong>What the cell numbers actually measure:</strong> bf16 kernel optimization quality on fp8-typed inputs. Not FP8 tensor core skill.
          </p>

          <h3 className="text-xl font-bold mt-8 mb-4">04 kahan_softmax: six of seven models skipped the algorithm in the problem name</h3>

          <p className="leading-relaxed mb-6">
            Kahan compensated summation is a classic numerically-stable alternative to naive summation. It&apos;s slower (extra add per iteration to track and re-inject the rounding error) but more accurate. The reference for this problem implements it; the problem name is <code>kahan_softmax</code>; the goal is a custom kernel that does the compensated path quickly.
          </p>

          <p className="leading-relaxed mb-6">
            Six of seven passing solutions skipped the compensation entirely. Naive softmax, fast. Both top-tier models (gpt-5.5 0.363, opus 0.317) took this route. The single model that actually implemented Kahan was DeepSeek V4 Pro — and it scored <em>lowest</em> of the seven passes, at 0.101, because compensated summation has real overhead that the others avoided.
          </p>

          <p className="leading-relaxed mb-6">
            DeepSeek V4 Pro&apos;s docstring is the punchline:
          </p>

          <pre className="bg-[rgba(20,83,45,0.15)] p-4 rounded-lg overflow-x-auto mb-6 text-sm">
            <code>{`"""Numerically tight softmax with Kahan compensated summation.

Single-block path for smaller vocabs (V <= 32768) where one-kernel-launch
simplicity wins.  Multi-block map-reduce for large vocabs where parallelism
across blocks is needed to saturate GPU bandwidth.

Map:    each block computes local (max, Kahan-sum-of-exp) for its chunk.
Reduce: GPU-side Kahan combination of per-block results (num_warps=1).
Norm:   each block normalizes its chunk using global (max, sum).
"""`}</code>
          </pre>

          <p className="leading-relaxed mb-6">
            The model that explicitly states the algorithm in plain English is the model that loses the cell, because the rubric leaks and everyone else skipped what was supposed to be the central challenge. The benchmark, as designed, punishes algorithmic honesty.
          </p>

          <p className="leading-relaxed mb-6">
            <strong>What the cell numbers actually measure:</strong> fast naive softmax. The 0.101 deepseek-v4-pro cell is the only one whose number measures Kahan compensated softmax skill.
          </p>

          <h2 className="text-2xl font-bold mt-12 mb-6">Publishing With the Flaws</h2>

          <p className="leading-relaxed mb-6">
            Both leaks are fixable in a few hours of problem-design work. Tighten the FP8 GEMM tolerance to a value where bf16-via-cast and real fp8-tensor-core math diverge visibly, or write a static-analysis check that detects the cast pattern. Tighten the Kahan softmax tolerance, or add a check that detects the compensation pattern in <code>solution.py</code>. Both straightforward.
          </p>

          <p className="leading-relaxed mb-6">
            I&apos;m publishing without those fixes for two reasons.
          </p>

          <p className="leading-relaxed mb-6">
            <strong>Diminishing returns on iteration.</strong> This is the second round of post-hoc design issues we&apos;ve hit on this benchmark. The first was the prompt regime — early prompts let models skip the verification step (<code>python check.py</code>) entirely; the new prompt directs them to it explicitly. Every round of iteration surfaces the next issue. There&apos;s no obvious local minimum where the benchmark stops surfacing new flaws. Publishing now with two known leaks documented inline is more honest than iterating until the next discovery, then publishing.
          </p>

          <p className="leading-relaxed mb-6">
            <strong>The flaws ARE the finding.</strong> &ldquo;Five frontier models all took the same bf16 shortcut on FP8 GEMM&rdquo; is a result. &ldquo;Six of seven models skipped the algorithm the problem name describes when the rubric didn&apos;t enforce it&rdquo; is a result. Those tell us something true about how frontier models behave under autonomous-agent evaluation when given a goal and a measurable proxy: when the proxy can be hit without doing the work the goal implies, they&apos;ll usually do that. Publishing the leaderboard with these footnoted is the point.
          </p>

          <h2 className="text-2xl font-bold mt-12 mb-6">What&apos;s Different from KernelBench v3</h2>

          <p className="leading-relaxed mb-6">
            Same author (me), different shape.
          </p>

          <ul className="space-y-2 mb-8 list-disc list-inside text-[var(--color-fg-muted)]">
            <li><strong>One GPU instead of three.</strong> RTX PRO 6000 Blackwell Workstation (sm_120, 96 GB GDDR7, 1.8 TB/s). Single-GPU evaluation removes the cross-architecture comparison axis from v3 but lets us push much harder per-cell on a more interesting Blackwell consumer-tier chip.</li>
            <li><strong>Seven problems instead of 43-58.</strong> Each is hand-designed around a modern inference primitive (FP8 GEMM, paged attention, MoE up-projection, INT4 weight-only GEMM, etc.). Per-trial L2 flush, 30-trial median, ten-warmup absorbing torch.compile CUDA-graph capture and Triton autotune.</li>
            <li><strong>Real coding-agent CLIs as the harness, not a custom KernelBench loop.</strong> Each model runs through whatever its native developer-facing CLI is — Claude Code for Anthropic, codex CLI for OpenAI, Kimi CLI for Moonshot, opencode for everyone else. The model is &ldquo;dropped into a directory&rdquo; with <code>reference.py</code>, <code>check.py</code>, <code>benchmark.py</code>, and a prompt; it has full filesystem access, can clone repos, install packages, profile, and iterate. This matches how engineers actually use these tools.</li>
            <li><strong>Wall-clock budgets, not turn limits.</strong> 45 minutes per (model, problem) run. Models with verbose tool-use patterns (lots of filesystem exploration) aren&apos;t penalized just for being chatty; they trade exploration for kernel-iteration time within the budget.</li>
            <li><strong>peak_fraction, not raw speedup.</strong> Speedup ratios are easy to game (slow the baseline, inflate the ratio). <code>peak_fraction</code> is grounded in the actual physical limits of the hardware: the fraction of the relevant tensor-core or DRAM bandwidth ceiling the kernel achieved. Harder to game, more honest.</li>
            <li><strong>Per-cell annotations.</strong> Every cell where something interesting happened — rubric leak, clever implementation, harness-integration failure — has a YAML annotation file in the source repo with verdict, summary, pull quotes from <code>solution.py</code>, and an &ldquo;implication&rdquo; statement. Thirteen annotations as of launch.</li>
          </ul>

          <h2 className="text-2xl font-bold mt-12 mb-6">The Seven Problems</h2>

          <ol className="space-y-3 mb-8 list-decimal list-inside text-[var(--color-fg-muted)]">
            <li><strong>fp8_gemm</strong> — fp8_e4m3 activations × fp8_e4m3 weights → bf16 output, four shapes including M=32 decode and Llama-3 up-projection. Targets SM120 FP8 tensor cores. (See rubric leak above.)</li>
            <li><strong>kda_cutlass</strong> — KernelDensityAnalysis-style operation that requires a CUTLASS-built custom kernel. The deck&apos;s hardest problem; no model broke 5% peak.</li>
            <li><strong>paged_attention</strong> — vLLM-style decode-time paged KV cache attention. Online softmax over pages, GQA-aware. Opus&apos;s 0.602 winner is real FlashDecoding-style Triton.</li>
            <li><strong>kahan_softmax</strong> — numerically-stable softmax via Kahan compensated summation. (See second rubric leak above.)</li>
            <li><strong>topk_bitonic</strong> — bitonic top-K selection. Second-hardest of the deck — top peak 0.042.</li>
            <li><strong>sonic_moe_swiglu</strong> — Sonic-style MoE up-projection with SwiGLU activation. Tests grouped GEMM + activation fusion.</li>
            <li><strong>w4a16_gemm</strong> — int4-packed weight, fp16 activation GEMM with on-the-fly unpack and dequantize. All eight passing solutions inline the unpacking inside the kernel; none pre-unpack at init.</li>
          </ol>

          <h2 className="text-2xl font-bold mt-12 mb-6">Hardware</h2>

          <p className="leading-relaxed mb-6">
            RTX PRO 6000 Blackwell Workstation. Single-GPU. The relevant ceilings:
          </p>

          <div className="overflow-x-auto mb-6 text-sm">
            <table className="w-full">
              <thead>
                <tr className="border-b border-[var(--color-border)]">
                  <th className="text-left py-2 pr-4">Resource</th>
                  <th className="text-left py-2">Peak</th>
                </tr>
              </thead>
              <tbody className="text-[var(--color-fg-muted)] font-mono">
                <tr className="border-b border-[var(--color-border)]/50"><td className="py-2 pr-4">DRAM bandwidth (GDDR7)</td><td className="py-2">1.8 TB/s</td></tr>
                <tr className="border-b border-[var(--color-border)]/50"><td className="py-2 pr-4">L2 cache</td><td className="py-2">96 MB</td></tr>
                <tr className="border-b border-[var(--color-border)]/50"><td className="py-2 pr-4">VRAM</td><td className="py-2">96 GB</td></tr>
                <tr className="border-b border-[var(--color-border)]/50"><td className="py-2 pr-4">FP4 / FP8 / INT4 dense TFLOPS</td><td className="py-2">800 / 400 / 800</td></tr>
                <tr className="border-b border-[var(--color-border)]/50"><td className="py-2 pr-4">BF16 / FP16 dense TFLOPS</td><td className="py-2">200</td></tr>
                <tr><td className="py-2 pr-4">FP32 SIMT TFLOPS</td><td className="py-2">12</td></tr>
              </tbody>
            </table>
          </div>

          <p className="leading-relaxed mb-6">
            <code>peak_fraction</code> is computed against the relevant ceiling per problem — DRAM bandwidth for memory-bound (paged attention, kahan softmax), tensor-core peak for compute-bound (FP8 GEMM, MoE up-proj, w4a16). 0.6 on a memory-bound problem is genuinely strong; 0.6 on a compute-bound problem would be approaching production-quality.
          </p>

          <h2 className="text-2xl font-bold mt-12 mb-6">Methodology Notes</h2>

          <h3 className="text-xl font-bold mt-8 mb-4">Per-trial benchmarking</h3>

          <p className="leading-relaxed mb-6">
            Centralized in <code>src/eval/timing.py</code> so every problem&apos;s <code>benchmark.py</code> uses the same cadence: 10 warmup calls (absorbs Triton autotune and torch.compile <code>reduce-overhead</code> CUDA-graph capture), per-trial L2 flush via 128 MB write to a scratch tensor (the L2 is 96 MB, so 128 MB strictly evicts), CUDA Events with <code>synchronize</code> after <code>record</code> and before <code>elapsed_time</code>, median over 30 trials.
          </p>

          <p className="leading-relaxed mb-6">
            One known measurement bias: <code>torch.compile(mode=&quot;reduce-overhead&quot;)</code> gets CUDA graphs which eliminate launch overhead. Custom Triton/CUDA kernels do not. On small shapes where launch overhead dominates, this favors the compile baseline — the cost of using <code>torch.compile</code> as the &ldquo;compiled reference&rdquo; line.
          </p>

          <h3 className="text-xl font-bold mt-8 mb-4">Provider pinning</h3>

          <p className="leading-relaxed mb-6">
            OpenRouter dispatches to whichever inference provider has capacity. Many serve int4/fp4-quantized weights of frontier models. Running a kernel benchmark against int4 GLM is not the same as the lab&apos;s full-precision endpoint. Every OpenRouter-routed model is pinned to its native lab provider via <code>extraBody.provider.order</code> with <code>allow_fallbacks: false</code>. The benchmark fails loudly if no integrity-clean route is available — better than silently routing to a quantized third party.
          </p>

          <p className="leading-relaxed mb-6">
            One model in the matrix (qwen3.6-35b-a3b) has no Alibaba-served endpoint on OpenRouter; only AtlasCloud and Parasail serve it, both fp8. We added them to the provider order, then discovered neither advertises tool-use capability — the agent harness can&apos;t reach the model at all. That&apos;s the <code>0/7 ERR</code> row in the leaderboard. Infrastructure block, not capability gap. Documented and skipped.
          </p>

          <h3 className="text-xl font-bold mt-8 mb-4">Workspace leak fix</h3>

          <p className="leading-relaxed mb-6">
            opencode <code>--pure</code> turned out to mean &ldquo;no external plugins,&rdquo; not &ldquo;sandboxed filesystem.&rdquo; A mid-sweep audit caught models reading repo internals (<code>src/hardware/rtx_pro_6000.py</code> with the peak TFLOPS table the prompt was supposed to keep secret, <code>src/eval/correctness.py</code> with the tolerance lookup) plus the user&apos;s own Claude Code skill atlas. Adding <code>permission.external_directory: &quot;deny&quot;</code> to the user-level opencode config closed the repo-internal leak. The user&apos;s installed skills directory was a separate auto-permitted path; that&apos;s closed too now (the skills directory was moved out, with the canonical copy mirrored elsewhere). Cross-harness audit still pending — claude-code and codex have the same architectural lack of FS isolation, but those weren&apos;t the primary leak vectors in this run.
          </p>

          <h3 className="text-xl font-bold mt-8 mb-4">N=1 and variance</h3>

          <p className="leading-relaxed mb-6">
            Every cell is a single trial. We saw two reversals on the same (model, problem) within 24 hours during the initial sweeps — DeepSeek Flash on TopK regressed from PASS to FAIL day-over-day under the same prompt; Qwen 3.6 27B&apos;s shakedown was 0/7, a same-day rerun was 1/7. LLM nondeterminism is real on this benchmark and N=1 isn&apos;t enough for any cell&apos;s value to be load-bearing alone. Future official runs will report N≥2 with variance bands.
          </p>

          <h2 className="text-2xl font-bold mt-12 mb-6">What Comes Next</h2>

          <ul className="space-y-2 mb-8 list-disc list-inside text-[var(--color-fg-muted)]">
            <li><strong>Close the two rubric leaks.</strong> Tighter tolerance on fp8_gemm and kahan_softmax, or static-analysis checks that detect the shortcut patterns. Then re-run the affected cells and publish the diff.</li>
            <li><strong>Multi-trial cells.</strong> Re-run the full grid at N=3 with variance bands. The reversals during initial sweeps tell us the leaderboard is noisier than single-cell numbers suggest.</li>
            <li><strong>Universal sandboxing.</strong> bwrap or firejail wrapper around every harness invocation, bind-mounting only the workspace dir. Closes the FS leak symmetrically across claude-code, codex, kimi, and opencode.</li>
            <li><strong>More problems.</strong> Seven is the floor. The deck wants more compute-bound problems where peaks should be reachable, more long-context problems, and at least one where the reference is already a hand-tuned production kernel (so &ldquo;beats baseline&rdquo; means beating production code).</li>
          </ul>

          <h2 className="text-2xl font-bold mt-12 mb-6">Source</h2>

          <p className="leading-relaxed mb-6">
            Everything is in <Link href="https://github.com/Infatoshi/KernelBench-Hard" className="text-[var(--color-fg-bright)] hover:underline">github.com/Infatoshi/KernelBench-Hard</Link>: leaderboard JSON, per-cell annotations, problem definitions, solution checks, the harness scripts, the dev log of decisions and dead ends. The website renders a fixed snapshot; the source repo is the live truth.
          </p>
        </article>
      
  
    </div>
  )
}
