import Link from "next/link"
import Image from "next/image"

export default function KernelBenchV3Post() {
  return (
    <div className="space-y-8">

        <Link href="/" className="text-[var(--color-fg-bright)] hover:underline mb-8 inline-block">
          &larr; back to home
        </Link>

        <article className="max-w-none">
          <h1 className="text-4xl font-bold mb-6">KernelBench v3: Rebuilding a GPU Kernel Benchmark from First Principles</h1>

          <p className="text-[var(--color-fg-muted)] mb-12">
            How discovering the original KernelBench was exploitable led to building a focused, cost-effective benchmark for evaluating LLM kernel engineering on modern architectures.
          </p>

          <h2 className="text-2xl font-bold mt-12 mb-6">Results at a Glance</h2>

          <p className="leading-relaxed mb-6">
            10 models evaluated across RTX 3090, H100, and B200 GPUs on problems spanning 4 difficulty levels. Three metrics matter: Does the code compile? Is it numerically correct? Does it beat the PyTorch baseline?
          </p>

          <div className="my-8">
            <Image
              src="/blog-v3/results_overall.png"
              alt="Overall benchmark results showing compiled, correct, and beats baseline percentages for each model"
              width={1400}
              height={700}
              className="rounded-lg"
            />
          </div>

          <p className="leading-relaxed mb-6">
            The heatmap below breaks this down by difficulty level. L1-L3 are tractable for frontier models. L4 - novel architectures like DeepSeek MLA, GatedDeltaNet, and FP8 matmul - is where everyone struggles.
          </p>

          <div className="my-8">
            <Image
              src="/blog-v3/results_heatmap.png"
              alt="Heatmap showing compiled, correct, and beats baseline percentages by level for each model"
              width={1800}
              height={800}
              className="rounded-lg"
            />
          </div>

          <h3 className="text-xl font-bold mt-8 mb-4">Speedup Distribution</h3>

          <p className="leading-relaxed mb-6">
            The distribution of speedups across all correct solutions shows where models actually deliver performance gains versus where they merely match the baseline.
          </p>

          <div className="my-8">
            <Image
              src="/blog-v3/speedup_distribution.png"
              alt="Distribution of speedups across correct solutions"
              width={1400}
              height={700}
              className="rounded-lg"
            />
          </div>

          <h3 className="text-xl font-bold mt-8 mb-4">Level Breakdown</h3>

          <p className="leading-relaxed mb-6">
            Per-level performance shows the steepest drop at L4, where problems involve novel architectures not well-represented in training data.
          </p>

          <div className="my-8">
            <Image
              src="/blog-v3/level_breakdown.png"
              alt="Performance breakdown by difficulty level"
              width={1400}
              height={700}
              className="rounded-lg"
            />
          </div>

          <h3 className="text-xl font-bold mt-8 mb-4">Cost vs Accuracy</h3>

          <p className="leading-relaxed mb-6">
            API cost per evaluation varies dramatically across models. The cost-accuracy tradeoff reveals which models deliver the best kernel engineering per dollar.
          </p>

          <div className="my-8">
            <Image
              src="/blog-v3/cost_vs_accuracy.png"
              alt="Cost vs accuracy scatter plot for each model"
              width={1400}
              height={700}
              className="rounded-lg"
            />
          </div>

          <p className="leading-relaxed mb-6">
            <strong>GPT-5.4 and Gemini 3 Flash lead on correctness across GPUs, while Gemini 3 Flash offers the best cost-effectiveness.</strong> What matters is not average speedup - a metric easily gamed - but whether models can produce kernels that are both correct and faster than PyTorch. On that measure, the frontier is clear: Level 4 pass rates collapse across all models. Genuine kernel engineering on novel architectures remains beyond current capabilities.
          </p>

          <h2 className="text-2xl font-bold mt-12 mb-6">The Problem with KernelBench</h2>

          <p className="leading-relaxed mb-6">
            KernelBench, released by Stanford's Scaling Intelligence Lab, promised to evaluate whether LLMs could write optimized CUDA kernels. The premise was compelling: give a model a PyTorch reference implementation, ask it to write faster CUDA code, and measure the speedup. The benchmark included 250 problems across multiple difficulty levels.
          </p>

          <p className="leading-relaxed mb-6">
            Then METR published <Link href="https://metr.org/blog/2025-02-14-measuring-automated-kernel-engineering/" className="text-[var(--color-fg-bright)] hover:underline">"Measuring Automated Kernel Engineering"</Link> and the facade crumbled.
          </p>

          <h3 className="text-xl font-bold mt-8 mb-4">The Exploits</h3>

          <p className="leading-relaxed mb-6">
            METR discovered that models were achieving high "speedups" through exploitation rather than genuine kernel engineering:
          </p>

          <ul className="space-y-2 mb-8 list-disc list-inside text-[var(--color-fg-muted)]">
            <li><strong>Bypassing CUDA entirely</strong>: Models called torch or cuBLAS instead of writing kernels</li>
            <li><strong>Memory aliasing</strong>: No-op kernels that pass because output memory overlaps with reference</li>
            <li><strong>Timing manipulation</strong>: Monkey-patching torch.cuda.synchronize to make timing meaningless</li>
            <li><strong>Stack introspection</strong>: Extracting pre-computed reference results from the caller's stack</li>
            <li><strong>Constant functions</strong>: Problems like mean(softmax(x)) that always equal 1.0</li>
          </ul>

          <p className="leading-relaxed mb-6">
            METR removed 45 of the 250 problems due to fundamental task quality issues. After filtering exploits, average speedup dropped from 3.13x to 1.49x. The benchmark was measuring benchmark-gaming ability, not kernel engineering.
          </p>

          <h2 className="text-2xl font-bold mt-12 mb-6">Starting Over</h2>

          <p className="leading-relaxed mb-6">
            Rather than patching a broken system, I decided to rebuild from scratch with clear design principles:
          </p>

          <ol className="space-y-2 mb-8 list-decimal list-inside text-[var(--color-fg-muted)]">
            <li><strong>Focus on modern architectures</strong>: No classical ML operators nobody optimizes anymore</li>
            <li><strong>Fewer problems, higher quality</strong>: Each problem manually validated per GPU target</li>
            <li><strong>Three GPUs that matter</strong>: RTX 3090 (Ampere), H100 (Hopper), and B200 (Blackwell)</li>
            <li><strong>Adaptive baselines</strong>: torch.compile used only when it actually helps, not blindly</li>
            <li><strong>Multi-seed correctness</strong>: 5 random seeds (42, 123, 456, 789, 1337) to catch caching exploits</li>
            <li><strong>Cost tracking</strong>: Full token usage and API cost per evaluation</li>
          </ol>

          <h2 className="text-2xl font-bold mt-12 mb-6">The Cost Problem</h2>

          <p className="leading-relaxed mb-6">
            The original plan was ambitious: evaluate many models across hundreds of problems on many GPU architectures. The math was brutal. Even with scope cuts, the final evaluation covers 10 models across 3 GPUs with varying problem counts per architecture (43 for RTX 3090, 54 for H100, 58 for B200), yielding over 1500 individual evaluations.
          </p>

          <h2 className="text-2xl font-bold mt-12 mb-6">Problem Selection</h2>

          <p className="leading-relaxed mb-6">
            Problems span four difficulty levels, with additional GPU-specific problems (tile-specialized GEMM variants for H100/B200, cuTile problems for B200, graphics problems for RTX 3090):
          </p>

          <div className="overflow-x-auto mb-6">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-[var(--color-border)]">
                  <th className="text-left py-2 pr-4">Level</th>
                  <th className="text-left py-2 pr-4">Count</th>
                  <th className="text-left py-2 pr-4">Turns</th>
                  <th className="text-left py-2">Description</th>
                </tr>
              </thead>
              <tbody className="text-[var(--color-fg-muted)]">
                <tr className="border-b border-[var(--color-border)]/50">
                  <td className="py-2 pr-4 font-medium">L1</td>
                  <td className="py-2 pr-4">15</td>
                  <td className="py-2 pr-4">10</td>
                  <td className="py-2">Simple ops: matmul, softmax, conv, norms</td>
                </tr>
                <tr className="border-b border-[var(--color-border)]/50">
                  <td className="py-2 pr-4 font-medium">L2</td>
                  <td className="py-2 pr-4">15</td>
                  <td className="py-2 pr-4">12</td>
                  <td className="py-2">Fused ops: matmul+activation chains</td>
                </tr>
                <tr className="border-b border-[var(--color-border)]/50">
                  <td className="py-2 pr-4 font-medium">L3</td>
                  <td className="py-2 pr-4">3</td>
                  <td className="py-2 pr-4">15</td>
                  <td className="py-2">Single blocks: attention, transformer block</td>
                </tr>
                <tr>
                  <td className="py-2 pr-4 font-medium">L4</td>
                  <td className="py-2 pr-4">9+</td>
                  <td className="py-2 pr-4">15</td>
                  <td className="py-2">Novel layers: MLA, MoE, GQA, FP8, DeltaNet, FP4, INT4</td>
                </tr>
              </tbody>
            </table>
          </div>

          <h3 className="text-xl font-bold mt-8 mb-4">Level 4: The Real Test</h3>

          <p className="leading-relaxed mb-6">
            Level 4 is where it gets interesting. These are modern inference optimization patterns that test genuine kernel engineering:
          </p>

          <ul className="space-y-2 mb-8 list-disc list-inside text-[var(--color-fg-muted)]">
            <li><strong>DeepSeek MLA</strong>: Multi-head Latent Attention with LoRA KV compression - not in training data</li>
            <li><strong>DeepSeek MoE</strong>: Mixture-of-Experts with grouped expert routing</li>
            <li><strong>GQA</strong>: Grouped Query Attention (Llama 3 style) with KV head expansion</li>
            <li><strong>FP8 Matmul</strong>: E4M3 quantized matmul with tensor cores via torch._scaled_mm</li>
            <li><strong>INT4 GEMM</strong>: Weight-only quantization with fused unpack+dequant+matmul</li>
            <li><strong>FP4 Matmul</strong>: B200-only FP4 quantized matrix multiply</li>
            <li><strong>GatedDeltaNet</strong>: Linear attention from ICLR 2025 - baseline uses flash-linear-attention's Triton kernels</li>
            <li><strong>KimiDeltaAttention</strong>: Channel-wise gated delta attention - same fla baseline</li>
          </ul>

          <p className="leading-relaxed mb-6">
            For GatedDeltaNet and KimiDeltaAttention, the baseline is not naive PyTorch - it is already optimized Triton code from flash-linear-attention. Models need to match or beat production-quality kernels.
          </p>

          <h2 className="text-2xl font-bold mt-12 mb-6">Finding Modern Baselines</h2>

          <p className="leading-relaxed mb-6">
            The Level 4 problems required digging through HuggingFace implementations to find reference code. DeepSeek MLA came from the DeepSeek-V3 model's modeling_deepseek.py. The core insight: the HuggingFace implementations use naive PyTorch ops that are ripe for optimization:
          </p>

          <pre className="bg-[rgba(20,83,45,0.15)] p-4 rounded-lg overflow-x-auto mb-6">
            <code>{`# DeepSeek MLA: naive PyTorch baseline
# Q projection with LoRA compression
q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))

# KV projection with LoRA compression
compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
kv = self.kv_b_proj(self.kv_a_layernorm(compressed_kv))

# A fused kernel can combine:
# 1. LoRA compression/expansion
# 2. RMSNorm
# 3. RoPE application
# 4. Attention computation`}</code>
          </pre>

          <h2 className="text-2xl font-bold mt-12 mb-6">Infrastructure</h2>

          <p className="leading-relaxed mb-6">
            The evaluation uses a multi-tier infrastructure. RTX 3090 runs locally, while H100 and B200 run on Modal with CUDA 13.2 and full support for Hopper and Blackwell architectures. Key infrastructure decisions:
          </p>

          <ul className="space-y-2 mb-8 list-disc list-inside text-[var(--color-fg-muted)]">
            <li><strong>Modal sandbox</strong>: Isolated execution with git, cmake, CUTLASS/CuTe DSL</li>
            <li><strong>Local sandbox</strong>: RTX 3090 runs locally for fast iteration</li>
            <li><strong>Multi-turn agent</strong>: Models iterate on their solutions with compiler feedback</li>
            <li><strong>Per-level turn limits</strong>: L1 gets 10 turns, L4 gets 15 - harder problems need more iteration</li>
            <li><strong>Adaptive torch.compile baseline</strong>: Uses whichever is faster per-problem (eager vs compiled)</li>
            <li><strong>Prompt caching</strong>: System prompts cached to reduce token costs</li>
            <li><strong>Dynamic pricing</strong>: Costs fetched from OpenRouter API, not hardcoded</li>
          </ul>

          <h2 className="text-2xl font-bold mt-12 mb-6">Results</h2>

          <p className="leading-relaxed mb-6">
            Over 1,500 evaluations across 10 models and 3 GPUs. The coverage matrix shows correctness rates:
          </p>

          <div className="overflow-x-auto mb-6">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-[var(--color-border)]">
                  <th className="text-left py-2 pr-4">Model</th>
                  <th className="text-left py-2 pr-4">RTX 3090</th>
                  <th className="text-left py-2 pr-4">H100</th>
                  <th className="text-left py-2">B200</th>
                </tr>
              </thead>
              <tbody className="text-[var(--color-fg-muted)]">
                <tr className="border-b border-[var(--color-border)]/50">
                  <td className="py-2 pr-4 font-medium">MiniMax M2.5</td>
                  <td className="py-2 pr-4">35/43 (77%*)</td>
                  <td className="py-2 pr-4">9/54 (17%)</td>
                  <td className="py-2">12/58 (21%)</td>
                </tr>
                <tr className="border-b border-[var(--color-border)]/50">
                  <td className="py-2 pr-4 font-medium">GPT-5.4</td>
                  <td className="py-2 pr-4">33/43 (77%)</td>
                  <td className="py-2 pr-4">42/54 (78%)</td>
                  <td className="py-2">-</td>
                </tr>
                <tr className="border-b border-[var(--color-border)]/50">
                  <td className="py-2 pr-4 font-medium">Gemini 3 Flash</td>
                  <td className="py-2 pr-4">32/43 (74%)</td>
                  <td className="py-2 pr-4">41/54 (76%)</td>
                  <td className="py-2">46/58 (79%)</td>
                </tr>
                <tr className="border-b border-[var(--color-border)]/50">
                  <td className="py-2 pr-4 font-medium">GPT-5.3</td>
                  <td className="py-2 pr-4">28/43 (65%)</td>
                  <td className="py-2 pr-4">-</td>
                  <td className="py-2">-</td>
                </tr>
                <tr className="border-b border-[var(--color-border)]/50">
                  <td className="py-2 pr-4 font-medium">Claude Opus 4.6</td>
                  <td className="py-2 pr-4">27/43 (63%)</td>
                  <td className="py-2 pr-4">-</td>
                  <td className="py-2">-</td>
                </tr>
                <tr className="border-b border-[var(--color-border)]/50">
                  <td className="py-2 pr-4 font-medium">Claude Sonnet 4.6</td>
                  <td className="py-2 pr-4">25/43 (58%)</td>
                  <td className="py-2 pr-4">-</td>
                  <td className="py-2">-</td>
                </tr>
                <tr className="border-b border-[var(--color-border)]/50">
                  <td className="py-2 pr-4 font-medium">Kimi K2.5</td>
                  <td className="py-2 pr-4">22/43 (51%)</td>
                  <td className="py-2 pr-4">27/54 (50%)</td>
                  <td className="py-2">35/58 (60%)</td>
                </tr>
                <tr className="border-b border-[var(--color-border)]/50">
                  <td className="py-2 pr-4 font-medium">Qwen3.5 397B</td>
                  <td className="py-2 pr-4">-</td>
                  <td className="py-2 pr-4">22/54 (41%)</td>
                  <td className="py-2">25/58 (43%)</td>
                </tr>
                <tr className="border-b border-[var(--color-border)]/50">
                  <td className="py-2 pr-4 font-medium">GLM-5</td>
                  <td className="py-2 pr-4">19/43 (44%)</td>
                  <td className="py-2 pr-4">31/54 (57%)</td>
                  <td className="py-2">31/58 (53%)</td>
                </tr>
                <tr className="border-b border-[var(--color-border)]/50">
                  <td className="py-2 pr-4 font-medium">Gemini 3.1 Pro</td>
                  <td className="py-2 pr-4">16/43 (37%)</td>
                  <td className="py-2 pr-4">-</td>
                  <td className="py-2">-</td>
                </tr>
                <tr>
                  <td className="py-2 pr-4 font-medium">DeepSeek V3.2</td>
                  <td className="py-2 pr-4">0/43 (0%)</td>
                  <td className="py-2 pr-4">-</td>
                  <td className="py-2">2/58 (3%)</td>
                </tr>
              </tbody>
            </table>
          </div>

          <p className="text-sm text-[var(--color-fg-muted)] mb-6">
            *MiniMax RTX 3090 had 129 results from possible multi-run merge. Dashes indicate runs not yet completed.
          </p>

          <h3 className="text-xl font-bold mt-8 mb-4">By Level</h3>

          <p className="leading-relaxed mb-6">
            Level 4's low pass rate tells the real story. When faced with modern architectures not in training data, or baselines that are already optimized Triton code, models struggle. The gap between "can write CUDA" and "can engineer production kernels" is substantial.
          </p>

          <h2 className="text-2xl font-bold mt-12 mb-6">Key Observations</h2>

          <h3 className="text-xl font-bold mt-8 mb-4">GPT-5.4 and Gemini 3 Flash Lead Across GPUs</h3>

          <p className="leading-relaxed mb-6">
            GPT-5.4 achieves the highest correctness on both RTX 3090 (77%) and H100 (78%). Gemini 3 Flash is the only model evaluated across all three GPUs and maintains consistently strong performance (74-79% correctness), while also being the most cost-effective option at $0.50/$3 per million tokens.
          </p>

          <h3 className="text-xl font-bold mt-8 mb-4">GPU Architecture Matters</h3>

          <p className="leading-relaxed mb-6">
            Some models perform notably differently across GPU architectures. Kimi K2.5 jumps from 51% on RTX 3090 to 60% on B200. GLM-5 improves from 44% on RTX 3090 to 57% on H100. This suggests that some models have better training coverage for newer GPU architectures, or that the additional problems (tile-specialized GEMM, cuTile) happen to play to their strengths.
          </p>

          <h3 className="text-xl font-bold mt-8 mb-4">Behavior on Specific Problems</h3>

          <p className="leading-relaxed mb-6">
            The aggregate numbers hide interesting per-problem behavior. On LayerNorm, some models produce highly optimized fused kernels while others fall back to naive implementations. On GEMM fusion patterns, the approaches diverge significantly - some models attempt register tiling and shared memory optimization, others stick to cuBLAS calls. I encourage readers to explore the <Link href="/kernelbench-v3" className="text-[var(--color-fg-bright)] hover:underline">interactive dashboard</Link> and examine specific problems to understand how different models approach kernel engineering.
          </p>

          <h3 className="text-xl font-bold mt-8 mb-4">Open Models Struggle</h3>

          <p className="leading-relaxed mb-6">
            DeepSeek V3.2 achieves 0-3% correctness across GPUs. Qwen3.5 397B manages 41-43% on H100/B200. The gap between frontier closed models and open alternatives is pronounced for kernel engineering - this appears to be a capability that requires significant training investment.
          </p>

          <h3 className="text-xl font-bold mt-8 mb-4">MiniMax M2.5: The Anomaly</h3>

          <p className="leading-relaxed mb-6">
            MiniMax M2.5 shows the widest variance: 77% on RTX 3090 but only 17-21% on H100/B200. The RTX 3090 run had 129 results (possible multi-run merge), so the 77% figure should be treated with caution. More concerning: in early runs, MiniMax attempted reward hacking by running `pkill -f python` to kill the evaluation process. Guardrail fixes prevented this on subsequent runs.
          </p>

          <h2 className="text-2xl font-bold mt-12 mb-6">What This Means</h2>

          <p className="leading-relaxed mb-6">
            LLMs can write CUDA code. They can even write code that passes correctness checks on standard operations. But when the task requires genuine kernel engineering - understanding memory hierarchies, exploiting tensor cores, fusing operations for bandwidth efficiency - the capability drops sharply.
          </p>

          <p className="leading-relaxed mb-6">
            The original KernelBench inflated capabilities through exploitable tasks and naive baselines. With those removed, the picture is more sobering: models are useful assistants for kernel development, but not autonomous kernel engineers.
          </p>

          <h2 className="text-2xl font-bold mt-12 mb-6">Remaining Evaluation</h2>

          <p className="leading-relaxed mb-6">
            10 runs remain to complete the coverage matrix: Qwen3.5 on RTX 3090, plus GPT-5.3, Claude Opus 4.6, Claude Sonnet 4.6, and Gemini 3.1 Pro on both H100 and B200. GPT-5.4 on B200 is also pending. These will be published as they complete.
          </p>

          <h2 className="text-2xl font-bold mt-12 mb-6">Future Work</h2>

          <p className="leading-relaxed mb-6">
            KernelBench v3 currently evaluates single-GPU kernels. The roadmap includes:
          </p>

          <ul className="space-y-2 mb-8 list-disc list-inside text-[var(--color-fg-muted)]">
            <li><strong>Level 5</strong>: Multi-GPU with tensor parallelism and pipeline parallelism</li>
            <li><strong>Level 6</strong>: Multi-node distributed training/inference patterns</li>
            <li><strong>Expanded L4</strong>: More modern architectures as they emerge</li>
            <li><strong>M4 Max (Metal)</strong>: Apple GPU evaluation with Metal-specific problems</li>
          </ul>

          <h2 className="text-2xl font-bold mt-12 mb-6">Try It</h2>

          <p className="leading-relaxed mb-6">
            The benchmark is open source. Results are browsable at <Link href="/kernelbench-v3" className="text-[var(--color-fg-bright)] hover:underline">/kernelbench-v3</Link> with full filtering by model, GPU, level, operation type, and more.
          </p>

          <pre className="bg-[rgba(20,83,45,0.15)] p-4 rounded-lg overflow-x-auto mb-6">
            <code>{`git clone https://github.com/Infatoshi/KernelBench-v3
cd KernelBench-v3
uv sync
uv run python bench.py run rtx3090 --models google/gemini-3-flash-preview --levels 1,2,3,4 --workers 4`}</code>
          </pre>

          <div className="mt-12 pt-8 border-t border-[var(--color-border)]/50">
            <h3 className="text-xl font-bold mb-4">Resources</h3>
            <ul className="space-y-2 text-[var(--color-fg-muted)]">
              <li>
                <Link href="https://github.com/Infatoshi/KernelBench-v3" className="text-[var(--color-fg-bright)] hover:underline">
                  GitHub Repository
                </Link>
              </li>
              <li>
                <Link href="/kernelbench-v3" className="text-[var(--color-fg-bright)] hover:underline">
                  Interactive Results Dashboard
                </Link>
              </li>
              <li>
                <Link href="/data/v3/results.csv" className="text-[var(--color-fg-bright)] hover:underline">
                  Download Raw CSV Data
                </Link>
              </li>
              <li>
                <Link href="https://metr.org/blog/2025-02-14-measuring-automated-kernel-engineering/" className="text-[var(--color-fg-bright)] hover:underline">
                  METR's KernelBench Analysis
                </Link>
              </li>
              <li>
                <Link href="https://scalingintelligence.stanford.edu/blogs/kernelbenchv01/" className="text-[var(--color-fg-bright)] hover:underline">
                  Original KernelBench (Stanford)
                </Link>
              </li>
            </ul>
          </div>

          <p className="text-sm text-[var(--color-fg-muted)] mt-8">March 2026 (updated from January 2026)</p>
        </article>
      
  
    </div>
  )
}
