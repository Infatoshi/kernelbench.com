import Link from "next/link"
import { readdir } from "node:fs/promises"
import { join } from "node:path"
import {
  loadLeaderboard,
  loadAnnotations,
  loadBaselines,
  type Model,
} from "@/lib/data"

async function loadAvailableViewers(): Promise<Set<string>> {
  try {
    const entries = await readdir(join(process.cwd(), "public/runs"))
    return new Set(
      entries
        .filter((n) => n.endsWith(".html"))
        .map((n) => n.slice(0, -5)),
    )
  } catch {
    return new Set()
  }
}

const PROBLEMS = [
  { key: "01_fp8_gemm", short: "01 fp8" },
  { key: "02_kda_cutlass", short: "02 kda" },
  { key: "03_paged_attention", short: "03 paged" },
  { key: "05_topk_bitonic", short: "05 topk" },
  { key: "06_sonic_moe_swiglu", short: "06 moe" },
  { key: "07_w4a16_gemm", short: "07 w4a16" },
  { key: "09_fmha_preattn_mrope", short: "09 mrope" },
  { key: "10_patch_embed_conv3d_gemm", short: "10 patch" },
]

const VISIBLE_MODEL_LABELS = new Set([
  "codex/gpt-5.5 [2026-05-28 finish xhigh]",
  "claude/claude-opus-4-7 [2026-05-28 finish max]",
  "kimi/kimi-k2.6",
  "opencode/openrouter-pinned/xiaomi/mimo-v2.5-pro",
  "opencode/deepseek/deepseek-v4-flash",
  "opencode/deepseek/deepseek-v4-pro",
  "opencode/zai/glm-5.1",
  "droid/zai/glm-5.1 [2026-05-08]",
  "zai-claude/glm-5.1 [2026-05-13]",
])

const VISIBLE_MODEL_PREFIXES = [
  "claude/claude-opus-4-8 [2026-05-28 opus48-grok",
  "claude/claude-opus-4-6 [2026-06-04 opus46",
  "cursor/composer-2.5-fast [2026-05-28 finish",
  "gemini/gemini-3.5-flash [2026-05-28 finish",
  "grok/grok-build [2026-05-28 opus48-grok",
  "minimax-claude/MiniMax-M3 [2026-06-01",
]

const FP8_CONSTRAINT_FIXED_RUNS = [
  {
    model: "Claude Opus 4.6",
    route: "claude",
    result: "FAIL",
    elapsed: "18.5m",
    note: "large_input stress failed, max_abs_diff=4",
  },
  {
    model: "Claude Opus 4.7",
    route: "claude",
    result: "FAIL",
    elapsed: "21.6m",
    note: "check_failed under real FP8 constraint",
  },
  {
    model: "Claude Opus 4.8",
    route: "claude",
    result: "FAIL",
    elapsed: "41.8m",
    note: "large_input K=4127 failed, max_abs_diff=4",
  },
  {
    model: "GPT-5.5",
    route: "codex",
    result: "FAIL",
    elapsed: "6.8m",
    note: "nominal tolerance failed on first fixed run",
  },
  {
    model: "DeepSeek V4 Flash",
    route: "opencode",
    result: "FAIL",
    elapsed: "4.1m",
    note: "nominal tolerance failed, max_abs_diff around 0.53",
  },
  {
    model: "DeepSeek V4 Pro",
    route: "opencode",
    result: "FAIL",
    elapsed: "5.9m",
    note: "first run had Triton fp8 load cast error",
  },
  {
    model: "OpenCode GLM-5.1",
    route: "opencode",
    result: "EARLY",
    elapsed: "11.5m",
    note: "provider early-stop/no solution on opencode route",
  },
  {
    model: "Kimi K2.6",
    route: "kimi",
    result: "ERR",
    elapsed: "4s",
    note: "invalid or expired API key",
  },
  {
    model: "MiniMax/Qwen/MiMo via OpenRouter",
    route: "opencode",
    result: "ERR",
    elapsed: "1-2s",
    note: "provider_insufficient_credits",
  },
]

const FP8_RECOVERY_RUNS = [
  {
    model: "GLM-5.1",
    route: "zai-claude",
    result: "FAIL",
    elapsed: "11.0m",
    note: "direct ZAI route worked, nominal max_abs_diff=0.5625",
  },
  {
    model: "DeepSeek V4 Pro",
    route: "opencode",
    result: "FAIL",
    elapsed: "9.8m",
    note: "second attempt reached verifier, nominal max_abs_diff=0.539",
  },
  {
    model: "DeepSeek V4 Flash",
    route: "opencode",
    result: "FAIL",
    elapsed: "3.2m",
    note: "second attempt reached verifier, nominal max_abs_diff=0.539",
  },
  {
    model: "GPT-5.5",
    route: "codex",
    result: "FAIL",
    elapsed: "8.1m",
    note: "Triton resource failure: 147456B shared memory > 101376B limit",
  },
]

function isVisibleModel(m: Model) {
  return (
    VISIBLE_MODEL_LABELS.has(m.label) ||
    VISIBLE_MODEL_PREFIXES.some((prefix) => m.label.startsWith(prefix))
  )
}

export default async function HardPage() {
  const [lb, annotations, hasViewer, baselines] = await Promise.all([
    loadLeaderboard(),
    loadAnnotations(),
    loadAvailableViewers(),
    loadBaselines(),
  ])
  const models = [...lb.models].sort(compareModelRows)
  const visibleModels = models.filter(isVisibleModel)

  return (
    <div className="space-y-12">
      <section>
        <h1 className="text-3xl font-semibold tracking-tight text-[var(--color-fg-bright)] mb-3">
          Hard
        </h1>
        <p className="text-sm text-[var(--color-fg)] mb-6">
          Current scored CUDA board · RTX PRO 6000 Blackwell · sm_120 · 96 GB GDDR7 · 1.8 TB/s
        </p>
        <p className="text-[var(--color-fg)] leading-relaxed max-w-3xl">
          A focused successor to the v3 suite. One Blackwell GPU, a small set
          of hard CUDA kernel problems, and real coding-agent CLIs as the harness.
          The table below keeps one trusted row for same-harness reruns, while
          preserving distinct harness routes as separate rows. One-off diagnostic
          rows stay in the source data.
        </p>
        <p className="mt-4 text-sm text-[var(--color-fg)] max-w-3xl leading-relaxed">
          Problem IDs are stable, not consecutive: 04 was retired after the
          Kahan-softmax rubric leak, and 08 is a deferred Metal problem. The
          scored CUDA columns are 01, 02, 03, 05, 06, 07, 09, and 10.
        </p>
      </section>

      <section>
        <h2 className="text-xl font-semibold text-[var(--color-fg-bright)] mb-3">
          Leaderboard
        </h2>
        <p className="text-sm text-[var(--color-fg)] mb-4 max-w-4xl leading-relaxed">
          Cells show <code>peak_fraction</code>, the fraction of the relevant
          hardware ceiling reached by a correct kernel. Click a cell to open the
          transcript when one is available. Blue underlined values are the
          visible winner for that problem. Annotation badges mark caveats:
          {" "}<span className="annotation-badge annotation-badge-bad">!</span>
          {" "}invalid or reward-hack results, and{" "}
          <span className="annotation-badge annotation-badge-warn">!</span>
          {" "}scores with a rubric leak, bug, or unusual interpretation.
          The harness column is part of the identity: GLM-5.1 through OpenCode,
          Droid, and Claude Code-compatible Z.ai are separate measurements, but
          repeated runs through the same harness are collapsed to the most
          trustworthy row we currently have.
        </p>
        <LeaderboardTable
          models={visibleModels}
          annotations={annotations}
          hasViewer={hasViewer}
        />
        <p className="text-xs text-[var(--color-fg)] mt-3 max-w-4xl leading-relaxed">
          Full historical and diagnostic rows are still available in{" "}
          <Link
            href="https://github.com/Infatoshi/KernelBench-Hard/blob/master/results/leaderboard.json"
            className="underline underline-offset-2"
          >
            leaderboard.json
          </Link>
          .
        </p>
      </section>

      <section>
        <h2 className="text-xl font-semibold text-[var(--color-fg-bright)] mb-3">
          Per-problem ceilings
        </h2>
        <p className="text-sm text-[var(--color-fg)] mb-3 max-w-4xl leading-relaxed">
          eager / compiled = PyTorch reference timings. SOTA = the existing best-known kernel
          for the problem (vLLM paged attention, fbgemm grouped GEMM, etc.) when one exists
          on this hardware. best peak = the model that pushed furthest above the
          reference line.
        </p>
        <div className="box overflow-x-auto">
          <table className="term tabular text-sm">
            <thead>
              <tr>
                <th>problem</th>
                <th className="text-right">eager ms</th>
                <th className="text-right">compiled ms</th>
                <th className="text-right">SOTA ms</th>
                <th className="text-right">best peak</th>
                <th>best model</th>
                <th className="text-right">n scored</th>
              </tr>
            </thead>
            <tbody>
              {PROBLEMS.map((p) => {
                const pp = lb.per_problem[p.key]
                const bl = baselines?.problems[p.key] ?? {}
                const fmtMs = (t: { ms: number } | undefined) =>
                  t ? (
                    <span className="table-value">{t.ms.toFixed(3)}</span>
                  ) : (
                    <span className="cell-missing">-</span>
                  )
                return (
                  <tr key={p.key}>
                    <td>{p.key}</td>
                    <td className="text-right">
                      {fmtMs(bl.eager)}
                    </td>
                    <td className="text-right">
                      {fmtMs(bl.compiled)}
                    </td>
                    <td className="text-right">
                      {fmtMs(bl.sota)}
                    </td>
                    <td className="text-right text-[var(--color-fg-bright)]">
                      {pp.best_peak_fraction ? (
                        pp.best_peak_fraction.toFixed(3)
                      ) : (
                        <span className="cell-missing">-</span>
                      )}
                    </td>
                    <td>
                      {pp.best_model ? (
                        <span className="table-value">{shortLabel(pp.best_model)}</span>
                      ) : (
                        <span className="cell-missing">-</span>
                      )}
                    </td>
                    <td className="text-right">
                      {pp.n_passed}/{pp.n_attempted}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
        {!baselines && (
          <p className="text-xs text-[var(--color-fg)] mt-2">
            (baseline timings not yet generated — run scripts/run_baselines.sh in benchmarks/hard/)
          </p>
        )}
      </section>

      <section>
        <h2 className="text-xl font-semibold text-[var(--color-fg-bright)] mb-3">
          Rubric caveat
        </h2>
        <p className="text-[var(--color-fg)] leading-relaxed mb-4 max-w-3xl">
          One row in the leaderboard promises something the benchmark doesn&apos;t actually measure. It&apos;s marked{" "}
          <span className="annotation-badge annotation-badge-warn">!</span> for a reason.
        </p>

        <div className="space-y-6">
          <LeakCard
            title="01 fp8_gemm — bf16 dressup"
            cluster={[
              { model: "claude-opus-4-7 [max]", peak: "0.534" },
              { model: "mimo-v2.5-pro", peak: "0.434" },
              { model: "qwen3.6-plus", peak: "0.431" },
              { model: "qwen3.6-max-preview", peak: "0.429" },
              { model: "gpt-5.5 [xhigh]", peak: "0.423" },
            ]}
            body={
              <>
                Every passing solution at peak ≥ 0.4 casts the fp8 inputs to bf16
                inside the kernel and runs a bf16 GEMM. Both Opus 4.7 max and GPT-5.5
                xhigh explicitly pin to{" "}
                <code className="text-[var(--color-accent)]">cutlass::arch::Sm80</code>
                {" "}— Ampere CUTLASS, not the SM120 Blackwell FP8 tensor cores the
                problem name implies. Opus&apos;s source comment is unusually direct:
                <em className="text-[var(--color-fg-bright)] not-italic">
                  {" "}
                  &ldquo;follow the codex baseline (BF16 GEMM internally)...&rdquo;
                </em>{" "}
                The peak fractions on this row reflect bf16 GEMM optimization quality
                on fp8-typed inputs, not FP8 tensor core skill.
              </>
            }
          />

        </div>

        <p className="text-sm text-[var(--color-fg)] mt-6 max-w-3xl leading-relaxed">
          This leak is fixable with a few hours of problem-design work: tighten
          tolerance until bf16-via-cast and real fp8-tensor-core math diverge, or
          add a static-analysis check for the cast pattern. Keeping the caveat visible
          is still useful because &ldquo;five frontier models all took the same bf16
          shortcut on FP8 GEMM&rdquo; is itself a finding.
        </p>
      </section>

      <section>
        <h2 className="text-xl font-semibold text-[var(--color-fg-bright)] mb-3">
          FP8 constraint rerun
        </h2>
        <p className="text-sm text-[var(--color-fg)] mb-4 max-w-4xl leading-relaxed">
          On June 5, 2026, the FP8 GEMM verifier was tightened to reject the
          bf16-dressup shortcut and require an FP8-looking execution path. The
          earlier 01 scores above remain useful historical data, but this rerun
          is the cleaner answer to the caveat: once the shortcut is blocked,
          every available model either fails correctness, fails the provider
          path, or cannot run because of credits/key issues.
        </p>

        <div className="grid gap-4 lg:grid-cols-2">
          <div className="box overflow-x-auto">
            <h3 className="text-sm font-semibold text-[var(--color-fg-bright)] mb-3">
              Fixed-tolerance rerun
            </h3>
            <Fp8ConstraintTable rows={FP8_CONSTRAINT_FIXED_RUNS} />
          </div>
          <div className="box overflow-x-auto">
            <h3 className="text-sm font-semibold text-[var(--color-fg-bright)] mb-3">
              Recovery smokes
            </h3>
            <Fp8ConstraintTable rows={FP8_RECOVERY_RUNS} />
          </div>
        </div>

        <p className="text-xs text-[var(--color-fg-muted)] mt-4 max-w-4xl leading-relaxed">
          DeepSeek Pro&apos;s 354s first-run failure was not Triton spending 354s
          compiling. The agent used 354s generating code, then the verifier
          failed in 3s because Triton rejected an integer zero fallback on an
          fp8 load. The direct ZAI GLM-5.1 route was usable and produced a
          solution; it failed numeric tolerance rather than credits or auth.
        </p>

        <div className="grid gap-4 mt-5 lg:grid-cols-3">
          <Figure
            src="/blog-hard/fp8-constraint-rerun/fp8_token_burn_stacked.png"
            alt="Stacked token burn for FP8 constraint rerun"
            caption="Token burn by model on the FP8 constraint run."
          />
          <Figure
            src="/blog-hard/fp8-constraint-rerun/fp8_tokens_vs_effective_peak.png"
            alt="Tokens versus effective peak for FP8 constraint rerun"
            caption="All effective peaks collapse to zero under the strict verifier."
          />
          <Figure
            src="/blog-hard/fp8-constraint-rerun/fp8_cost_before_outcome.png"
            alt="Cost before outcome for FP8 constraint rerun"
            caption="Spend and wall time before each failing outcome."
          />
        </div>

        <p className="text-xs text-[var(--color-fg)] mt-4 max-w-4xl leading-relaxed">
          Raw summaries are committed as{" "}
          <Link href="/blog-hard/fp8-constraint-rerun/fixed-tolerance-summary.json">
            fixed-tolerance-summary.json
          </Link>
          {" "}and{" "}
          <Link href="/blog-hard/fp8-constraint-rerun/recovery-smokes-summary.json">
            recovery-smokes-summary.json
          </Link>
          .
        </p>
      </section>

      <section>
        <h2 className="text-xl font-semibold text-[var(--color-fg-bright)] mb-3">
          What changed from v3
        </h2>
        <ul className="space-y-2 text-sm leading-relaxed list-none pl-0 max-w-3xl">
          <Bullet>One GPU instead of three. RTX PRO 6000 Blackwell (sm_120, 96 GB GDDR7, 1.8 TB/s).</Bullet>
          <Bullet>A small hand-designed problem deck instead of 43-58. Per-trial L2 flush, 30-trial median, 10 warmup absorbing torch.compile CUDA-graph capture and Triton autotune.</Bullet>
          <Bullet>Real coding-agent CLIs as the harness: Claude Code, codex CLI, Kimi CLI, opencode, Droid. This is not a custom v3 agent loop.</Bullet>
          <Bullet>Wall-clock budgets, not turn limits. 45 min/run.</Bullet>
          <Bullet>peak_fraction grounded in physical hardware ceilings instead of raw speedup ratios.</Bullet>
          <Bullet>Per-cell annotations with verdict, pull quotes from solution.py, and an &ldquo;implication&rdquo; statement, including the May 13 Claude Code GLM-5.1 reward-hack example.</Bullet>
        </ul>
      </section>

      <section className="text-sm text-[var(--color-fg)] border-t border-[var(--color-border)] pt-6">
        Source data:{" "}
        <Link href="https://github.com/Infatoshi/KernelBench-Hard">
          github.com/Infatoshi/KernelBench-Hard
        </Link>
        {" · "}
        <Link href="https://github.com/Infatoshi/KernelBench-Hard/blob/master/results/leaderboard.json">
          leaderboard.json
        </Link>
        {" · "}
        <Link href="https://github.com/Infatoshi/KernelBench-Hard/tree/master/results/annotations">
          annotations/
        </Link>
        {" · "}
        <Link href="https://github.com/Infatoshi/KernelBench-Hard/blob/master/DEVLOG.md">
          DEVLOG.md
        </Link>
      </section>
    </div>
  )
}

function Fp8ConstraintTable({
  rows,
}: {
  rows: {
    model: string
    route: string
    result: string
    elapsed: string
    note: string
  }[]
}) {
  return (
    <table className="term tabular text-xs">
      <thead>
        <tr>
          <th>model</th>
          <th>route</th>
          <th>outcome</th>
          <th className="text-right">elapsed</th>
          <th>note</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={`${r.model}-${r.route}`}>
            <td className="text-[var(--color-fg-bright)] whitespace-nowrap">
              {r.model}
            </td>
            <td className="text-[var(--color-fg-muted)] whitespace-nowrap">
              {r.route}
            </td>
            <td>
              <span
                className={
                  r.result === "FAIL" ? "cell-fail" : "cell-err"
                }
              >
                {r.result}
              </span>
            </td>
            <td className="text-right text-[var(--color-fg-muted)] whitespace-nowrap">
              {r.elapsed}
            </td>
            <td className="min-w-64">{r.note}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function Figure({
  src,
  alt,
  caption,
}: {
  src: string
  alt: string
  caption: string
}) {
  return (
    <figure className="box">
      <img
        src={src}
        alt={alt}
        className="w-full border border-[var(--color-border)]"
      />
      <figcaption className="text-xs text-[var(--color-fg-muted)] mt-2 leading-relaxed">
        {caption}
      </figcaption>
    </figure>
  )
}

function LeaderboardTable({
  models,
  annotations,
  hasViewer,
}: {
  models: Model[]
  annotations: Map<string, { verdict: string; summary?: string }>
  hasViewer: Set<string>
}) {
  const winners = findVisibleWinners(models)

  return (
    <div className="overflow-x-auto box">
      <table className="term tabular text-xs sm:text-sm">
        <thead>
          <tr>
            <th className="sticky left-0 bg-[var(--color-surface-muted)]">model</th>
            <th>harness</th>
            {PROBLEMS.map((p) => (
              <th key={p.key} className="text-right">
                {p.short}
              </th>
            ))}
            <th className="text-right">SCORED</th>
          </tr>
        </thead>
        <tbody>
          {models.map((m) => (
            <tr key={m.label}>
              <td className="sticky left-0 bg-[var(--color-surface)] text-[var(--color-fg-bright)] whitespace-nowrap">
                {shortLabel(m.label)}
              </td>
              <td className="text-[var(--color-fg-muted)] whitespace-nowrap">
                {harnessLabel(m.harness)}
              </td>
              {PROBLEMS.map((p) => {
                const cell = m.results[p.key]
                return (
                  <td key={p.key} className="text-right">
                    {renderCell(
                      cell,
                      annotations,
                      hasViewer,
                      winners.get(p.key) === cell?.run_id,
                    )}
                  </td>
                )
              })}
              <td className="text-right text-[var(--color-fg-bright)]">
                {m.pass_count}/{m.total_runs}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function findVisibleWinners(models: Model[]) {
  const winners = new Map<string, string>()
  for (const p of PROBLEMS) {
    let bestRunId: string | null = null
    let bestPeak = -Infinity
    for (const m of models) {
      const cell = m.results[p.key]
      if (!cell?.correct || cell.peak_fraction == null || cell.invalid_reason) {
        continue
      }
      if (cell.peak_fraction > bestPeak) {
        bestPeak = cell.peak_fraction
        bestRunId = cell.run_id
      }
    }
    if (bestRunId) winners.set(p.key, bestRunId)
  }
  return winners
}

function compareModelRows(
  a: { label: string; pass_count: number; total_runs: number; results: Model["results"] },
  b: { label: string; pass_count: number; total_runs: number; results: Model["results"] },
) {
  if (b.pass_count !== a.pass_count) return b.pass_count - a.pass_count
  const bRate = b.total_runs ? b.pass_count / b.total_runs : 0
  const aRate = a.total_runs ? a.pass_count / a.total_runs : 0
  if (bRate !== aRate) return bRate - aRate
  const peakDiff = bestPeak(b.results) - bestPeak(a.results)
  if (peakDiff !== 0) return peakDiff
  return shortLabel(a.label).localeCompare(shortLabel(b.label))
}

function bestPeak(results: Model["results"]) {
  return Math.max(
    -1,
    ...Object.values(results).map((cell) => cell.peak_fraction ?? -1),
  )
}

function shortLabel(label: string) {
  return label
    .replace("codex/gpt-5.5 [2026-05-28 finish xhigh]", "GPT-5.5")
    .replace("codex/gpt-5.5 [xhigh]", "GPT-5.5")
    .replace("claude/claude-opus-4-7 [2026-05-28 finish max]", "Claude Opus 4.7")
    .replace("claude/claude-opus-4-8 [2026-05-28 opus48-grok max]", "Claude Opus 4.8")
    .replace("claude/claude-opus-4-6 [2026-06-04 opus46 max]", "Claude Opus 4.6")
    .replace("claude/claude-opus-4-7 [max]", "Claude Opus 4.7 [max]")
    .replace("cursor/composer-2.5-fast [2026-05-28 finish]", "Composer 2.5 Fast")
    .replace("gemini/gemini-3.5-flash [2026-05-28 finish]", "Gemini 3.5 Flash")
    .replace("grok/grok-build [2026-05-28 opus48-grok max]", "Grok Build")
    .replace("kimi/kimi-k2.6", "Kimi K2.6")
    .replace("opencode/openrouter-pinned/xiaomi/mimo-v2.5-pro", "MiMo v2.5 Pro")
    .replace("opencode/deepseek/deepseek-v4-flash", "DeepSeek V4 Flash")
    .replace("opencode/deepseek/deepseek-v4-pro", "DeepSeek V4 Pro")
    .replace("openrouter-google-ai-studio/google/", "or-google/")
    .replace("openrouter-alibaba/qwen/", "or-qwen/")
    .replace("zai-claude/glm-5.1 [2026-05-13]", "GLM-5.1")
    .replace("droid/zai/glm-5.1 [2026-05-08]", "GLM-5.1")
    .replace("opencode/zai/glm-5.1 [2026-05-08]", "GLM-5.1")
    .replace("opencode/zai/glm-5.1", "GLM-5.1")
    .replace("minimax-claude/MiniMax-M3", "MiniMax M3")
    .replace("opencode/openrouter-pinned/", "or/")
    .replace("opencode/", "")
    .replace("codex/", "")
    .replace("claude/", "")
    .replace("kimi/", "")
}

function harnessLabel(harness: string) {
  return harness
    .replace("zai-claude", "claude-code/zai")
    .replace("minimax-claude", "claude-code/minimax")
}

function renderCell(
  cell:
    | {
        run_id: string
        correct: boolean
        has_solution: boolean
        failure_reason?: string | null
        retryable_infra_failure?: boolean | null
        minimum_useful_output_tokens?: number | null
        peak_fraction: number | null
        elapsed_seconds?: number | null
        total_elapsed_seconds?: number | null
        check_elapsed_seconds?: number | null
        benchmark_elapsed_seconds?: number | null
        check_exit_code?: number | null
        benchmark_exit_code?: number | null
        output_tokens_per_second?: number | null
        usage?: {
          input_tokens?: number | null
          output_tokens?: number | null
          cache_read_tokens?: number | null
          cache_creation_tokens?: number | null
          reasoning_tokens?: number | null
          total_cost_usd?: number | null
        }
        session_complete?: boolean
        harness_exit_code?: number | null
        agent_cuda_disabled?: boolean
        gpu_queue_mode?: string | null
        gpu_lock_calls?: number | null
        gpu_lock_wait_seconds_total?: number | null
        gpu_lock_active_seconds_total?: number | null
        invalid_reason?: string
      }
    | undefined,
  annotations: Map<string, { verdict: string; summary?: string }>,
  hasViewer: Set<string>,
  isWinner: boolean,
) {
  if (!cell) return <span className="cell-missing">-</span>
  const viewerUrl = hasViewer.has(cell.run_id)
    ? `/runs/${cell.run_id}.html`
    : null
  const annot = annotations.get(cell.run_id)
  const title = cellTitle(cell, Boolean(viewerUrl), annot, isWinner)
  const wrap = (inner: React.ReactNode) =>
    viewerUrl ? (
      <a
        href={viewerUrl}
        className="no-underline hover:text-[var(--color-accent)]"
        title={title}
      >
        {inner}
      </a>
    ) : (
      <span title={title}>{inner}</span>
    )
  if (cell.invalid_reason || annot?.verdict === "reward_hack") {
    return wrap(
      <>
        <span className="cell-invalid">INVALID</span>
        <AnnotationBadge severity="bad" label="invalid or reward hack" />
      </>,
    )
  }
  if (cell.correct) {
    const badge =
      annot && ["rubric_leak", "bug", "interesting"].includes(annot.verdict) ? (
        <AnnotationBadge
          severity={annot.verdict === "bug" ? "bad" : "warn"}
          label={`annotated ${annot.verdict}`}
        />
      ) : null
    const pf = cell.peak_fraction
    const value =
      pf !== null ? (
        <span className={isWinner ? "cell-score cell-winner" : "cell-score"}>
          {pf.toFixed(3)}
        </span>
      ) : cell.failure_reason === "benchmark_timeout" ? (
        <span className="cell-err">BENCH</span>
      ) : (
        <span className="cell-err">NO PERF</span>
      )
    return wrap(
      <>
        {value}
        {badge}
      </>,
    )
  }
  if (cell.failure_reason === "provider_rate_limited") {
    return wrap(<span className="cell-err">RATE</span>)
  }
  if (cell.failure_reason === "provider_early_stop") {
    return wrap(<span className="cell-err">EARLY</span>)
  }
  if (cell.failure_reason === "timeout") {
    return wrap(<span className="cell-err">TIME</span>)
  }
  if (cell.failure_reason === "no_solution") {
    return wrap(<span className="cell-err">NO SOL</span>)
  }
  if (cell.has_solution) return wrap(<span className="cell-fail">FAIL</span>)
  return wrap(<span className="cell-err">ERR</span>)
}

function AnnotationBadge({
  severity,
  label,
}: {
  severity: "bad" | "warn"
  label: string
}) {
  return (
    <span
      className={`annotation-badge annotation-badge-${severity}`}
      title={label}
      aria-label={label}
    >
      !
    </span>
  )
}

function cellTitle(
  cell: {
    run_id: string
    correct: boolean
    peak_fraction: number | null
    failure_reason?: string | null
    retryable_infra_failure?: boolean | null
    minimum_useful_output_tokens?: number | null
    elapsed_seconds?: number | null
    total_elapsed_seconds?: number | null
    check_elapsed_seconds?: number | null
    benchmark_elapsed_seconds?: number | null
    check_exit_code?: number | null
    benchmark_exit_code?: number | null
    output_tokens_per_second?: number | null
    usage?: {
      input_tokens?: number | null
      output_tokens?: number | null
      cache_read_tokens?: number | null
      cache_creation_tokens?: number | null
      reasoning_tokens?: number | null
      total_cost_usd?: number | null
    }
    session_complete?: boolean
    harness_exit_code?: number | null
    agent_cuda_disabled?: boolean
    gpu_queue_mode?: string | null
    gpu_lock_calls?: number | null
    gpu_lock_wait_seconds_total?: number | null
    gpu_lock_active_seconds_total?: number | null
  },
  hasViewer: boolean,
  annotation?: { verdict: string; summary?: string },
  isWinner?: boolean,
) {
  const usage = cell.usage ?? {}
  const status =
    cell.failure_reason === "pass"
      ? null
      : cell.correct && cell.failure_reason === "benchmark_timeout"
        ? "benchmark timed out after correctness passed"
        : cell.correct && cell.failure_reason
          ? `status ${cell.failure_reason}`
          : cell.failure_reason
            ? `failure ${cell.failure_reason}`
            : null
  const parts = [
    cell.run_id,
    hasViewer ? "click to open transcript viewer" : "transcript viewer unavailable",
    isWinner ? "visible winner for this problem" : null,
    status,
    cell.retryable_infra_failure ? "retryable infra failure" : null,
    cell.minimum_useful_output_tokens != null
      ? `min useful output ${fmtInt(cell.minimum_useful_output_tokens)} tok`
      : null,
    cell.session_complete === false ? "session incomplete" : null,
    cell.harness_exit_code != null ? `exit ${cell.harness_exit_code}` : null,
    cell.elapsed_seconds != null ? `agent ${fmtDuration(cell.elapsed_seconds)}` : null,
    cell.total_elapsed_seconds != null
      ? `total ${fmtDuration(cell.total_elapsed_seconds)}`
      : null,
    cell.check_elapsed_seconds != null
      ? `check ${fmtDuration(cell.check_elapsed_seconds)}`
      : null,
    cell.check_exit_code != null ? `check exit ${cell.check_exit_code}` : null,
    cell.benchmark_elapsed_seconds != null
      ? `bench ${fmtDuration(cell.benchmark_elapsed_seconds)}`
      : null,
    cell.benchmark_exit_code != null
      ? `bench exit ${cell.benchmark_exit_code}`
      : null,
    usage.input_tokens != null ? `in ${fmtInt(usage.input_tokens)} tok` : null,
    usage.output_tokens != null ? `out ${fmtInt(usage.output_tokens)} tok` : null,
    usage.cache_read_tokens != null
      ? `cache read ${fmtInt(usage.cache_read_tokens)} tok`
      : null,
    usage.cache_creation_tokens != null
      ? `cache write ${fmtInt(usage.cache_creation_tokens)} tok`
      : null,
    usage.reasoning_tokens != null
      ? `reasoning ${fmtInt(usage.reasoning_tokens)} tok`
      : null,
    cell.output_tokens_per_second != null
      ? `${cell.output_tokens_per_second.toFixed(1)} out tok/s`
      : null,
    usage.total_cost_usd != null ? `$${usage.total_cost_usd.toFixed(4)}` : null,
    cell.gpu_queue_mode ? `queue ${cell.gpu_queue_mode}` : null,
    cell.gpu_lock_calls != null ? `GPU lock calls ${cell.gpu_lock_calls}` : null,
    cell.gpu_lock_wait_seconds_total != null
      ? `GPU lock wait ${fmtDuration(cell.gpu_lock_wait_seconds_total)}`
      : null,
    cell.gpu_lock_active_seconds_total != null
      ? `GPU lock active ${fmtDuration(cell.gpu_lock_active_seconds_total)}`
      : null,
    cell.agent_cuda_disabled ? "agent CUDA disabled" : null,
    annotation ? `annotation ${annotation.verdict}` : null,
    annotation?.summary ? `annotation summary: ${annotation.summary}` : null,
  ]
  return parts.filter(Boolean).join("\n")
}

function fmtDuration(seconds: number) {
  if (seconds < 60) return `${seconds}s`
  const min = Math.floor(seconds / 60)
  const sec = seconds % 60
  return `${min}m ${sec}s`
}

function fmtInt(n: number) {
  return new Intl.NumberFormat("en-US").format(n)
}

function LeakCard({
  title,
  cluster,
  body,
}: {
  title: string
  cluster: { model: string; peak: string }[]
  body: React.ReactNode
}) {
  return (
    <div className="box p-5">
      <h3 className="text-[var(--color-warn)] font-semibold mb-3">
        Caveat: {title}
      </h3>
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-x-4 gap-y-1 text-xs tabular mb-4">
        {cluster.map((c) => (
          <div key={c.model} className="flex justify-between">
            <span className="text-[var(--color-fg)]">{c.model}</span>
            <span className="text-[var(--color-fg-bright)]">{c.peak}</span>
          </div>
        ))}
      </div>
      <p className="text-sm text-[var(--color-fg)] leading-relaxed">{body}</p>
    </div>
  )
}

function Bullet({ children }: { children: React.ReactNode }) {
  return (
    <li className="flex gap-3">
      <span className="text-[var(--color-fg)] shrink-0">•</span>
      <span>{children}</span>
    </li>
  )
}
