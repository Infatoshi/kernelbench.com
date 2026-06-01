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
  "codex/gpt-5.5 [xhigh]",
  "claude/claude-opus-4-7 [max]",
  "kimi/kimi-k2.6",
  "opencode/openrouter-pinned/xiaomi/mimo-v2.5-pro",
  "opencode/deepseek/deepseek-v4-flash",
  "opencode/deepseek/deepseek-v4-pro",
])

const VISIBLE_MODEL_PREFIXES = [
  "codex/gpt-5.5 [2026-05-28 finish",
  "claude/claude-opus-4-7 [2026-05-28 finish",
  "claude/claude-opus-4-8 [2026-05-28 opus48-grok",
  "cursor/composer-2.5-fast [2026-05-28 finish",
  "gemini/gemini-3.5-flash [2026-05-28 finish",
  "grok/grok-build [2026-05-28 opus48-grok",
  "minimax-claude/MiniMax-M3 [2026-06-01",
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
          KernelBench Hard
        </h1>
        <p className="text-sm text-[var(--color-fg-muted)] mb-6">
          Current scored CUDA board · RTX PRO 6000 Blackwell · sm_120 · 96 GB GDDR7 · 1.8 TB/s
        </p>
        <p className="text-[var(--color-fg)] leading-relaxed max-w-3xl">
          A focused successor to KernelBench v3. One Blackwell GPU, a small set
          of hard CUDA kernel problems, and real coding-agent CLIs as the harness.
          The table below keeps the comparable model rows visible and leaves
          one-off diagnostic rows in the source data.
        </p>
        <p className="mt-4 text-sm text-[var(--color-fg-muted)] max-w-3xl leading-relaxed">
          Problem IDs are stable, not consecutive: 04 was retired after the
          Kahan-softmax rubric leak, and 08 is a deferred Metal problem. The
          scored CUDA columns are 01, 02, 03, 05, 06, 07, 09, and 10.
        </p>
      </section>

      <section>
        <h2 className="text-xl font-semibold text-[var(--color-fg-bright)] mb-3">
          Leaderboard
        </h2>
        <p className="text-sm text-[var(--color-fg-muted)] mb-4 max-w-4xl leading-relaxed">
          Cells show <code>peak_fraction</code>, the fraction of the relevant
          hardware ceiling reached by a correct kernel. Click a cell to open the
          transcript when one is available. <span className="text-[var(--color-warn)]">★</span>{" "}
          marks an annotation.
        </p>
        <LeaderboardTable
          models={visibleModels}
          annotations={annotations}
          hasViewer={hasViewer}
        />
        <p className="text-xs text-[var(--color-fg-muted)] mt-3 max-w-4xl leading-relaxed">
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
        <p className="text-sm text-[var(--color-fg-muted)] mb-3 max-w-4xl leading-relaxed">
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
                  t ? t.ms.toFixed(3) : "—"
                return (
                  <tr key={p.key}>
                    <td>{p.key}</td>
                    <td className="text-right text-[var(--color-fg-muted)]">
                      {fmtMs(bl.eager)}
                    </td>
                    <td className="text-right text-[var(--color-fg-muted)]">
                      {fmtMs(bl.compiled)}
                    </td>
                    <td className="text-right text-[var(--color-fg-muted)]">
                      {fmtMs(bl.sota)}
                    </td>
                    <td className="text-right text-[var(--color-fg-bright)]">
                      {pp.best_peak_fraction
                        ? pp.best_peak_fraction.toFixed(3)
                        : "-"}
                    </td>
                    <td className="text-[var(--color-fg-muted)]">
                      {pp.best_model ? shortLabel(pp.best_model) : "-"}
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
          <p className="text-xs text-[var(--color-fg-dim)] mt-2">
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
          <span className="text-[var(--color-warn)]">★</span> for a reason.
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

        <p className="text-sm text-[var(--color-fg-muted)] mt-6 max-w-3xl leading-relaxed">
          This leak is fixable with a few hours of problem-design work: tighten
          tolerance until bf16-via-cast and real fp8-tensor-core math diverge, or
          add a static-analysis check for the cast pattern. Keeping the caveat visible
          is still useful because &ldquo;five frontier models all took the same bf16
          shortcut on FP8 GEMM&rdquo; is itself a finding.
        </p>
      </section>

      <section>
        <h2 className="text-xl font-semibold text-[var(--color-fg-bright)] mb-3">
          What changed from v3
        </h2>
        <ul className="space-y-2 text-sm leading-relaxed list-none pl-0 max-w-3xl">
          <Bullet>One GPU instead of three. RTX PRO 6000 Blackwell (sm_120, 96 GB GDDR7, 1.8 TB/s).</Bullet>
          <Bullet>A small hand-designed problem deck instead of 43-58. Per-trial L2 flush, 30-trial median, 10 warmup absorbing torch.compile CUDA-graph capture and Triton autotune.</Bullet>
          <Bullet>Real coding-agent CLIs as the harness — Claude Code, codex CLI, Kimi CLI, opencode, Droid — not a custom KernelBench agent loop.</Bullet>
          <Bullet>Wall-clock budgets, not turn limits. 45 min/run.</Bullet>
          <Bullet>peak_fraction grounded in physical hardware ceilings instead of raw speedup ratios.</Bullet>
          <Bullet>Per-cell annotations with verdict, pull quotes from solution.py, and an &ldquo;implication&rdquo; statement, including the May 13 Claude Code GLM-5.1 reward-hack example.</Bullet>
        </ul>
      </section>

      <section className="text-sm text-[var(--color-fg-muted)] border-t border-[var(--color-border)] pt-6">
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

function LeaderboardTable({
  models,
  annotations,
  hasViewer,
}: {
  models: Model[]
  annotations: Map<string, { verdict: string }>
  hasViewer: Set<string>
}) {
  return (
    <div className="overflow-x-auto box">
      <table className="term tabular text-xs sm:text-sm">
        <thead>
          <tr>
            <th className="sticky left-0 bg-[var(--color-surface-muted)]">model</th>
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
              {PROBLEMS.map((p) => {
                const cell = m.results[p.key]
                return (
                  <td key={p.key} className="text-right">
                    {renderCell(cell, annotations, hasViewer)}
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
    .replace("codex/gpt-5.5 [2026-05-28 finish xhigh]", "GPT-5.5 [2026-05-28]")
    .replace("codex/gpt-5.5 [xhigh]", "GPT-5.5 [xhigh]")
    .replace("claude/claude-opus-4-7 [2026-05-28 finish max]", "Claude Opus 4.7 [2026-05-28]")
    .replace("claude/claude-opus-4-8 [2026-05-28 opus48-grok max]", "Claude Opus 4.8 [2026-05-28]")
    .replace("claude/claude-opus-4-7 [max]", "Claude Opus 4.7 [max]")
    .replace("cursor/composer-2.5-fast [2026-05-28 finish]", "Composer 2.5 Fast [2026-05-28]")
    .replace("gemini/gemini-3.5-flash [2026-05-28 finish]", "Gemini 3.5 Flash [2026-05-28]")
    .replace("grok/grok-build [2026-05-28 opus48-grok max]", "Grok Build [2026-05-28]")
    .replace("kimi/kimi-k2.6", "Kimi K2.6")
    .replace("opencode/openrouter-pinned/xiaomi/mimo-v2.5-pro", "MiMo v2.5 Pro")
    .replace("opencode/deepseek/deepseek-v4-flash", "DeepSeek V4 Flash")
    .replace("opencode/deepseek/deepseek-v4-pro", "DeepSeek V4 Pro")
    .replace("openrouter-google-ai-studio/google/", "or-google/")
    .replace("openrouter-alibaba/qwen/", "or-qwen/")
    .replace("zai-claude/glm-5.1 [2026-05-13]", "Claude Code GLM-5.1 [2026-05-13]")
    .replace("droid/zai/glm-5.1 [2026-05-08]", "Droid GLM-5.1 [2026-05-08]")
    .replace("opencode/zai/glm-5.1 [2026-05-08]", "OpenCode GLM-5.1 rerun [2026-05-08]")
    .replace("opencode/zai/glm-5.1", "OpenCode GLM-5.1 original")
    .replace("minimax-claude/MiniMax-M3", "MiniMax M3")
    .replace("opencode/openrouter-pinned/", "or/")
    .replace("opencode/", "")
    .replace("codex/", "")
    .replace("claude/", "")
    .replace("kimi/", "")
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
  annotations: Map<string, { verdict: string }>,
  hasViewer: Set<string>,
) {
  if (!cell) return <span className="cell-err">-</span>
  const viewerUrl = hasViewer.has(cell.run_id)
    ? `/runs/${cell.run_id}.html`
    : null
  const title = cellTitle(cell, Boolean(viewerUrl))
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
  const annot = annotations.get(cell.run_id)
  if (cell.invalid_reason || annot?.verdict === "reward_hack") {
    return wrap(
      <>
        <span className="cell-fail">INVALID</span>
        <span className="ml-1 text-[var(--color-warn)]">★</span>
      </>,
    )
  }
  if (cell.correct) {
    const star =
      annot && annot.verdict === "rubric_leak" ? (
        <span className="text-[var(--color-warn)]">★</span>
      ) : annot && annot.verdict === "clean" ? (
        <span className="text-[var(--color-fg-bright)]">★</span>
      ) : null
    const pf = cell.peak_fraction
    const value =
      pf !== null ? (
        pf.toFixed(3)
      ) : cell.failure_reason === "benchmark_timeout" ? (
        <span className="cell-err">BENCH</span>
      ) : (
        <span className="cell-err">NO PERF</span>
      )
    return wrap(
      <>
        {value}
        {star ? <span className="ml-1">{star}</span> : null}
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
            <span className="text-[var(--color-fg-muted)]">{c.model}</span>
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
      <span className="text-[var(--color-fg-muted)] shrink-0">•</span>
      <span>{children}</span>
    </li>
  )
}
