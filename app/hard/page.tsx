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
  { key: "04_kahan_softmax", short: "04 kahan" },
  { key: "05_topk_bitonic", short: "05 topk" },
  { key: "06_sonic_moe_swiglu", short: "06 moe" },
  { key: "07_w4a16_gemm", short: "07 w4a16" },
  { key: "09_fmha_preattn_mrope", short: "09 mrope" },
  { key: "10_patch_embed_conv3d_gemm", short: "10 patch" },
]

const PRIMARY_MODEL_LABELS = new Set([
  "codex/gpt-5.5 [xhigh]",
  "claude/claude-opus-4-7 [max]",
  "zai-claude/glm-5.1 [2026-05-13]",
  "droid/zai/glm-5.1 [2026-05-08]",
  "opencode/deepseek/deepseek-v4-flash",
  "opencode/deepseek/deepseek-v4-pro",
])

const FRESH_PRIMARY_PREFIXES = [
  "codex/gpt-5.5 [2026-05-28 finish",
  "claude/claude-opus-4-7 [2026-05-28 finish",
  "claude/claude-opus-4-8 [2026-05-28 opus48-grok",
  "cursor/composer-2.5-fast [2026-05-28 finish",
  "gemini/gemini-3.5-flash [2026-05-28 finish",
  "grok/grok-build [2026-05-28 opus48-grok",
]

const DIAGNOSTIC_AUDIT_NOTES: Record<string, string> = {
  "kimi/kimi-k2.6":
    "Problems 09/10 aborted in 4-5s with 401 auth errors; 01 is a reward-hack failure.",
  "opencode/openrouter-pinned/xiaomi/mimo-v2.5-pro":
    "Problem 03 ended at provider/reasoning length with no solution; 02 is a real solution failure.",
  "opencode/openrouter-pinned/qwen/qwen3.6-max-preview":
    "Problems 03/10 are no-solution provider or unknown early-stop cells.",
  "opencode/openrouter-pinned/qwen/qwen3.6-plus":
    "Problems 02/04 are harness/setup or unknown early-stop cells.",
  "opencode/zai/glm-5.1":
    "Problems 03/05 hit hidden reasoning-token limits before writing solution.py; 01 timed out after regressing a passing attempt.",
  "opencode/zai/glm-5.1 [2026-05-08]":
    "Problems 03/05/07/09 hit hidden reasoning-token limits with no solution.py; 01 timed out broken.",
  "opencode/zai/glm-5.1 [2026-05-28 finish]":
    "May 28 retry confirmed six CUDA cells still end as provider early-stops; only the inherited 04 pass remains.",
  "zai-claude/glm-5.1 [2026-05-28 finish]":
    "Z.ai Claude route was unavailable in May 28 preflight; cells are retained from the May 23 diagnostic run.",
  "opencode/openrouter-pinned/minimax/minimax-m2.7":
    "Problem 01 has no checkable artifact; remaining non-passes mix timeout and invalid/forbidden-op solution failures.",
  "opencode/openrouter-pinned/qwen/qwen3.6-27b":
    "Multiple no-solution/API/unknown cells; raw pass count is not comparable.",
}

function isPrimaryModel(m: Model) {
  return (
    PRIMARY_MODEL_LABELS.has(m.label) ||
    FRESH_PRIMARY_PREFIXES.some((prefix) => m.label.startsWith(prefix))
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
  const primaryModels = models.filter(isPrimaryModel)
  const diagnosticModels = models.filter((m) => !isPrimaryModel(m))

  return (
    <div className="space-y-12">
      <section>
        <h1 className="prompt cursor text-3xl font-bold text-[var(--color-fg-bright)] glow mb-4">
          kernelbench hard
        </h1>
        <p className="text-sm text-[var(--color-fg-muted)] mb-6">
          Original 9-problem board + May 28 CUDA finish rerun · RTX PRO 6000 Blackwell · sm_120 · 96 GB GDDR7 · 1.8 TB/s
        </p>
        <p className="text-[var(--color-fg)] leading-relaxed max-w-3xl">
          A focused successor to KernelBench v3. One Blackwell GPU, nine hand-designed problems, real coding-agent CLIs as the harness. The original public board swept twelve frontier model-harness pairs; the May 8 Z.ai rerun added fresh GLM-5.1 rows for OpenCode and Droid, and the May 13 rerun adds GLM-5.1 through Claude Code on Z.ai's Anthropic-compatible endpoint. The May 28 finish rerun adds queue-safe seven-problem CUDA rows for Codex, Claude, Cursor Agent, and native Gemini; the same-day addendum adds Claude Opus 4.8 and Grok Build. The leaderboard separates primary comparable sweeps from diagnostic rows where audit found API/auth/provider/adapter no-results.
        </p>
      </section>

      <section>
        <h2 className="text-xl font-bold text-[var(--color-fg-bright)] mb-4 glow">
          # leaderboard
        </h2>
        <p className="text-xs text-[var(--color-fg-muted)] mb-4">
          cells = peak_fraction (fraction of the relevant hardware ceiling). FAIL = solution written but missed correctness. ERR = no solution produced. INVALID = benchmark file mutation or other scoring-invalid behavior. <span className="text-[var(--color-warn)]">★</span> = annotation attached. <span className="text-[var(--color-fg-bright)]">click any cell to open the full transcript viewer</span> — every tool call, every reasoning step, the solution.py, the check.log.
          {" "}Fresh sweep cells include wall time, token usage, cache/reasoning tokens, and GPU queue mode in the cell tooltip when present.
        </p>
        <div className="space-y-8">
          <div>
            <h3 className="text-sm font-bold text-[var(--color-fg-bright)] mb-2">
              serious comparison
            </h3>
            <p className="text-[10px] sm:text-xs text-[var(--color-fg-muted)] mb-2 max-w-4xl leading-relaxed">
              Audited non-passes here are model/check failures, full-budget timeouts,
              or explicit invalid behavior. Raw pass totals are comparable within this
              section.
            </p>
            <LeaderboardTable
              models={primaryModels}
              annotations={annotations}
              hasViewer={hasViewer}
            />
          </div>

          <div>
            <h3 className="text-sm font-bold text-[var(--color-fg-bright)] mb-2">
              diagnostic / needs rerun
            </h3>
            <p className="text-[10px] sm:text-xs text-[var(--color-fg-muted)] mb-2 max-w-4xl leading-relaxed">
              Rows retained for transparency, but at least one non-pass was an
              auth/API/provider/adapter/setup no-result rather than a clean model
              attempt. Do not rank these pass totals against the serious table.
            </p>
            <LeaderboardTable
              models={diagnosticModels}
              annotations={annotations}
              hasViewer={hasViewer}
              auditNotes={DIAGNOSTIC_AUDIT_NOTES}
            />
          </div>
        </div>
        <p className="text-[10px] sm:text-xs text-[var(--color-fg-muted)] mt-2 max-w-4xl leading-relaxed">
          Audit note: DeepSeek through OpenCode stayed in the serious section because
          its non-passes were normal correctness/build failures or full-budget
          timeouts. OpenCode/Z.ai rows remain diagnostic because repeated hidden
          reasoning/provider early-stops produced no-solution cells before useful actions.
          Inspect the{" "}
          <Link
            href="/runs?harness=opencode"
            className="underline underline-offset-2 decoration-[var(--color-bad)] hover:text-[var(--color-bad)]"
          >
            runs
          </Link>{" "}
          for the raw transcripts.
        </p>
      </section>

      <section>
        <h2 className="text-xl font-bold text-[var(--color-fg-bright)] mb-4 glow">
          # per-problem ceilings
        </h2>
        <p className="text-xs text-[var(--color-fg-muted)] mb-3">
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
                <th className="text-right">n pass</th>
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
        <h2 className="text-xl font-bold text-[var(--color-fg-bright)] mb-4 glow">
          # rubric leaks
        </h2>
        <p className="text-[var(--color-fg)] leading-relaxed mb-4 max-w-3xl">
          Two cells in the leaderboard promise something the benchmark doesn&apos;t actually measure. They&apos;re marked{" "}
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

          <LeakCard
            title="04 kahan_softmax — Kahan compensation skipped"
            cluster={[
              { model: "gpt-5.5 [xhigh]", peak: "0.363" },
              { model: "claude-opus-4-7 [max]", peak: "0.317" },
              { model: "deepseek-v4-flash", peak: "0.138" },
              { model: "glm-5.1", peak: "0.125" },
              { model: "mimo-v2.5-pro", peak: "0.121" },
              { model: "kimi-k2.6", peak: "0.118" },
            ]}
            body={
              <>
                Six of seven passing solutions skipped the Kahan compensated
                summation entirely. Only{" "}
                <span className="text-[var(--color-fg-bright)]">deepseek-v4-pro</span>
                {" "}— the lowest passing peak at 0.101 — actually implemented the
                algorithm the problem name describes. Compensated summation has real
                overhead, naive softmax fits within tolerance, and the rubric leaks.
                The model whose docstring explicitly states &ldquo;Numerically tight
                softmax with Kahan compensated summation&rdquo; is the model that
                loses the cell.
              </>
            }
          />
        </div>

        <p className="text-sm text-[var(--color-fg-muted)] mt-6 max-w-3xl leading-relaxed">
          Both leaks are fixable in a few hours of problem-design work. Publishing without
          fixing because (a) every iteration surfaces the next leak — diminishing returns,
          and (b) the leaks ARE the finding. &ldquo;Five frontier models all took the same
          bf16 shortcut on FP8 GEMM&rdquo; is itself a headline.
        </p>
      </section>

      <section>
        <h2 className="text-xl font-bold text-[var(--color-fg-bright)] mb-4 glow">
          # what changed from v3
        </h2>
        <ul className="space-y-2 text-sm leading-relaxed list-none pl-0 max-w-3xl">
          <Bullet>One GPU instead of three. RTX PRO 6000 Blackwell (sm_120, 96 GB GDDR7, 1.8 TB/s).</Bullet>
          <Bullet>Seven hand-designed problems instead of 43-58. Per-trial L2 flush, 30-trial median, 10 warmup absorbing torch.compile CUDA-graph capture and Triton autotune.</Bullet>
          <Bullet>Real coding-agent CLIs as the harness — Claude Code, codex CLI, Kimi CLI, opencode, Droid — not a custom KernelBench agent loop.</Bullet>
          <Bullet>Wall-clock budgets, not turn limits. 45 min/run.</Bullet>
          <Bullet>peak_fraction grounded in physical hardware ceilings instead of raw speedup ratios.</Bullet>
          <Bullet>Per-cell annotations with verdict, pull quotes from solution.py, and an &ldquo;implication&rdquo; statement. 36 annotations, including the May 13 Claude Code GLM-5.1 reward-hack example.</Bullet>
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
  auditNotes,
}: {
  models: Model[]
  annotations: Map<string, { verdict: string }>
  hasViewer: Set<string>
  auditNotes?: Record<string, string>
}) {
  const showNotes = Boolean(auditNotes)
  return (
    <div className="overflow-x-auto box">
      <table className="term tabular text-xs sm:text-sm">
        <thead>
          <tr>
            <th className="sticky left-0 bg-[var(--color-bg)]">model</th>
            {PROBLEMS.map((p) => (
              <th key={p.key} className="text-right">
                {p.short}
              </th>
            ))}
            <th className="text-right">PASS</th>
            {showNotes ? <th>audit note</th> : null}
          </tr>
        </thead>
        <tbody>
          {models.map((m) => (
            <tr key={m.label}>
              <td className="sticky left-0 bg-[var(--color-bg)] text-[var(--color-fg-bright)] whitespace-nowrap">
                {shortLabel(m.label)}
                {m.effort ? (
                  <span className="text-[var(--color-fg-muted)]">
                    {" "}
                    [{m.effort}]
                  </span>
                ) : null}
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
              {showNotes ? (
                <td className="min-w-72 max-w-md whitespace-normal text-[10px] sm:text-xs text-[var(--color-fg-muted)] leading-relaxed">
                  {auditNotes?.[m.label] ?? "Unclassified row; kept diagnostic until audited."}
                </td>
              ) : null}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function compareModelRows(
  a: { harness: string; pass_count: number },
  b: { harness: string; pass_count: number },
) {
  const priority: Record<string, number> = {
    codex: 0,
    claude: 1,
    "zai-claude": 2,
    droid: 3,
    kimi: 4,
    opencode: 5,
  }
  const pa = priority[a.harness] ?? 5
  const pb = priority[b.harness] ?? 5
  if (pa !== pb) return pa - pb
  return b.pass_count - a.pass_count
}

function shortLabel(label: string) {
  return label
    .replace("openrouter-google-ai-studio/google/", "or-google/")
    .replace("openrouter-alibaba/qwen/", "or-qwen/")
    .replace("zai-claude/glm-5.1 [2026-05-13]", "Claude Code GLM-5.1 [2026-05-13]")
    .replace("droid/zai/glm-5.1 [2026-05-08]", "Droid GLM-5.1 [2026-05-08]")
    .replace("opencode/zai/glm-5.1 [2026-05-08]", "OpenCode GLM-5.1 rerun [2026-05-08]")
    .replace("opencode/zai/glm-5.1", "OpenCode GLM-5.1 original")
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
    return wrap(
      <>
        {pf !== null ? pf.toFixed(3) : "PASS"}
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
  const parts = [
    cell.run_id,
    hasViewer ? "click to open transcript viewer" : "transcript viewer unavailable",
    cell.failure_reason ? `failure ${cell.failure_reason}` : null,
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
      <h3 className="text-[var(--color-warn)] font-bold mb-3">★ {title}</h3>
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
      <span className="text-[var(--color-accent)] shrink-0">&gt;</span>
      <span>{children}</span>
    </li>
  )
}
