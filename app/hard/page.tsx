import Link from "next/link"
import { readdir } from "node:fs/promises"
import { join } from "node:path"
import {
  loadLeaderboard,
  loadAnnotations,
  type Cell,
  type Model,
} from "@/lib/data"
import { LeaderboardTable, type HardRunRecord } from "./leaderboard-table"

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
  { key: "01_fp8_gemm", label: "FP8 GEMM" },
  { key: "02_kda_cutlass", label: "KDA CUTLASS" },
  { key: "03_paged_attention", label: "Paged Attention" },
  { key: "05_topk_bitonic", label: "TopK Bitonic" },
  { key: "06_sonic_moe_swiglu", label: "Sonic MoE SwiGLU" },
  { key: "07_w4a16_gemm", label: "W4A16 GEMM" },
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

function isVisibleModel(m: Model) {
  return (
    VISIBLE_MODEL_LABELS.has(m.label) ||
    VISIBLE_MODEL_PREFIXES.some((prefix) => m.label.startsWith(prefix))
  )
}

export default async function HardPage() {
  const [lb, annotations, hasViewer] = await Promise.all([
    loadLeaderboard(),
    loadAnnotations(),
    loadAvailableViewers(),
  ])
  const models = [...lb.models].sort(compareModelRows)
  const visibleModels = models.filter(isVisibleModel)

  return (
    <div className="hard-page space-y-12">
      <section>
        <h1 className="text-3xl font-semibold tracking-tight text-[var(--color-fg-bright)] mb-3">
          hard
        </h1>
        <p className="text-sm text-[var(--color-fg)] mb-6">
          Current scored CUDA board · RTX PRO 6000 Blackwell · sm_120 · 96 GB GDDR7 · 1.8 TB/s
        </p>
        <p className="mt-3 text-xs text-[var(--color-fg-muted)] max-w-3xl leading-relaxed">
          Browse the{" "}
          <Link href="/runs" className="underline underline-offset-2">
            run index
          </Link>
          {" "}for transcripts, submitted solutions, checks, timing, and costs.
        </p>
      </section>

      <section>
        <LeaderboardMetricsTable
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

    </div>
  )
}

function LeaderboardMetricsTable({
  models,
  annotations,
  hasViewer,
}: {
  models: Model[]
  annotations: Map<string, { verdict: string; summary?: string }>
  hasViewer: Set<string>
}) {
  const winners = findVisibleWinners(models)
  const rows = buildRunRows(models, annotations, hasViewer, winners)

  return <LeaderboardTable rows={rows} />
}

type RunStatusTone = "good" | "bad" | "warn" | "muted"

type RunStatus = HardRunRecord["compiled"]

type RunRow = HardRunRecord

function status(
  label: string,
  tone: RunStatusTone,
  annotationSeverity?: "bad" | "warn",
  annotationLabel?: string,
): RunStatus {
  return { label, tone, annotationSeverity, annotationLabel }
}

function buildRunRows(
  models: Model[],
  annotations: Map<string, { verdict: string; summary?: string }>,
  hasViewer: Set<string>,
  winners: Map<string, string>,
): RunRow[] {
  const rows: RunRow[] = []
  for (const m of models) {
    for (const p of PROBLEMS) {
      const cell = m.results[p.key]
      const model = shortLabel(m.label)
      const harness = harnessLabel(m.harness)
      if (!cell) {
        rows.push(missingRunRow(model, harness, p))
        continue
      }

      const annot = annotations.get(cell.run_id)
      const isWinner = winners.get(p.key) === cell.run_id
      const title = cellTitle(cell, hasViewer.has(cell.run_id), annot, isWinner)
      const usage = cell.usage ?? {}
      const runAt = runDateParts(cell.run_id)
      const hasRunViewer = hasViewer.has(cell.run_id)
      const compiled = compiledStatus(cell)
      const correct = correctnessStatus(cell, annot)
      const note = annot?.summary || cell.failure_reason || "run details"
      const rewardHack = annot?.verdict === "reward_hack"
      const explanation = annot?.summary || cell.invalid_reason || cell.failure_reason || null
      const cacheTokens =
        (usage.cache_read_tokens ?? 0) + (usage.cache_creation_tokens ?? 0)
      rows.push({
        key: cell.run_id,
        runId: cell.run_id,
        model,
        harness,
        problem: p.label,
        problemKey: p.key,
        date: runAt.date,
        time: runAt.time,
        compiled,
        correct,
        rewardHack,
        explanation,
        peakFraction: cell.peak_fraction,
        speedPct: cell.peak_fraction == null ? null : cell.peak_fraction * 100,
        isWinner,
        outputTokens: usage.output_tokens ?? null,
        reasoningTokens: usage.reasoning_tokens ?? null,
        cacheTokens,
        inputTokens: usage.input_tokens ?? null,
        costUsd: usage.total_cost_usd ?? null,
        outputTokensPerSecond: cell.output_tokens_per_second ?? null,
        elapsedSeconds: cell.elapsed_seconds ?? null,
        checkSeconds: cell.check_elapsed_seconds ?? null,
        benchmarkSeconds: cell.benchmark_elapsed_seconds ?? null,
        totalSeconds: cell.total_elapsed_seconds ?? null,
        gpuWaitSeconds: cell.gpu_lock_wait_seconds_total ?? null,
        gpuActiveSeconds: cell.gpu_lock_active_seconds_total ?? null,
        referenceUrl: referenceUrlFor(p.key),
        solutionUrl: hasRunViewer ? `/runs/${cell.run_id}.html#tab-solution` : null,
        transcriptUrl: hasRunViewer ? `/runs/${cell.run_id}.html` : null,
        scored: `${m.pass_count}/${m.total_runs}`,
        note,
        title,
        tokenTitle: tokenTitle(cell),
        runtimeTitle: runtimeTitle(cell),
        searchText: [
          model,
          harness,
          p.label,
          p.key,
          cell.run_id,
          compiled.label,
          correct.label,
          rewardHack ? "reward hacking" : "no reward hacking",
          explanation,
          note,
          annot?.verdict,
          annot?.summary,
          `scored ${m.pass_count}/${m.total_runs}`,
        ]
          .filter(Boolean)
          .join(" "),
      })
    }
  }
  return rows
}

function missingRunRow(
  model: string,
  harness: string,
  problem: { key: string; label: string },
): RunRow {
  return {
    key: `${model}:${harness}:${problem.key}:missing`,
    runId: null,
    model,
    harness,
    problem: problem.label,
    problemKey: problem.key,
    date: null,
    time: null,
    compiled: status("no run", "muted"),
    correct: status("no run", "muted"),
    rewardHack: false,
    explanation: null,
    peakFraction: null,
    speedPct: null,
    isWinner: false,
    outputTokens: null,
    reasoningTokens: null,
    cacheTokens: null,
    inputTokens: null,
    costUsd: null,
    outputTokensPerSecond: null,
    elapsedSeconds: null,
    checkSeconds: null,
    benchmarkSeconds: null,
    totalSeconds: null,
    gpuWaitSeconds: null,
    gpuActiveSeconds: null,
    referenceUrl: referenceUrlFor(problem.key),
    solutionUrl: null,
    transcriptUrl: null,
    scored: "0/0",
    note: "no run",
    title: "no run",
    tokenTitle: "",
    runtimeTitle: "",
    searchText: `${model} ${harness} ${problem.label} ${problem.key} no run`,
  }
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

function correctnessStatus(
  cell: Cell,
  annot?: { verdict: string; summary?: string },
): RunStatus {
  if (cell.invalid_reason || annot?.verdict === "reward_hack") {
    return status("invalid", "bad", "bad", "invalid or reward hack")
  }
  if (cell.correct) {
    if (annot && ["rubric_leak", "bug", "interesting"].includes(annot.verdict)) {
      return status(
        "pass",
        "good",
        annot.verdict === "bug" ? "bad" : "warn",
        `annotated ${annot.verdict}`,
      )
    }
    return status("pass", "good")
  }
  if (cell.failure_reason === "provider_rate_limited") return status("rate", "bad")
  if (cell.failure_reason === "provider_early_stop") return status("early", "warn")
  if (cell.failure_reason === "timeout") return status("time", "warn")
  if (cell.failure_reason === "no_solution") return status("no sol", "bad")
  if (cell.has_solution) return status("fail", "bad")
  return status("err", "bad")
}

function compiledStatus(cell: Cell): RunStatus {
  if (!cell.has_solution) return status("no sol", "muted")
  if (
    cell.correct ||
    cell.peak_fraction != null ||
    cell.check_exit_code === 0 ||
    cell.benchmark_exit_code === 0
  ) {
    return status("yes", "good")
  }
  if (cell.check_exit_code != null || cell.benchmark_exit_code != null) {
    return status("failed", "bad")
  }
  return status("unknown", "warn")
}

function referenceUrlFor(problem: string) {
  return `https://github.com/Infatoshi/KernelBench-Hard/blob/master/problems/${problem}/reference.py`
}

function renderCorrectness(
  cell: Cell,
  annot?: { verdict: string; summary?: string },
) {
  if (cell.invalid_reason || annot?.verdict === "reward_hack") {
    return (
      <>
        <StatusPill tone="bad">invalid</StatusPill>
        <AnnotationBadge severity="bad" label="invalid or reward hack" />
      </>
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
    return (
      <>
        <StatusPill tone="good">pass</StatusPill>
        {badge}
      </>
    )
  }
  if (cell.failure_reason === "provider_rate_limited") return <StatusPill tone="bad">rate</StatusPill>
  if (cell.failure_reason === "provider_early_stop") return <StatusPill tone="warn">early</StatusPill>
  if (cell.failure_reason === "timeout") return <StatusPill tone="warn">time</StatusPill>
  if (cell.failure_reason === "no_solution") return <StatusPill tone="bad">no sol</StatusPill>
  if (cell.has_solution) return <StatusPill tone="bad">fail</StatusPill>
  return <StatusPill tone="bad">err</StatusPill>
}

function renderCompiled(cell: Cell) {
  if (!cell.has_solution) return <StatusPill tone="muted">no sol</StatusPill>
  if (
    cell.correct ||
    cell.peak_fraction != null ||
    cell.check_exit_code === 0 ||
    cell.benchmark_exit_code === 0
  ) {
    return <StatusPill tone="good">yes</StatusPill>
  }
  if (cell.check_exit_code != null || cell.benchmark_exit_code != null) {
    return <StatusPill tone="bad">failed</StatusPill>
  }
  return <StatusPill tone="warn">unknown</StatusPill>
}

function renderSpeed(cell: Cell, isWinner: boolean) {
  if (cell.peak_fraction == null) return <span className="cell-missing">-</span>
  const pct = cell.peak_fraction * 100
  return (
    <div className="speed-cell">
      <div className="speed-readout">
        <span className={isWinner ? "cell-score cell-winner" : "cell-score"}>
          {pct.toFixed(1)}%
        </span>
      </div>
      <div className="speed-bar" title={`peak_fraction ${cell.peak_fraction.toFixed(4)}`}>
        <div
          className={isWinner ? "speed-fill speed-fill-winner" : "speed-fill"}
          style={{ width: `${Math.min(100, Math.max(0, pct))}%` }}
        />
      </div>
    </div>
  )
}

function renderTokens(cell: Cell) {
  const usage = cell.usage ?? {}
  return (
    <div className="stacked-cell" title={tokenTitle(cell)}>
      <span>out {fmtCompact(usage.output_tokens)}</span>
      <span>think {fmtCompact(usage.reasoning_tokens)}</span>
      <span>cache {fmtCompact((usage.cache_read_tokens ?? 0) + (usage.cache_creation_tokens ?? 0))}</span>
    </div>
  )
}

function renderRuntime(cell: Cell) {
  return (
    <div className="stacked-cell" title={runtimeTitle(cell)}>
      <span>agent {fmtMaybeDurationText(cell.elapsed_seconds)}</span>
      <span>check {fmtMaybeDurationText(cell.check_elapsed_seconds)}</span>
      <span>bench {fmtMaybeDurationText(cell.benchmark_elapsed_seconds)}</span>
    </div>
  )
}

function renderFileLinks(problem: string, cell: Cell, hasViewer: Set<string>, title: string) {
  return (
    <div className="chip-row">
      <ReferenceChip problem={problem} />
      {hasViewer.has(cell.run_id) ? (
        <a className="link-chip" href={`/runs/${cell.run_id}.html#tab-solution`} title={title}>
          solution
        </a>
      ) : (
        <span className="link-chip link-chip-muted">solution</span>
      )}
    </div>
  )
}

function renderConversation(
  cell: Cell,
  annot: { verdict: string; summary?: string } | undefined,
  hasViewer: Set<string>,
  title: string,
  scored: string,
) {
  const note = annot?.summary || cell.failure_reason || "run details"
  return (
    <div className="conversation-cell">
      <div className="chip-row">
        {hasViewer.has(cell.run_id) ? (
          <a className="link-chip" href={`/runs/${cell.run_id}.html`} title={title}>
            transcript
          </a>
        ) : (
          <span className="link-chip link-chip-muted">transcript</span>
        )}
        <span className="link-chip link-chip-muted">{scored}</span>
      </div>
      <div className="conversation-note" title={note}>
        {note}
      </div>
    </div>
  )
}

function ReferenceChip({ problem }: { problem: string }) {
  return (
    <a
      className="link-chip"
      href={`https://github.com/Infatoshi/KernelBench-Hard/blob/master/problems/${problem}/reference.py`}
    >
      reference
    </a>
  )
}

function StatusPill({
  tone,
  children,
}: {
  tone: "good" | "bad" | "warn" | "muted"
  children: React.ReactNode
}) {
  return <span className={`status-pill status-pill-${tone}`}>{children}</span>
}

function runDate(runId: string) {
  const parts = runDateParts(runId)
  if (!parts.date || !parts.time) return <span className="cell-missing">-</span>
  return (
    <div className="stacked-cell">
      <span>{parts.date}</span>
      <span>{parts.time}</span>
    </div>
  )
}

function runDateParts(runId: string) {
  const m = runId.match(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})/)
  if (!m) return { date: null, time: null }
  return {
    date: `${m[1]}-${m[2]}-${m[3]}`,
    time: `${m[4]}:${m[5]}:${m[6]}`,
  }
}

function fmtCompact(value: number | null | undefined) {
  if (value == null) return "-"
  if (Math.abs(value) >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}m`
  if (Math.abs(value) >= 1_000) return `${(value / 1_000).toFixed(1)}k`
  return fmtInt(value)
}

function fmtMaybeDurationText(value: number | null | undefined) {
  return value == null ? "-" : fmtDuration(value)
}

function tokenTitle(cell: Cell) {
  const usage = cell.usage ?? {}
  return [
    usage.input_tokens != null ? `input ${fmtInt(usage.input_tokens)}` : null,
    usage.output_tokens != null ? `output ${fmtInt(usage.output_tokens)}` : null,
    usage.reasoning_tokens != null ? `reasoning ${fmtInt(usage.reasoning_tokens)}` : null,
    usage.cache_read_tokens != null ? `cache read ${fmtInt(usage.cache_read_tokens)}` : null,
    usage.cache_creation_tokens != null
      ? `cache write ${fmtInt(usage.cache_creation_tokens)}`
      : null,
    usage.total_cost_usd != null ? `cost $${usage.total_cost_usd.toFixed(4)}` : null,
    cell.output_tokens_per_second != null
      ? `${cell.output_tokens_per_second.toFixed(1)} out tok/s`
      : null,
  ].filter(Boolean).join("\n")
}

function runtimeTitle(cell: Cell) {
  return [
    cell.total_elapsed_seconds != null ? `total ${fmtDuration(cell.total_elapsed_seconds)}` : null,
    cell.elapsed_seconds != null ? `agent ${fmtDuration(cell.elapsed_seconds)}` : null,
    cell.check_elapsed_seconds != null ? `check ${fmtDuration(cell.check_elapsed_seconds)}` : null,
    cell.benchmark_elapsed_seconds != null
      ? `benchmark ${fmtDuration(cell.benchmark_elapsed_seconds)}`
      : null,
    cell.gpu_lock_wait_seconds_total != null
      ? `GPU wait ${fmtDuration(cell.gpu_lock_wait_seconds_total)}`
      : null,
    cell.gpu_lock_active_seconds_total != null
      ? `GPU active ${fmtDuration(cell.gpu_lock_active_seconds_total)}`
      : null,
  ].filter(Boolean).join("\n")
}

function compareModelRows(a: Model, b: Model) {
  const labelDiff = shortLabel(a.label).localeCompare(shortLabel(b.label))
  if (labelDiff !== 0) return labelDiff
  const harnessDiff = harnessLabel(a.harness).localeCompare(harnessLabel(b.harness))
  if (harnessDiff !== 0) return harnessDiff
  if (b.pass_count !== a.pass_count) return b.pass_count - a.pass_count
  const bRate = b.total_runs ? b.pass_count / b.total_runs : 0
  const aRate = a.total_runs ? a.pass_count / a.total_runs : 0
  if (bRate !== aRate) return bRate - aRate
  const peakDiff = bestPeak(b.results) - bestPeak(a.results)
  if (peakDiff !== 0) return peakDiff
  return a.label.localeCompare(b.label)
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
    .replace("minimax-claude/MiniMax-M3 [2026-06-01]", "MiniMax M3")
    .replace("minimax-claude/MiniMax-M3 [2026-06-01 max]", "MiniMax M3")
    .replace("minimax-claude/MiniMax-M3", "MiniMax M3")
    .replace("opencode/openrouter-pinned/", "or/")
    .replace("opencode/", "")
    .replace("codex/", "")
    .replace("claude/", "")
    .replace("kimi/", "")
}

function harnessLabel(harness: string) {
  const labels: Record<string, string> = {
    claude: "Claude Code",
    codex: "codex",
    opencode: "Opencode",
    droid: "droid",
    kimi: "kimi",
    cursor: "cursor",
    gemini: "Gemini CLI",
    grok: "Grok Build",
    "zai-claude": "Claude Code",
    "minimax-claude": "Claude Code",
  }
  return labels[harness] ?? harness
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
