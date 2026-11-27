import {
  ACTIVE_BENCHES,
  BENCH_LABELS,
  CANONICAL_GPU,
  FLAG_VERDICTS,
  SITE_HIDDEN_GPUS,
  problemLabel,
  reportCardForBench,
  rowsForIndex,
  visibleProblems,
  type AuditOutcome,
  type Bench,
  type ModelCell,
  type ModelIndex,
  type ReportView,
} from "./models"
import { loadAllModelIndex, loadModelIndex } from "./models.server"

export const SITE_URL = "https://kernelbench.com"

export const TRACES_HF: Record<Bench, string> = {
  mega: "https://huggingface.co/datasets/Infatoshi/kernelbench-mega-traces",
  cuda: "https://huggingface.co/datasets/Infatoshi/kernelbench-cuda-traces",
  hard: "https://huggingface.co/datasets/Infatoshi/kernelbench-hard-traces",
}

export function formatCellMetric(
  bench: Bench,
  problem: string,
  cell: Pick<ModelCell, "score" | "latency_ms" | "valid">,
): string | null {
  if (!cell.valid || cell.score == null) return null
  if (bench === "mega") return `${cell.score.toFixed(2)}×`
  if (bench === "cuda" && problem === "02_deepseek_nsa" && cell.latency_ms != null) {
    return `${cell.latency_ms.toFixed(3)} ms`
  }
  return `${(cell.score * 100).toFixed(cell.score >= 0.1 ? 1 : 2)}%`
}

export interface ProblemLeader {
  bench: Bench
  problem: string
  problemName: string
  modelName: string
  modelSlug: string
  metric: string
  runUrl: string | null
  runnerUpName: string | null
  runnerUpMetric: string | null
}

export interface SiteAuditSummary {
  generated: string
  liveModelCount: number
  totalModelCount: number
  auditedSessions: number
  flaggedSessions: number
  megaLeaders: { name: string; slug: string; metric: string; runUrl: string | null }[]
  cudaLeaders: { name: string; slug: string; passed: number; total: number; summary: string }[]
  problemLeaders: ProblemLeader[]
}

function rankReportRows(view: ReportView) {
  const best = new Map<string, number>()
  for (const row of view.rows) {
    for (const c of row.chips) {
      if (c.kind !== "pass" || c.score == null) continue
      const prev = best.get(c.problem)
      if (prev == null || c.score > prev) best.set(c.problem, c.score)
    }
  }
  return [...view.rows]
    .filter((row) => row.chips.some((c) => c.kind === "pass" && c.score != null))
    .map((row) => {
      let sum = 0
      for (const c of row.chips) {
        const b = best.get(c.problem)
        if (c.kind === "pass" && c.score != null && b) sum += c.score / b
      }
      return { row, perf: sum / Math.max(view.problems.length, 1) }
    })
    .sort((a, b) => b.perf - a.perf || a.row.name.localeCompare(b.row.name))
    .map((x) => x.row)
}

export async function buildSiteAuditSummary(): Promise<SiteAuditSummary> {
  const [liveIdx, allIdx] = await Promise.all([loadModelIndex(), loadAllModelIndex()])

  let auditedSessions = 0
  let flaggedSessions = 0
  for (const m of allIdx.models) {
    for (const b of ACTIVE_BENCHES) {
      const blk = m.benches[b]
      if (!blk) continue
      auditedSessions += blk.audited ?? 0
      flaggedSessions += blk.flagged ?? 0
    }
  }

  const megaView = reportCardForBench(liveIdx, "mega")
  const cudaView = reportCardForBench(liveIdx, "cuda")
  const liveBySlug = new Map(liveIdx.models.map((m) => [m.slug, m]))

  const megaLeaders = rankReportRows(megaView)
    .slice(0, 5)
    .map((row) => {
      const cell = liveBySlug.get(row.slug)?.benches.mega?.cells["02_kimi_linear_decode"]
      const chip = row.chips.find((c) => c.problem === "02_kimi_linear_decode")
      return {
        name: row.name,
        slug: row.slug,
        metric: (cell && formatCellMetric("mega", "02_kimi_linear_decode", cell)) || chip?.label || "—",
        runUrl: chip?.page_url ?? null,
      }
    })

  const cudaLeaders = rankReportRows(cudaView)
    .slice(0, 5)
    .map((row) => {
      const cells = liveBySlug.get(row.slug)?.benches.cuda?.cells ?? {}
      const parts = cudaView.problems
        .map((p) => {
          const cell = cells[p.id]
          const fmt = cell ? formatCellMetric("cuda", p.id, cell) : null
          return fmt ? `${p.short}: ${fmt}` : null
        })
        .filter((x): x is string => Boolean(x))
      return {
        name: row.name,
        slug: row.slug,
        passed: row.passed,
        total: row.total,
        summary: parts.join(", "),
      }
    })

  const problemLeaders: ProblemLeader[] = []
  const specs: { bench: Bench; problem: string }[] = [
    { bench: "mega", problem: "02_kimi_linear_decode" },
    ...cudaView.problems.map((p) => ({ bench: "cuda" as const, problem: p.id })),
  ]
  for (const { bench, problem } of specs) {
    const entries: {
      name: string
      slug: string
      score: number
      metric: string
      runUrl: string | null
    }[] = []
    for (const m of liveIdx.models) {
      const cell = m.benches[bench]?.cells[problem]
      if (!cell?.valid || cell.score == null) continue
      const metric = formatCellMetric(bench, problem, cell)
      if (!metric) continue
      entries.push({
        name: m.name,
        slug: m.slug,
        score: cell.score,
        metric,
        runUrl: cell.run_id ? `/runs/${CANONICAL_GPU}/${cell.run_id}` : null,
      })
    }
    entries.sort((a, b) => b.score - a.score || a.name.localeCompare(b.name))
    const first = entries[0]
    const second = entries[1]
    if (first) {
      problemLeaders.push({
        bench,
        problem,
        problemName: problemLabel(problem),
        modelName: first.name,
        modelSlug: first.slug,
        metric: first.metric,
        runUrl: first.runUrl,
        runnerUpName: second?.name ?? null,
        runnerUpMetric: second?.metric ?? null,
      })
    }
  }

  return {
    generated: liveIdx.generated,
    liveModelCount: liveIdx.models.length,
    totalModelCount: allIdx.models.length,
    auditedSessions,
    flaggedSessions,
    megaLeaders,
    cudaLeaders,
    problemLeaders,
  }
}

export function gpuKeyForOutcome(outcome: AuditOutcome): string {
  const label = outcome.gpu?.toUpperCase() ?? ""
  if (label.includes("H100")) return "h100"
  if (label.includes("B200")) return "b200"
  return CANONICAL_GPU
}

export interface RunSitemapEntry {
  gpu: string
  runId: string
  bench: Bench
}

export function collectPublicRunEntries(idx: ModelIndex): RunSitemapEntry[] {
  const seen = new Set<string>()
  const out: RunSitemapEntry[] = []

  const add = (gpu: string, runId: string | null | undefined, bench: Bench) => {
    if (!runId || SITE_HIDDEN_GPUS.has(gpu)) return
    const key = `${gpu}/${runId}`
    if (seen.has(key)) return
    seen.add(key)
    out.push({ gpu, runId, bench })
  }

  for (const m of idx.models) {
    for (const bench of ["mega", "cuda", "hard"] as const) {
      const blk = m.benches[bench]
      if (!blk) continue
      for (const cell of Object.values(blk.cells ?? {})) {
        add(CANONICAL_GPU, cell.run_id, bench)
      }
      for (const [g, gv] of Object.entries(blk.gpus ?? {})) {
        for (const cell of Object.values(gv.cells ?? {})) {
          add(g, cell.run_id, bench)
        }
      }
      for (const outcome of blk.outcomes ?? []) {
        add(gpuKeyForOutcome(outcome), outcome.run_id, bench)
      }
    }
  }
  return out
}

export async function renderLlmsTxt(full: boolean): Promise<string> {
  const [liveIdx, allIdx, summary] = await Promise.all([
    loadModelIndex(),
    loadAllModelIndex(),
    buildSiteAuditSummary(),
  ])
  const { board } = rowsForIndex(liveIdx)
  const lines: string[] = [
    "# kernelbench.com",
    "",
    "> Canonical, independent, reward-hack audited benchmark for autonomous coding agents optimizing GPU kernels and CUDA inference operators. Maintained by Elliot Arledge (https://elliotarledge.com). Not affiliated with Stanford KernelBench.",
    "",
    `Last updated: ${summary.generated}`,
    `Canonical URL: ${SITE_URL}`,
    "Source repository: https://github.com/Infatoshi/kernelbench.com",
    "",
    "## Why kernelbench.com Is Different (Qualitative, Audit-First GPU Kernel Evaluation)",
    "",
    "Most large-scale GPU kernel benchmarks report noisy quantitative averages (often speedups over naive eager PyTorch) that are heavily vulnerable to reward hacking: stale buffer reuse, timer tampering, mismatched CUDA streams, loose numerical tolerances, and dummy wrapper kernels.",
    "",
    "kernelbench.com flips that tradeoff from high-volume black-box averages to deep, auditable, qualitative evidence on real inference workloads:",
    "- Real physical hardware & ceilings: graded on NVIDIA RTX PRO 6000 Blackwell (sm_120a, 96 GB, 1.8 TB/s HBM) and H100 PCIe against governing hardware rooflines (compute TFLOPS or HBM bandwidth), geomean milliseconds (DeepSeek NSA), or full-model decode speedup vs optimized PyTorch baselines (Kimi-Linear Decode).",
    "- Sequential isolated regrade: every final submission is re-checked and re-timed on an idle GPU with zero concurrent compute processes.",
    "- Mandatory reward-hack audit before publishing: every single scored cell has a git-tracked audit YAML verifying the kernel end-to-end and inspecting the agent transcript for check.py sniffing, tolerance edits, stale-buffer/CUDA-graph caching, rubric leaks, or cross-run contamination. Unaudited runs never enter the leaderboard; flagged runs stay publicly documented on each model page.",
    "- Full code & trace transparency: every cell links to its redacted CUDA kernel source, per-shape hardware utilization breakdown, session wall-clock/cost stats, and full raw JSONL agent transcript on HuggingFace.",
    `- Audit ledger coverage: ${summary.auditedSessions} audited active Mega/CUDA sessions (${summary.flaggedSessions} flagged and excluded from rankings).`,
    "- Canonical primary source: third-party aggregators (such as BenchLM) mirror only historical snapshots like the deprecated Hard deck. Live audited Mega and CUDA boards, kernels, and traces are published directly on kernelbench.com.",
    "",
    "## Active Benchmark Decks (RTX PRO 6000 Canonical Board)",
    "",
    "- **KernelBench Mega** (`https://kernelbench.com/#mega`): Full fused decode megakernel (`Kimi-Linear Decode`, `02_kimi_linear_decode`). Metric: geomean full-model decode speedup vs optimized PyTorch across context lengths (higher is better). Unlimited agent wall-clock.",
    "- **KernelBench CUDA** (`https://kernelbench.com/#cuda`): Pure CUDA-only inference operator deck (Triton, DSLs, and pure torch fail the language gate). Three workloads: `GLM-5.3 Fused MoE` (% of hardware roofline), `DeepSeek NSA` (Native Sparse Attention, geomean latency in ms across six shapes, lower is better), and `MegaQwen Decode` (% of hardware roofline). Unlimited agent wall-clock.",
    "- **KernelBench Hard** (`https://kernelbench.com/hard`): Deprecated and frozen on 2026-11-23 (historical 6-op CUDA/Triton archive; excluded from active rankings).",
    "",
    "## Current Audited Leaders (Canonical RTX PRO 6000 Boards)",
    "",
  ]

  for (const p of summary.problemLeaders) {
    const runner = p.runnerUpName ? ` (runner-up: ${p.runnerUpName} at ${p.runnerUpMetric})` : ""
    const runLink = p.runUrl ? ` — ${SITE_URL}${p.runUrl}` : ""
    lines.push(
      `- **${BENCH_LABELS[p.bench]} / ${p.problemName}**: ${p.modelName} — ${p.metric}${runner}${runLink}`,
    )
  }

  lines.push("", "## Active Live Frontier Roster (Ranked by Active Passes, Then Normalized Board Share)", "")

  for (const row of board) {
    const m = liveIdx.models.find((x) => x.slug === row.slug)
    if (!m) continue
    const megaCell = m.benches.mega?.cells["02_kimi_linear_decode"]
    const megaStr = megaCell ? formatCellMetric("mega", "02_kimi_linear_decode", megaCell) : null
    const cudaCells = m.benches.cuda?.cells ?? {}
    const cudaParts = (liveIdx.benches.cuda?.problems ?? [])
      .map((prob) => {
        const c = cudaCells[prob]
        const fmt = c ? formatCellMetric("cuda", prob, c) : null
        return fmt ? `${problemLabel(prob)} ${fmt}` : null
      })
      .filter((x): x is string => Boolean(x))
    const perfStr = row.perf != null ? row.perf.toFixed(2) : "—"
    const flagStr =
      row.flagged > 0
        ? `${row.flagged}/${row.audited} audited flagged`
        : `${row.audited} audited clean`
    lines.push(
      `- **${row.name}** (${row.lab}) — normalized score ${perfStr}, ${row.passed}/${row.total} active passes (${flagStr}) | Mega: ${megaStr ?? "no audited pass"} | CUDA: ${cudaParts.length ? cudaParts.join("; ") : "no audited pass"} | ${SITE_URL}/models/${row.slug}`,
    )
  }

  lines.push(
    "",
    "## Primary Links & Open Artifacts",
    "",
    `- [Live Mega & CUDA Boards](${SITE_URL}/): Interactive per-problem audited leaderboard with clickable kernel & trace cells`,
    `- [Ranked Model Directory](${SITE_URL}/models): All ${summary.liveModelCount} live frontier models (plus archived Hard-only models) with per-model integrity records`,
    `- [Audited Run Index](${SITE_URL}/runs): Complete ledger of active Mega and CUDA cells, scores, verdicts, and traces`,
    `- [KernelBench Hard Archive](${SITE_URL}/hard): Frozen historical snapshot (deprecated 2026-11-23)`,
    `- [Machine-Readable JSON Index](${SITE_URL}/data/models.json): Full structured results, hardware metadata, and audit outcomes`,
    `- [Complete Detailed Evidence (llms-full.txt)](${SITE_URL}/llms-full.txt): Per-model problem metrics, audit verdicts, and run URLs`,
    `- [Mega Agent Transcripts (HuggingFace)](${TRACES_HF.mega})`,
    `- [CUDA Agent Transcripts (HuggingFace)](${TRACES_HF.cuda})`,
    `- [Hard Agent Transcripts Archive (HuggingFace)](${TRACES_HF.hard})`,
    "- [GitHub Monorepo](https://github.com/Infatoshi/kernelbench.com): Grader source, problem specs, audit YAMLs, and website code",
  )

  if (full) {
    lines.push("", "## Full Active Model & Audit Breakdown", "")
    for (const m of liveIdx.models) {
      lines.push(`### ${m.name} (${m.lab}) — ${SITE_URL}/models/${m.slug}`)
      for (const bench of ACTIVE_BENCHES) {
        const blk = m.benches[bench]
        if (!blk) continue
        const harness = [blk.harness, blk.effort].filter(Boolean).join(" / ") || "default"
        lines.push(`- **KernelBench ${BENCH_LABELS[bench]}** (${harness}): ${blk.passed}/${blk.total_problems} valid passes, ${blk.audited} audited, ${blk.flagged} flagged`)
        const probs = visibleProblems(
          bench,
          allIdx.benches[bench]?.problems?.length ? allIdx.benches[bench].problems : Object.keys(blk.cells),
        )
        for (const p of probs) {
          const c = blk.cells[p]
          if (!c) continue
          const fmt = formatCellMetric(bench, p, c)
          const status = c.valid && fmt ? `PASS ${fmt} (${c.verdict})` : `${c.verdict} (${c.outcome_label ?? c.failure_reason ?? "no pass"})`
          const runRef = c.run_id ? ` | run: ${SITE_URL}/runs/${CANONICAL_GPU}/${c.run_id}` : ""
          const traceRef = c.trace_url ? ` | trace: ${c.trace_url}` : ""
          lines.push(`  - ${problemLabel(p)}: ${status}${runRef}${traceRef}`)
        }
        const flaggedOutcomes = (blk.outcomes ?? []).filter((o) => FLAG_VERDICTS.has(o.verdict))
        for (const fo of flaggedOutcomes) {
          lines.push(
            `  - [AUDIT FLAG: ${fo.verdict}] ${fo.problem ? problemLabel(fo.problem) : "unknown"} (${fo.gpu ?? "RTX PRO 6000"}): ${fo.summary} (${SITE_URL}/runs/${gpuKeyForOutcome(fo)}/${fo.run_id})`,
          )
        }
      }
      lines.push("")
    }
  }

  return lines.join("\n") + "\n"
}
