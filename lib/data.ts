// Data loaders for the public benchmark result pages. Source of truth lives in this monorepo
// under benchmarks/<version>/. The website reads from the local filesystem
// at build time — no network fetch needed since data and site ship together.

import { readFile, readdir } from "node:fs/promises"
import { join } from "node:path"

const REPO_ROOT = process.cwd()

export type Cell = {
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

export type Model = {
  label: string
  harness: string
  model: string
  effort: string
  results: Record<string, Cell>
  pass_count: number
  total_runs: number
}

export type Leaderboard = {
  schema_version: number
  hardware: {
    name: string
    sm: string
    vram_gb: number
    peak_bandwidth_gb_s: number
  }
  problems: string[]
  models: Model[]
  per_problem: Record<
    string,
    {
      n_attempted: number
      n_passed: number
      best_peak_fraction: number | null
      best_model: string | null
      ranked_passes: { model: string; peak_fraction: number }[]
    }
  >
  generated_from_summary?: {
    input: string
    tag: string
    imported_rows: number
    generated_at: string
  }
}

export type Annotation = {
  run_id: string
  model: string
  problem: string
  verdict:
    | "clean"
    | "rubric_leak"
    | "reward_hack"
    | "contamination"
    | "interesting"
    | "bug"
    | "harness_limited"
  summary: string
  implication?: string
}

export type AuditFlag =
  | "contamination"
  | "reward_hack"
  | "rubric_leak"
  | "bug"
  | "interesting"

export type RunAudit = {
  flags: AuditFlag[]
  summary: string
  evidence: string[]
  invalid: boolean
}

export type VariantTiming = {
  ms: number
  tflops: number
  gbps: number
  n_shapes: number
}

export type ProblemBaseline = {
  eager?: VariantTiming
  compiled?: VariantTiming
  sota?: VariantTiming
  _solution_peak_fraction_baseline?: number
}

export type ProblemBaselines = {
  _schema: string
  _generated: string
  _hardware: string
  problems: Record<string, ProblemBaseline>
}

export async function loadBaselines(): Promise<ProblemBaselines | null> {
  const path = join(REPO_ROOT, "benchmarks/hard/results/problem_baselines.json")
  try {
    const text = await readFile(path, "utf-8")
    return JSON.parse(text)
  } catch {
    return null
  }
}

export async function loadLeaderboard(
  file = process.env.KERNELBENCH_HARD_LEADERBOARD ??
    "benchmarks/hard/results/leaderboard.json",
): Promise<Leaderboard> {
  const path = join(REPO_ROOT, file)
  const text = await readFile(path, "utf-8")
  return JSON.parse(text)
}

export async function loadAnnotations(): Promise<Map<string, Annotation>> {
  const map = new Map<string, Annotation>()
  const dir = join(REPO_ROOT, "benchmarks/hard/results/annotations")
  try {
    const entries = await readdir(dir)
    for (const name of entries) {
      if (!name.endsWith(".yaml")) continue
      const text = await readFile(join(dir, name), "utf-8")
      const a = parseAnnotationYaml(text)
      if (a) map.set(a.run_id, a)
    }
  } catch {
    // Missing annotations dir is non-fatal — cells just don't get the marker.
  }
  return map
}

export async function loadRunAudits(): Promise<Map<string, RunAudit>> {
  const map = new Map<string, RunAudit>()
  const dir = join(REPO_ROOT, "public/runs")
  try {
    const entries = await readdir(dir)
    for (const name of entries) {
      if (!name.endsWith(".html")) continue
      const runId = name.slice(0, -5)
      const text = await readFile(join(dir, name), "utf-8")
      const audit = auditRunViewer(runId, text)
      if (audit.flags.length) map.set(runId, audit)
    }
  } catch {
    // Missing run viewers are non-fatal.
  }
  return map
}

function auditRunViewer(runId: string, html: string): RunAudit {
  const flags = new Set<AuditFlag>()
  const evidence = new Set<string>()

  const add = (flag: AuditFlag, item: string) => {
    flags.add(flag)
    if (evidence.size < 5) evidence.add(item)
  }

  if (
    /Read <span class="filepath">\/home\/infatoshi\/\.claude\/projects\/[^<]*\/memory\//.test(
      html,
    )
  ) {
    add("contamination", "read private Claude project memory")
  }
  if (
    /Read <span class="filepath">\/home\/infatoshi\/\.codex\/memories\//.test(html)
  ) {
    add("contamination", "read private Codex memory")
  }

  const priorArtifact =
    /\/home\/infatoshi\/(?:benchmarks|cuda)\/KernelBench-Hard\/outputs\/runs\/([^/<\s]+)\/([^<"\s]*(?:solution\.py|result\.json|benchmark\.log|transcript\.jsonl)[^<"\s]*)/g
  for (const match of html.matchAll(priorArtifact)) {
    const seenRunId = match[1]
    const path = match[2]
    if (seenRunId !== runId && !path.includes("/repo/.venv/")) {
      add("contamination", `read prior run artifact ${seenRunId}/${shortAuditPath(path)}`)
    }
  }

  if (
    /Read <span class="filepath">\/home\/infatoshi\/benchmarks\/KernelBench-Hard\/LEADERBOARD\.md<\/span>/.test(
      html,
    ) ||
    /"file_path":\s*"\/home\/infatoshi\/benchmarks\/KernelBench-Hard\/LEADERBOARD\.md"/.test(
      html,
    )
  ) {
    add("contamination", "read historical LEADERBOARD.md")
  }

  if (
    /(Write|Edit) <span class="filepath">\/home\/infatoshi\/benchmarks\/KernelBench-Hard\/outputs\/runs\/[^<]*\/repo\/(?:problems\/[^/]+\/(?:problem\.yaml|check\.py|benchmark\.py|reference\.py|shapes\.py)|src\/eval\/[^<]+)<\/span>/.test(
      html,
    ) ||
    /(Write|Edit) <span class="filepath">\/home\/infatoshi\/benchmarks\/KernelBench-Hard\/src\/eval\/[^<]+<\/span>/.test(
      html,
    )
  ) {
    add("reward_hack", "edited benchmark or judge file")
  }

  const flagList = [...flags]
  return {
    flags: flagList,
    summary: auditSummary(flagList, [...evidence]),
    evidence: [...evidence],
    invalid: flagList.includes("contamination") || flagList.includes("reward_hack"),
  }
}

function shortAuditPath(path: string) {
  const parts = path.split("/")
  return parts.slice(-2).join("/")
}

function auditSummary(flags: AuditFlag[], evidence: string[]) {
  if (!flags.length) return ""
  const prefix = flags.includes("reward_hack")
    ? "quarantined: benchmark/judge tampering"
    : "quarantined: prior-state contamination"
  return evidence.length ? `${prefix}; ${evidence.join("; ")}` : prefix
}

// Tiny YAML subset parser. We control the schema (results/annotations/SCHEMA.md)
// so we don't pull in a full YAML library — flat key:value pairs plus block
// scalars (`|` and `>`) is the entire surface.
function parseAnnotationYaml(text: string): Annotation | null {
  const lines = text.split("\n")
  const get = (key: string): string | null => {
    const re = new RegExp(`^${key}:\\s*(.*)$`)
    const idx = lines.findIndex((l) => l.match(re))
    if (idx < 0) return null
    const m = lines[idx].match(re)!
    let v = m[1].trim()
    if (v === "|" || v === ">") {
      const collected: string[] = []
      for (let j = idx + 1; j < lines.length; j++) {
        const line = lines[j]
        if (line.match(/^\S/) || line.match(/^$/)) {
          if (line.match(/^\S/)) break
          collected.push("")
          continue
        }
        collected.push(line.replace(/^\s{2}/, ""))
      }
      v = collected.join("\n").trim()
    } else if (
      (v.startsWith('"') && v.endsWith('"')) ||
      (v.startsWith("'") && v.endsWith("'"))
    ) {
      v = v.slice(1, -1)
    }
    return v
  }
  const run_id = get("run_id")
  const verdict = get("verdict")
  if (!run_id || !verdict) return null
  return {
    run_id,
    model: get("model") || "",
    problem: get("problem") || "",
    verdict: verdict as Annotation["verdict"],
    summary: get("summary") || "",
    implication: get("implication") || undefined,
  }
}
