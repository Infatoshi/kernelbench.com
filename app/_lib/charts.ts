// Homepage chart data. Built at request/build time from the same sources the
// /mega and /hard pages read. Mega is scored as speedup over the reference
// megakernel; hard is scored as percent of the hardware roofline (peak_fraction),
// consistent with the site's stated metric.

import { readFile } from "node:fs/promises"
import { join } from "node:path"
import { loadLeaderboard } from "./data"

const REPO_ROOT = process.cwd()

// Formal display names for homepage / chart bars. Prefer adding a row here
// when a new model is published so labels stay pretty; `displayName()` falls
// back to the raw model id if a key is missing so charts never silently drop
// a published row (see AGENTS.md "Publishing results" → site charts).
export const MODEL_NAMES: Record<string, string> = {
  "claude-opus-4-8": "Claude Opus 4.8",
  "claude-fable-5": "Claude Fable 5",
  "claude-sonnet-5": "Claude Sonnet 5",
  "glm-5.2": "GLM-5.2",
  "gpt-5.5": "GPT-5.5",
  "gpt-5.6-sol": "GPT-5.6 Sol",
  "grok-4.5": "Grok 4.5",
  "MiniMax-M3": "MiniMax-M3",
  "kimi-k2.7-code": "Kimi K2.7-Code",
  "kinetic-0715": "Kimi K3 (256k)",
  "kinetic-0715[1m]": "Kimi K3 (1M)",
  "composer-2.5-fast": "Composer 2.5 Fast",
  "gemini-3.5-flash": "Gemini 3.5 Flash",
  "deepseek-v4-pro": "DeepSeek V4 Pro",
  "LongCat-2.0": "LongCat 2.0",
  hy3: "Tencent Hy3",
  "tencent/hy3-preview": "Tencent Hy3",
}

function displayName(modelId: string): string {
  return MODEL_NAMES[modelId] ?? modelId
}

/** Models pulled from the site entirely — keep in sync with
 *  REMOVED_MODEL_SLUGS in models.server.ts. Matches raw leaderboard/CSV ids
 *  like "gemini/gemini-3.5-flash". */
export function isRemovedModel(modelId: string): boolean {
  return /gemini-3\.1-pro|gemini-3\.5-flash/.test(modelId)
}

// GPU series colors — match the published charts (B200 is the NVIDIA accent).
export const GPU_SERIES = [
  { key: "RTX PRO 6000", color: "#4d9fff" },
  { key: "H100", color: "#b07cff" },
  { key: "B200", color: "#76b900" },
]

export type ChartGroup = { label: string; values: (number | null)[] }
export type ChartData = {
  series: { key: string; color: string }[]
  groups: ChartGroup[]
  max: number
  suffix: string
  decimals: number
}

const MEGA_GPU_LABEL: Record<string, string> = {
  "RTX PRO 6000 Blackwell": "RTX PRO 6000",
  H100: "H100",
  B200: "B200",
}

export async function loadMegaChart(): Promise<ChartData> {
  const text = await readFile(
    join(REPO_ROOT, "public/data/mega/results.csv"),
    "utf-8",
  )
  const lines = text.trim().split("\n")
  const header = lines[0].split(",")
  const idx = (k: string) => header.indexOf(k)
  const cell: Record<string, Record<string, number>> = {}
  for (const line of lines.slice(1)) {
    const f = line.split(",")
    if (f[idx("correct")] !== "true") continue
    const model = f[idx("model")]
    if (isRemovedModel(model)) continue
    const gpu = MEGA_GPU_LABEL[f[idx("gpu")]]
    if (!gpu) continue
    ;(cell[model] ??= {})[gpu] = parseFloat(f[idx("score")])
  }
  return assemble(cell, "x", 1)
}

export async function loadHardChart(): Promise<ChartData> {
  const files = {
    "RTX PRO 6000": "benchmarks/hard/results/leaderboard.json",
    H100: "benchmarks/hard/results/leaderboard.h100.json",
    B200: "benchmarks/hard/results/leaderboard.b200.json",
  }
  const cell: Record<string, Record<string, number>> = {}
  for (const [gpu, file] of Object.entries(files)) {
    const lb = await loadLeaderboard(file)
    for (const m of lb.models) {
      if (isRemovedModel(m.model)) continue
      const pfs = Object.values(m.results)
        .filter((c) => c.correct && c.peak_fraction != null)
        .map((c) => c.peak_fraction as number)
      if (pfs.length) {
        ;(cell[m.model] ??= {})[gpu] = (pfs.reduce((a, b) => a + b, 0) / pfs.length) * 100
      }
    }
  }
  return assemble(cell, "%", 0)
}

export type EffPoint = {
  label: string
  x: number // output tokens
  y: number // performance
  frontier: boolean
}
export type EffData = {
  points: EffPoint[]
  ySuffix: string
  yDecimals: number
  yLabel: string
}
export type EffByGpu = Record<string, EffData>

// Candidate GPUs in display order; only those with clean token telemetry for a
// benchmark actually appear (the loader drops empty ones). Tab order matches
// HOME_GPU_TABS (H100 → RTX PRO 6000 → B200); default selection is still B200
// via DEFAULT_GPU. RTX 3090 is hidden site-wide (2026-07-17).
export const EFF_GPU_ORDER = ["H100", "RTX PRO 6000", "B200"]
const MEGA_GPU_MAP: Record<string, string> = {
  "RTX PRO 6000 Blackwell": "RTX PRO 6000",
  H100: "H100",
  B200: "B200",
}
const HARD_SRC: Record<string, { lb: string; runs: string }> = {
  "RTX PRO 6000": { lb: "benchmarks/hard/results/leaderboard.json", runs: "benchmarks/hard/outputs/runs" },
  H100: { lb: "benchmarks/hard/results/leaderboard.h100.json", runs: "benchmarks/hard/outputs/runs-h100" },
  B200: { lb: "benchmarks/hard/results/leaderboard.b200.json", runs: "benchmarks/hard/outputs/runs-b200" },
}

function markFrontier(points: EffPoint[]): EffPoint[] {
  let best = -1
  for (const p of [...points].sort((a, b) => a.x - b.x)) {
    if (p.y > best) {
      p.frontier = true
      best = p.y
    }
  }
  return points
}

// Short labels for the dense scatter (full names crowd the clustered points).
// Same fallback rule as MODEL_NAMES — never drop a published model silently.
const SHORT_NAMES: Record<string, string> = {
  "claude-opus-4-8": "Opus 4.8",
  "claude-fable-5": "Fable 5",
  "claude-sonnet-5": "Sonnet 5",
  "glm-5.2": "GLM-5.2",
  "gpt-5.5": "GPT-5.5",
  "gpt-5.6-sol": "GPT-5.6 Sol",
  "grok-4.5": "Grok 4.5",
  "MiniMax-M3": "MiniMax",
  "kimi-k2.7-code": "Kimi",
  "kinetic-0715": "K3 (256k)",
  "kinetic-0715[1m]": "K3 (1M)",
  "composer-2.5-fast": "Composer",
  "gemini-3.5-flash": "Gemini",
  "deepseek-v4-pro": "DeepSeek",
  "LongCat-2.0": "LongCat",
  hy3: "Hy3",
  "tencent/hy3-preview": "Hy3",
}

function shortName(modelId: string): string {
  return SHORT_NAMES[modelId] ?? displayName(modelId)
}

// Hard GPUs shown on the efficiency chart, all with clean per-model token
// telemetry: RTX (re-extracted), B200 + H100 (fresh resweeps with caches on
// /ephemeral and parallel=3/budget=12000 so sessions complete). Order matches
// HOME_GPU_TABS. RTX 3090 is unpublished. The 50k token floor below drops any
// stale/truncated artifacts.
const HARD_EFF_GPUS = ["H100", "RTX PRO 6000", "B200"]

// Performance vs compute (output tokens), per GPU. Hard tokens are read from the
// committed leaderboard cells (output_tokens baked in by inject_tokens.py) so the
// chart works on Vercel, where outputs/runs is gitignored. A GPU appears only if
// >=2 of its models have clean token telemetry (>1000 output tokens).
export async function loadEfficiency(): Promise<{ mega: EffByGpu; hard: EffByGpu }> {
  // Mega: speedup vs output tokens, grouped by GPU.
  const csv = await readFile(join(REPO_ROOT, "public/data/mega/results.csv"), "utf-8")
  const lines = csv.trim().split("\n")
  const h = lines[0].split(",")
  const ix = (k: string) => h.indexOf(k)
  const megaPts: Record<string, EffPoint[]> = {}
  for (const line of lines.slice(1)) {
    const f = line.split(",")
    if (f[ix("correct")] !== "true") continue
    const gpu = MEGA_GPU_MAP[f[ix("gpu")]]
    if (!gpu) continue
    const tok = parseInt(f[ix("output_tokens")], 10)
    if (!tok || tok < 1000) continue
    if (isRemovedModel(f[ix("model")])) continue
    const name = shortName(f[ix("model")])
    ;(megaPts[gpu] ??= []).push({ label: name, x: tok, y: parseFloat(f[ix("score")]), frontier: false })
  }

  // Hard: avg roofline % vs total output tokens, per GPU. Tokens come from the
  // committed leaderboard cell (output_tokens), not the gitignored runs archive.
  const hardPts: Record<string, EffPoint[]> = {}
  for (const gpu of HARD_EFF_GPUS) {
    const src = HARD_SRC[gpu]
    let lb
    try {
      lb = await loadLeaderboard(src.lb)
    } catch {
      continue
    }
    const pts: EffPoint[] = []
    for (const m of lb.models) {
      if (isRemovedModel(m.model)) continue
      const name = shortName(m.model)
      const pfs: number[] = []
      let tok = 0
      for (const c of Object.values(m.results)) {
        if (!c.correct || c.peak_fraction == null) continue
        pfs.push(c.peak_fraction)
        tok += c.output_tokens ?? 0
      }
      // 50k floor: a model that genuinely solved hard kernels spent >>50k tokens
      // across the deck. Anything less is a stale/truncated-telemetry artifact
      // (e.g. pre-fix runs that never emitted a result event) — exclude it so the
      // frontier isn't distorted by fake hyper-efficiency points. Harnesses that
      // never record output_tokens (e.g. Grok streaming) are omitted from this
      // scatter until tokens are backfilled into the leaderboard cell.
      if (pfs.length && tok > 50000) {
        pts.push({
          label: name,
          x: tok,
          y: (pfs.reduce((a, b) => a + b, 0) / pfs.length) * 100,
          frontier: false,
        })
      }
    }
    hardPts[gpu] = pts
  }

  const pack = (
    byGpu: Record<string, EffPoint[]>,
    order: string[],
    ySuffix: string,
    yDecimals: number,
    yLabel: string,
  ): EffByGpu => {
    const out: EffByGpu = {}
    for (const gpu of order) {
      const pts = byGpu[gpu]
      if (pts && pts.length >= 2) out[gpu] = { points: markFrontier(pts), ySuffix, yDecimals, yLabel }
    }
    return out
  }

  return {
    mega: pack(megaPts, EFF_GPU_ORDER, "x", 1, "speedup over reference"),
    hard: pack(hardPts, HARD_EFF_GPUS, "%", 0, "percent of hardware roofline"),
  }
}

function assemble(
  cell: Record<string, Record<string, number>>,
  suffix: string,
  decimals: number,
): ChartData {
  const series = GPU_SERIES
  // Include every model present in the published data. Formal names come from
  // MODEL_NAMES when known; unknown ids still render so a publish cannot
  // silently omit a new row pending a chart-config edit.
  const models = Object.keys(cell)
  models.sort(
    (a, b) =>
      Math.max(...series.map((s) => cell[b][s.key] ?? 0)) -
      Math.max(...series.map((s) => cell[a][s.key] ?? 0)),
  )
  let max = 0
  const groups: ChartGroup[] = models.map((m) => {
    const values = series.map((s) => {
      const v = cell[m][s.key]
      if (v == null) return null
      if (v > max) max = v
      return v
    })
    return { label: displayName(m), values }
  })
  return { series, groups, max, suffix, decimals }
}
