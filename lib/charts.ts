// Homepage chart data. Built at request/build time from the same sources the
// /mega and /hard pages read. Mega is scored as speedup over the reference
// megakernel; hard is scored as percent of the hardware roofline (peak_fraction),
// consistent with the site's stated metric.

import { readFile } from "node:fs/promises"
import { join } from "node:path"
import { loadLeaderboard } from "./data"

const REPO_ROOT = process.cwd()

export const MODEL_NAMES: Record<string, string> = {
  "claude-opus-4-8": "Claude Opus 4.8",
  "glm-5.2": "GLM-5.2",
  "gpt-5.5": "GPT-5.5",
  "MiniMax-M3": "MiniMax-M3",
  "kimi-k2.7-code": "Kimi K2.7-Code",
  "composer-2.5-fast": "Composer 2.5 Fast",
  "gemini-3.5-flash": "Gemini 3.5 Flash",
  "deepseek-v4-pro": "DeepSeek V4 Pro",
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
  format: (v: number) => string
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
    const gpu = MEGA_GPU_LABEL[f[idx("gpu")]]
    if (!gpu) continue
    ;(cell[model] ??= {})[gpu] = parseFloat(f[idx("score")])
  }
  return assemble(cell, (v) => `${v.toFixed(1)}x`)
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
      const pfs = Object.values(m.results)
        .filter((c) => c.correct && c.peak_fraction != null)
        .map((c) => c.peak_fraction as number)
      if (pfs.length) {
        ;(cell[m.model] ??= {})[gpu] = (pfs.reduce((a, b) => a + b, 0) / pfs.length) * 100
      }
    }
  }
  return assemble(cell, (v) => `${v.toFixed(0)}%`)
}

function assemble(
  cell: Record<string, Record<string, number>>,
  fmt: (v: number) => string,
): ChartData {
  const series = GPU_SERIES
  const models = Object.keys(MODEL_NAMES).filter((m) => cell[m])
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
    return { label: MODEL_NAMES[m], values }
  })
  return { series, groups, max, format: fmt }
}
