// Data loaders for KernelBench. Source of truth lives in this monorepo
// under benchmarks/<version>/. The website reads from the local filesystem
// at build time — no network fetch needed since data and site ship together.

import { readFile, readdir } from "node:fs/promises"
import { join } from "node:path"

const REPO_ROOT = process.cwd()

export type Cell = {
  run_id: string
  correct: boolean
  has_solution: boolean
  peak_fraction: number | null
  elapsed_seconds?: number | null
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
}

export type Annotation = {
  run_id: string
  model: string
  problem: string
  verdict: "clean" | "rubric_leak" | "reward_hack" | "interesting" | "bug"
  summary: string
  implication?: string
}

export async function loadLeaderboard(): Promise<Leaderboard> {
  const path = join(REPO_ROOT, "benchmarks/v-hard/results/leaderboard.json")
  const text = await readFile(path, "utf-8")
  return JSON.parse(text)
}

export async function loadAnnotations(): Promise<Map<string, Annotation>> {
  const map = new Map<string, Annotation>()
  const dir = join(REPO_ROOT, "benchmarks/v-hard/results/annotations")
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
