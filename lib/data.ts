// Data loaders for KernelBench-Hard. Source of truth lives in the public
// GitHub repo at github.com/Infatoshi/KernelBench-Hard. We fetch raw JSON +
// YAML at build time (with hourly ISR) so the website mirrors whatever's
// in the repo.

const REPO_RAW =
  "https://raw.githubusercontent.com/Infatoshi/KernelBench-Hard/master"

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
  const res = await fetch(`${REPO_RAW}/results/leaderboard.json`, {
    next: { revalidate: 3600 },
  })
  if (!res.ok) {
    throw new Error(`Failed to fetch leaderboard.json: ${res.status}`)
  }
  return res.json()
}

// Annotations: list of YAML files under results/annotations/. We list the
// directory via the GitHub API, then fetch each. Cached for an hour to keep
// the API hits low.
export async function loadAnnotations(): Promise<Map<string, Annotation>> {
  const map = new Map<string, Annotation>()
  try {
    const apiRes = await fetch(
      "https://api.github.com/repos/Infatoshi/KernelBench-Hard/contents/results/annotations",
      {
        next: { revalidate: 3600 },
        headers: { Accept: "application/vnd.github.v3+json" },
      },
    )
    if (!apiRes.ok) return map
    type GhItem = { name: string; download_url: string | null; type: string }
    const items: GhItem[] = await apiRes.json()
    const yamls = items.filter(
      (it) => it.type === "file" && it.name.endsWith(".yaml"),
    )
    const fetched = await Promise.all(
      yamls.map(async (it) => {
        if (!it.download_url) return null
        const r = await fetch(it.download_url, {
          next: { revalidate: 3600 },
        })
        if (!r.ok) return null
        const text = await r.text()
        return parseAnnotationYaml(text)
      }),
    )
    for (const a of fetched) if (a) map.set(a.run_id, a)
  } catch {
    // Network failures shouldn't break the page render; cells just don't
    // get the star marker.
  }
  return map
}

// Tiny YAML subset parser. We control the schema so we don't need a full
// YAML library — just need to pull the top-level scalar fields. The quoted
// schema (results/annotations/SCHEMA.md) limits the surface enough that
// this is robust for our use case.
function parseAnnotationYaml(text: string): Annotation | null {
  const get = (key: string): string | null => {
    const re = new RegExp(`^${key}:\\s*(.*)$`, "m")
    const m = text.match(re)
    if (!m) return null
    let v = m[1].trim()
    // Multi-line block scalar: pipe or fold indicator
    if (v === "|" || v === ">") {
      const lineIdx = text.split("\n").findIndex((l) => l.match(re))
      const lines = text.split("\n").slice(lineIdx + 1)
      const collected: string[] = []
      for (const line of lines) {
        if (line.match(/^\S/)) break
        collected.push(line.replace(/^\s{2}/, ""))
      }
      v = collected.join("\n").trim()
    } else {
      // Strip surrounding quotes if present
      if (
        (v.startsWith('"') && v.endsWith('"')) ||
        (v.startsWith("'") && v.endsWith("'"))
      ) {
        v = v.slice(1, -1)
      }
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
