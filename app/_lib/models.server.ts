import { readFile } from "node:fs/promises"
import { join } from "node:path"
import type { ModelIndex } from "./models"

// Server-side loader for the model index. Kept separate from ./models so the
// shared types/transforms stay importable from client components.

/** Current models shown on the homepage and in the model directory.
 * Historical model pages and run pages still load the full index. */
const LIVE_MODEL_SLUGS = new Set([
  "claude-fable-5",
  "claude-fable-5-1",
  "claude-opus-5",
  "deepseek-v4-flash-0731",
  "gemini-3.8-flash-high",
  "glm-5.3",
  "gpt-5.6-sol",
  "gpt-6-astra-pro",
  "grok-4.6",
  "kinetic-0715",
  "muse-spark-1.3",
  "ox-alpha",
  "qwen3.8-max",
])

/** Models with published cells that are deliberately off the homepage and
 * /models roster (superseded or historical). Every slug in models.json must be
 * here or in LIVE_MODEL_SLUGS; scripts/check_publish_gates.py fails the
 * publish otherwise, so a fresh model cannot ship invisible again. */
export const RETIRED_MODEL_SLUGS = new Set([
  "claude-opus-4-8",
  "claude-sonnet-5",
  "composer-2.5-fast",
  "deepseek-v4-pro",
  "glm-5.2",
  "grok-4.5",
  "hy3",
  "inkling",
  "kinetic-0715-1m",
  "longcat-2.0",
  "minimax-m3",
  "qwen3.8-max-preview",
])

// No module-level cache: Next dev (and prod workers) keep one module graph per
// route segment, so a `cached ??=` here pins each page to whatever models.json
// said at that segment's first request — the roster visibly desyncs across
// pages after a publish. The file is ~1 MB; reading it per request is noise.
export function loadAllModelIndex(): Promise<ModelIndex> {
  return readFile(join(process.cwd(), "public/data/models.json"), "utf8").then(
    (raw) => JSON.parse(raw) as ModelIndex,
  )
}

export function loadModelIndex(): Promise<ModelIndex> {
  return loadAllModelIndex().then((idx) => {
    idx.models = idx.models.filter((m) => LIVE_MODEL_SLUGS.has(m.slug))
    return idx
  })
}
