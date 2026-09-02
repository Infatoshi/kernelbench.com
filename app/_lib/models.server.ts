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
  "glm-5.3",
  "gpt-5.6-sol",
  "grok-4.6",
  "kinetic-0715",
  "ox-alpha",
  "qwen3.8-max",
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
