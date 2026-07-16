import { readFile } from "node:fs/promises"
import { join } from "node:path"
import type { ModelIndex } from "./models"

// Server-side loader for the model index. Kept separate from ./models so the
// shared types/transforms stay importable from client components.

let cached: Promise<ModelIndex> | null = null

export function loadModelIndex(): Promise<ModelIndex> {
  cached ??= readFile(join(process.cwd(), "public/data/models.json"), "utf8").then(
    (raw) => JSON.parse(raw) as ModelIndex,
  )
  return cached
}
