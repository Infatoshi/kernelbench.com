#!/usr/bin/env node
import { readFile, writeFile, mkdir } from "node:fs/promises"
import { dirname, resolve } from "node:path"

const args = parseArgs(process.argv.slice(2))
const inputPath = required(args.input, "--input")
const basePath = args.base ?? "benchmarks/hard/results/leaderboard.json"
const outputPath =
  args.output ?? "benchmarks/hard/results/leaderboard.fresh-preview.json"
const tag = args.tag ?? "2026-05-23 guarded"

const base = JSON.parse(await readFile(resolve(basePath), "utf8"))
const summary = JSON.parse(await readFile(resolve(inputPath), "utf8"))
const rows = Array.isArray(summary.runs) ? summary.runs : []

const problems = [...base.problems]
for (const row of rows) {
  if (row.problem && !problems.includes(row.problem)) problems.push(row.problem)
}

const byLabel = new Map(base.models.map((m) => [m.label, clone(m)]))
for (const row of rows) {
  if (!row.problem || !row.harness || !row.model) continue
  const label = modelLabel(row, tag)
  const model = byLabel.get(label) ?? {
    label,
    harness: row.harness,
    model: row.model,
    effort: row.reasoning_effort ?? "",
    results: {},
    pass_count: 0,
    total_runs: 0,
  }
  model.results[row.problem] = cellFromRow(row)
  byLabel.set(label, model)
}

const models = [...byLabel.values()].map((model) => {
  const cells = Object.values(model.results)
  return {
    ...model,
    pass_count: cells.filter(isScoredCell).length,
    total_runs: cells.length,
  }
})

const next = {
  ...base,
  problems,
  models,
  per_problem: perProblem(problems, models),
  generated_from_summary: {
    input: inputPath,
    tag,
    imported_rows: rows.length,
    generated_at: new Date().toISOString(),
  },
}

await mkdir(dirname(resolve(outputPath)), { recursive: true })
await writeFile(resolve(outputPath), `${JSON.stringify(next, null, 2)}\n`)
console.log(`wrote ${outputPath} (${rows.length} imported rows)`)

function cellFromRow(row) {
  return prune({
    run_id: row.run_id,
    correct: Boolean(row.correct),
    has_solution: Boolean(row.has_solution),
    failure_reason: row.failure_reason ?? null,
    retryable_infra_failure: row.retryable_infra_failure ?? null,
    minimum_useful_output_tokens: row.minimum_useful_output_tokens ?? null,
    peak_fraction: row.peak_fraction ?? null,
    elapsed_seconds: row.elapsed_seconds ?? null,
    total_elapsed_seconds: row.total_elapsed_seconds ?? null,
    check_elapsed_seconds: row.check_elapsed_seconds ?? null,
    benchmark_elapsed_seconds: row.benchmark_elapsed_seconds ?? null,
    check_exit_code: row.check_exit_code ?? null,
    benchmark_exit_code: row.benchmark_exit_code ?? null,
    output_tokens_per_second: row.output_tokens_per_second ?? null,
    usage: prune({
      input_tokens: row.input_tokens ?? null,
      output_tokens: row.output_tokens ?? null,
      cache_read_tokens: row.cache_read_tokens ?? null,
      cache_creation_tokens: row.cache_creation_tokens ?? null,
      reasoning_tokens: row.reasoning_tokens ?? null,
      total_cost_usd: row.total_cost_usd ?? null,
    }),
    session_complete: row.session_complete,
    harness_exit_code: row.harness_exit_code ?? null,
    agent_cuda_disabled: row.agent_cuda_disabled,
    gpu_queue_mode: row.gpu_queue_mode ?? null,
    gpu_lock_calls: row.gpu_lock_calls ?? null,
    gpu_lock_wait_seconds_total: row.gpu_lock_wait_seconds_total ?? null,
    gpu_lock_active_seconds_total: row.gpu_lock_active_seconds_total ?? null,
  })
}

function perProblem(problems, models) {
  return Object.fromEntries(
    problems.map((problem) => {
      const attempted = []
      const passes = []
      for (const model of models) {
        const cell = model.results[problem]
        if (!cell) continue
        attempted.push(cell)
        if (isScoredCell(cell)) {
          passes.push({
            model: model.label,
            peak_fraction: cell.peak_fraction,
          })
        }
      }
      const rankedPasses = passes
        .filter((p) => typeof p.peak_fraction === "number")
        .sort((a, b) => b.peak_fraction - a.peak_fraction)
      const best = rankedPasses[0] ?? null
      return [
        problem,
        {
          n_attempted: attempted.length,
          n_passed: passes.length,
          best_peak_fraction: best?.peak_fraction ?? null,
          best_model: best?.model ?? null,
          ranked_passes: rankedPasses,
        },
      ]
    }),
  )
}

function isScoredCell(cell) {
  return cell.correct && typeof cell.peak_fraction === "number"
}

function modelLabel(row, tag) {
  const effort = row.reasoning_effort ? ` ${row.reasoning_effort}` : ""
  return `${row.harness}/${row.model} [${tag}${effort}]`
}

function parseArgs(raw) {
  const parsed = {}
  for (let i = 0; i < raw.length; i++) {
    const key = raw[i]
    if (!key.startsWith("--")) continue
    parsed[key.slice(2)] = raw[i + 1]
    i++
  }
  return parsed
}

function required(value, name) {
  if (!value) {
    console.error(`missing ${name}`)
    process.exit(2)
  }
  return value
}

function clone(value) {
  return JSON.parse(JSON.stringify(value))
}

function prune(obj) {
  return Object.fromEntries(
    Object.entries(obj).filter(([, value]) => value !== undefined),
  )
}
