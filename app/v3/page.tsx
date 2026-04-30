"use client"

import { useState, useEffect, useMemo } from "react"
import Image from "next/image"

interface Row {
  model: string
  gpu: string
  level: number
  problem: string
  compiled: boolean
  correct: boolean
  speedup: number | null
  turns: number
  total_tokens: number
  estimated_cost_usd: number | null
  op_type: string
  baseline_type: string
  precision_used: string
  solution_link: string
  baseline_link: string
}

type FilterKey = "model" | "gpu" | "level" | "status" | "op_type" | "baseline_type"

export default function V3Page() {
  const [data, setData] = useState<Row[]>([])
  const [loading, setLoading] = useState(true)
  const [filters, setFilters] = useState<Record<FilterKey, string>>({
    model: "all",
    gpu: "all",
    level: "all",
    status: "all",
    op_type: "all",
    baseline_type: "all",
  })
  const [sortKey, setSortKey] = useState<keyof Row>("model")
  const [sortDir, setSortDir] = useState<"asc" | "desc">("asc")

  useEffect(() => {
    fetch("/data/v3/results.csv")
      .then((r) => r.text())
      .then((csv) => {
        const lines = csv.trim().split("\n")
        const headers = lines[0].split(",")
        const rows: Row[] = []
        for (let i = 1; i < lines.length; i++) {
          const v = parseCsv(lines[i])
          const r: Record<string, unknown> = {}
          headers.forEach((h, idx) => {
            const val = v[idx] ?? ""
            if (["compiled", "correct"].includes(h)) {
              r[h] = val.toLowerCase() === "true"
            } else if (
              ["level", "turns", "total_tokens"].includes(h)
            ) {
              r[h] = parseInt(val) || 0
            } else if (["speedup", "estimated_cost_usd"].includes(h)) {
              r[h] =
                val === "" || val === "nan" || val === "None"
                  ? null
                  : parseFloat(val)
            } else {
              r[h] = val
            }
          })
          rows.push(r as unknown as Row)
        }
        setData(rows)
        setLoading(false)
      })
  }, [])

  const unique = useMemo(
    () => ({
      model: [...new Set(data.map((d) => d.model))].sort(),
      gpu: [...new Set(data.map((d) => d.gpu))].sort(),
      level: [...new Set(data.map((d) => d.level))].sort((a, b) => a - b),
      op_type: [...new Set(data.map((d) => d.op_type))].filter(Boolean).sort(),
      baseline_type: [...new Set(data.map((d) => d.baseline_type))]
        .filter(Boolean)
        .sort(),
    }),
    [data],
  )

  const filtered = useMemo(() => {
    return data.filter((r) => {
      if (filters.model !== "all" && r.model !== filters.model) return false
      if (filters.gpu !== "all" && r.gpu !== filters.gpu) return false
      if (filters.level !== "all" && r.level !== parseInt(filters.level))
        return false
      if (filters.status === "correct" && !r.correct) return false
      if (filters.status === "failed" && r.correct) return false
      if (
        filters.status === "beats_baseline" &&
        (!r.correct || (r.speedup ?? 0) < 1.0)
      )
        return false
      if (filters.op_type !== "all" && r.op_type !== filters.op_type)
        return false
      if (
        filters.baseline_type !== "all" &&
        r.baseline_type !== filters.baseline_type
      )
        return false
      return true
    })
  }, [data, filters])

  const sorted = useMemo(() => {
    return [...filtered].sort((a, b) => {
      const av = a[sortKey]
      const bv = b[sortKey]
      if (av === null || av === undefined) return 1
      if (bv === null || bv === undefined) return -1
      if (typeof av === "number" && typeof bv === "number") {
        return sortDir === "asc" ? av - bv : bv - av
      }
      const as = String(av)
      const bs = String(bv)
      return sortDir === "asc" ? as.localeCompare(bs) : bs.localeCompare(as)
    })
  }, [filtered, sortKey, sortDir])

  const stats = useMemo(() => {
    const t = filtered.length
    const compiled = filtered.filter((r) => r.compiled).length
    const correct = filtered.filter((r) => r.correct).length
    const beats = filtered.filter(
      (r) => r.correct && (r.speedup ?? 0) >= 1.0,
    ).length
    return { total: t, compiled, correct, beats }
  }, [filtered])

  const handleSort = (k: keyof Row) => {
    if (sortKey === k) setSortDir(sortDir === "asc" ? "desc" : "asc")
    else {
      setSortKey(k)
      setSortDir("asc")
    }
  }

  return (
    <div className="space-y-12">
      <section>
        <h1 className="prompt cursor text-3xl font-bold text-[var(--color-fg-bright)] glow mb-4">
          kernelbench v3
        </h1>
        <p className="text-sm text-[var(--color-fg-muted)] mb-6">
          10 models · RTX 3090 + H100 + B200 · 4 difficulty levels · 2071 evaluations
        </p>
        <p className="text-[var(--color-fg)] leading-relaxed max-w-3xl">
          The previous-generation benchmark. After METR&apos;s &ldquo;Measuring Automated
          Kernel Engineering&rdquo; paper showed the original Stanford KernelBench was
          riddled with exploits (no-op kernels passing via memory aliasing, models
          monkey-patching <code>torch.cuda.synchronize</code>, constant functions like{" "}
          <code>mean(softmax(x)) == 1.0</code>), v3 was rebuilt from scratch with
          adaptive baselines, multi-seed correctness, modern architectures (DeepSeek
          MLA, MoE, FP8/INT4 GEMM, GatedDeltaNet), and tracked cost per evaluation.
        </p>
      </section>

      <section>
        <h2 className="text-xl font-bold text-[var(--color-fg-bright)] glow mb-4">
          # results
        </h2>
        <div className="grid sm:grid-cols-2 gap-4">
          {[
            ["results_overall.png", "overall pass rates"],
            ["results_heatmap.png", "per-level heatmap"],
            ["speedup_distribution.png", "speedup distribution"],
            ["level_breakdown.png", "per-level breakdown"],
            ["cost_vs_accuracy.png", "cost vs accuracy"],
            ["compilation_funnel.png", "compilation funnel"],
          ].map(([f, label]) => (
            <figure key={f} className="box p-2">
              <Image
                src={`/v3/${f}`}
                alt={label}
                width={1400}
                height={700}
                className="w-full h-auto"
                unoptimized
              />
              <figcaption className="text-xs text-[var(--color-fg-muted)] text-center mt-2">
                {label}
              </figcaption>
            </figure>
          ))}
        </div>
      </section>

      <section>
        <h2 className="text-xl font-bold text-[var(--color-fg-bright)] glow mb-4">
          # explorer
        </h2>
        <p className="text-xs text-[var(--color-fg-muted)] mb-4">
          {loading
            ? "loading..."
            : `${stats.total.toLocaleString()} rows · ${stats.compiled} compiled · ${stats.correct} correct · ${stats.beats} beat baseline`}
        </p>
        <div className="grid sm:grid-cols-3 lg:grid-cols-6 gap-2 mb-4 text-xs">
          <FilterSelect
            label="model"
            value={filters.model}
            onChange={(v) => setFilters({ ...filters, model: v })}
            options={unique.model}
          />
          <FilterSelect
            label="gpu"
            value={filters.gpu}
            onChange={(v) => setFilters({ ...filters, gpu: v })}
            options={unique.gpu}
          />
          <FilterSelect
            label="level"
            value={filters.level}
            onChange={(v) => setFilters({ ...filters, level: v })}
            options={unique.level.map(String)}
          />
          <FilterSelect
            label="status"
            value={filters.status}
            onChange={(v) => setFilters({ ...filters, status: v })}
            options={["correct", "failed", "beats_baseline"]}
          />
          <FilterSelect
            label="op_type"
            value={filters.op_type}
            onChange={(v) => setFilters({ ...filters, op_type: v })}
            options={unique.op_type}
          />
          <FilterSelect
            label="baseline_type"
            value={filters.baseline_type}
            onChange={(v) => setFilters({ ...filters, baseline_type: v })}
            options={unique.baseline_type}
          />
        </div>

        <div className="overflow-x-auto box max-h-[60vh]">
          <table className="term tabular text-xs sm:text-sm">
            <thead className="sticky top-0 bg-[var(--color-bg)]">
              <tr>
                {(
                  [
                    "model",
                    "gpu",
                    "level",
                    "problem",
                    "correct",
                    "speedup",
                    "turns",
                    "total_tokens",
                    "estimated_cost_usd",
                  ] as (keyof Row)[]
                ).map((k) => (
                  <th
                    key={k}
                    onClick={() => handleSort(k)}
                    className="cursor-pointer hover:text-[var(--color-accent)]"
                  >
                    {k.replace(/_/g, " ")}
                    {sortKey === k ? (sortDir === "asc" ? " ↑" : " ↓") : ""}
                  </th>
                ))}
                <th>code</th>
              </tr>
            </thead>
            <tbody>
              {sorted.slice(0, 1000).map((r, i) => (
                <tr key={i}>
                  <td className="text-[var(--color-fg-bright)] whitespace-nowrap">
                    {r.model}
                  </td>
                  <td>{r.gpu}</td>
                  <td>L{r.level}</td>
                  <td className="text-[var(--color-fg-muted)]">{r.problem}</td>
                  <td>
                    {r.correct ? (
                      <span className="cell-clean">PASS</span>
                    ) : (
                      <span className="cell-fail">FAIL</span>
                    )}
                  </td>
                  <td>
                    {r.speedup !== null ? (
                      <span
                        className={
                          r.speedup >= 1.0
                            ? "text-[var(--color-fg-bright)]"
                            : "text-[var(--color-fg-dim)]"
                        }
                      >
                        {r.speedup.toFixed(2)}x
                      </span>
                    ) : (
                      "-"
                    )}
                  </td>
                  <td>{r.turns}</td>
                  <td>{r.total_tokens.toLocaleString()}</td>
                  <td>
                    {r.estimated_cost_usd !== null
                      ? `$${r.estimated_cost_usd.toFixed(3)}`
                      : "-"}
                  </td>
                  <td className="whitespace-nowrap">
                    {r.solution_link ? (
                      <a
                        href={r.solution_link.replace(
                          "/data/kernelbench-v3/",
                          "/data/v3/",
                        )}
                        target="_blank"
                        rel="noreferrer"
                        className="text-[var(--color-fg-bright)] no-underline hover:text-[var(--color-accent)] mr-2"
                        title="open the model's solution.py"
                      >
                        sol
                      </a>
                    ) : null}
                    {r.baseline_link ? (
                      <a
                        href={r.baseline_link.replace(
                          "/data/kernelbench-v3/",
                          "/data/v3/",
                        )}
                        target="_blank"
                        rel="noreferrer"
                        className="text-[var(--color-fg-muted)] no-underline hover:text-[var(--color-accent)]"
                        title="open the PyTorch reference"
                      >
                        ref
                      </a>
                    ) : null}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {sorted.length > 1000 && (
          <p className="text-xs text-[var(--color-fg-muted)] mt-2">
            showing first 1000 of {sorted.length.toLocaleString()} matching rows. apply filters to narrow.
          </p>
        )}
      </section>
    </div>
  )
}

function FilterSelect({
  label,
  value,
  onChange,
  options,
}: {
  label: string
  value: string
  onChange: (v: string) => void
  options: string[]
}) {
  return (
    <label className="block">
      <div className="text-[var(--color-fg-muted)] mb-1">{label}</div>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full bg-[var(--color-bg)] border border-[var(--color-border)] text-[var(--color-fg)] px-2 py-1 focus:outline-none focus:border-[var(--color-fg-bright)]"
      >
        <option value="all">all</option>
        {options.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
    </label>
  )
}

function parseCsv(line: string): string[] {
  const out: string[] = []
  let cur = ""
  let q = false
  for (let i = 0; i < line.length; i++) {
    const c = line[i]
    if (c === '"') q = !q
    else if (c === "," && !q) {
      out.push(cur)
      cur = ""
    } else cur += c
  }
  out.push(cur)
  return out
}
