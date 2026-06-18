"use client"

import { useState, useEffect, useMemo } from "react"
import Link from "next/link"

interface Row {
  gpu: string
  harness: string
  model: string
  problem: string
  correct: boolean
  score: number | null
  tok_s: number | null
  elapsed_s: number | null
  run_id: string
}

export default function MegaPage() {
  const [data, setData] = useState<Row[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch("/data/mega/results.csv")
      .then((r) => r.text())
      .then((csv) => {
        const lines = csv.trim().split("\n")
        const headers = lines[0].split(",")
        const rows: Row[] = []
        for (let i = 1; i < lines.length; i++) {
          if (!lines[i].trim()) continue
          const v = parseCsv(lines[i])
          const r: Record<string, unknown> = {}
          headers.forEach((h, idx) => {
            const val = (v[idx] ?? "").trim()
            if (h === "correct") {
              r[h] = val.toLowerCase() === "true"
            } else if (["score", "tok_s", "elapsed_s"].includes(h)) {
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

  // Group rows by GPU, each group sorted by speedup descending.
  const sorted = useMemo(() => {
    return [...data].sort(
      (a, b) =>
        a.gpu.localeCompare(b.gpu) || (b.score ?? -1) - (a.score ?? -1),
    )
  }, [data])

  return (
    <div className="space-y-12">
      <section>
        <h1 className="prompt cursor text-3xl font-bold text-[var(--color-fg-bright)] glow mb-4">
          mega
        </h1>
        <p className="text-sm text-[var(--color-fg-muted)] mb-6">
          KernelBench-Mega · whole-block megakernels · RTX PRO 6000 Blackwell +
          H100 + B200
        </p>
        <p className="text-[var(--color-fg)] leading-relaxed max-w-3xl">
          KernelBench-Mega tests <strong>whole-block megakernels</strong>:
          instead of grading a single isolated op, the agent fuses an entire
          model block into one kernel. Problem{" "}
          <code>03_kimi_linear_decode</code> is a Kimi-Linear W4A16 hybrid
          decode (4-bit weights, 16-bit activations). The headline metric{" "}
          <strong>score is a speedup over an optimized-PyTorch baseline</strong>
          {" "}(e.g. <code>5.62x</code> = 5.6x faster than the reference), not a
          0-1 roofline fraction. <code>tok/s</code> is decode tokens per second.
          Higher is better for both. Results are shown per GPU.
        </p>
      </section>

      <section>
        <h2 className="text-xl font-bold text-[var(--color-fg-bright)] glow mb-4">
          # leaderboard
        </h2>
        <p className="text-xs text-[var(--color-fg-muted)] mb-4">
          {loading
            ? "loading..."
            : `${sorted.length.toLocaleString()} runs · grouped by GPU, sorted by speedup vs PyTorch baseline`}
        </p>

        <div className="overflow-x-auto box max-h-[60vh]">
          <table className="term tabular text-xs sm:text-sm">
            <thead className="sticky top-0 bg-[var(--color-bg)]">
              <tr>
                <th>gpu</th>
                <th>model</th>
                <th>harness</th>
                <th>problem</th>
                <th title="speedup vs optimized-PyTorch baseline">speedup</th>
                <th title="decode tokens per second">tok/s</th>
                <th>correct</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((r, i) => (
                <tr key={r.run_id || i}>
                  <td className="text-[var(--color-accent)] whitespace-nowrap">
                    {r.gpu}
                  </td>
                  <td className="text-[var(--color-fg-bright)] whitespace-nowrap">
                    {r.model}
                  </td>
                  <td className="text-[var(--color-fg-muted)]">{r.harness}</td>
                  <td className="text-[var(--color-fg-muted)]">{r.problem}</td>
                  <td>
                    {r.score !== null ? (
                      <span className="text-[var(--color-fg-bright)]">
                        {r.score.toFixed(2)}x
                      </span>
                    ) : (
                      "-"
                    )}
                  </td>
                  <td>{r.tok_s !== null ? r.tok_s.toLocaleString() : "-"}</td>
                  <td>
                    {r.correct ? (
                      <span className="cell-clean">PASS</span>
                    ) : (
                      <span className="cell-fail">FAIL</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-xs text-[var(--color-fg-muted)] mt-2">
          speedup = wall-clock speedup over an optimized-PyTorch baseline;
          higher is better.
        </p>
      </section>

      <section className="text-sm text-[var(--color-fg)] border-t border-[var(--color-border)] pt-6">
        Source data:{" "}
        <Link href="https://github.com/Infatoshi/kernelbench.com/tree/master/benchmarks/mega">
          github.com/Infatoshi/kernelbench.com
        </Link>
        {" · "}
        <Link href="/runs">runs</Link>
        {" · "}
        <Link href="/#cite">citation</Link>
      </section>
    </div>
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
