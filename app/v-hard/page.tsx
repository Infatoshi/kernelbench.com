import Link from "next/link"
import { readdir } from "node:fs/promises"
import { join } from "node:path"
import { loadLeaderboard, loadAnnotations } from "@/lib/data"

async function loadAvailableViewers(): Promise<Set<string>> {
  try {
    const entries = await readdir(join(process.cwd(), "public/runs"))
    return new Set(
      entries
        .filter((n) => n.endsWith(".html"))
        .map((n) => n.slice(0, -5)),
    )
  } catch {
    return new Set()
  }
}

const PROBLEMS = [
  { key: "01_fp8_gemm", short: "01 fp8" },
  { key: "02_kda_cutlass", short: "02 kda" },
  { key: "03_paged_attention", short: "03 paged" },
  { key: "04_kahan_softmax", short: "04 kahan" },
  { key: "05_topk_bitonic", short: "05 topk" },
  { key: "06_sonic_moe_swiglu", short: "06 moe" },
  { key: "07_w4a16_gemm", short: "07 w4a16" },
]

export default async function VHardPage() {
  const [lb, annotations, hasViewer] = await Promise.all([
    loadLeaderboard(),
    loadAnnotations(),
    loadAvailableViewers(),
  ])

  return (
    <div className="space-y-12">
      <section>
        <h1 className="prompt cursor text-3xl font-bold text-[var(--color-fg-bright)] glow mb-4">
          kernelbench v-hard
        </h1>
        <p className="text-sm text-[var(--color-fg-muted)] mb-6">
          12 models × 7 problems · RTX PRO 6000 Blackwell · sm_120 · 96 GB GDDR7 · 1.8 TB/s
        </p>
        <p className="text-[var(--color-fg)] leading-relaxed max-w-3xl">
          A focused successor to KernelBench v3. One Blackwell GPU, seven hand-designed problems, real coding-agent CLIs as the harness. Twelve frontier models swept; only GPT-5.5 xhigh solved every problem. Two of the seven problems leak the rubric — five models all took the same bf16 shortcut on FP8 GEMM, and the only model that implemented Kahan compensated summation scored lowest of the seven passes.
        </p>
      </section>

      <section>
        <h2 className="text-xl font-bold text-[var(--color-fg-bright)] mb-4 glow">
          # leaderboard
        </h2>
        <p className="text-xs text-[var(--color-fg-muted)] mb-4">
          cells = peak_fraction (fraction of the relevant hardware ceiling). FAIL = solution written but missed correctness. ERR = no solution produced. <span className="text-[var(--color-warn)]">★</span> = annotation attached. <span className="text-[var(--color-fg-bright)]">click any cell to open the full transcript viewer</span> — every tool call, every reasoning step, the solution.py, the check.log.
        </p>
        <div className="overflow-x-auto box">
          <table className="term tabular text-xs sm:text-sm">
            <thead>
              <tr>
                <th className="sticky left-0 bg-[var(--color-bg)]">model</th>
                {PROBLEMS.map((p) => (
                  <th key={p.key} className="text-right">
                    {p.short}
                  </th>
                ))}
                <th className="text-right">PASS</th>
              </tr>
            </thead>
            <tbody>
              {lb.models.map((m) => (
                <tr key={m.label}>
                  <td className="sticky left-0 bg-[var(--color-bg)] text-[var(--color-fg-bright)] whitespace-nowrap">
                    {shortLabel(m.label)}
                    {m.effort ? (
                      <span className="text-[var(--color-fg-muted)]">
                        {" "}
                        [{m.effort}]
                      </span>
                    ) : null}
                  </td>
                  {PROBLEMS.map((p) => {
                    const cell = m.results[p.key]
                    return (
                      <td key={p.key} className="text-right">
                        {renderCell(cell, annotations, hasViewer)}
                      </td>
                    )
                  })}
                  <td className="text-right text-[var(--color-fg-bright)]">
                    {m.pass_count}/{m.total_runs}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section>
        <h2 className="text-xl font-bold text-[var(--color-fg-bright)] mb-4 glow">
          # per-problem ceilings
        </h2>
        <div className="box overflow-x-auto">
          <table className="term tabular text-sm">
            <thead>
              <tr>
                <th>problem</th>
                <th>best peak</th>
                <th>best model</th>
                <th>n correct</th>
              </tr>
            </thead>
            <tbody>
              {PROBLEMS.map((p) => {
                const pp = lb.per_problem[p.key]
                return (
                  <tr key={p.key}>
                    <td>{p.key}</td>
                    <td className="text-[var(--color-fg-bright)]">
                      {pp.best_peak_fraction
                        ? pp.best_peak_fraction.toFixed(3)
                        : "-"}
                    </td>
                    <td className="text-[var(--color-fg-muted)]">
                      {pp.best_model ? shortLabel(pp.best_model) : "-"}
                    </td>
                    <td>
                      {pp.n_passed}/{pp.n_attempted}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </section>

      <section>
        <h2 className="text-xl font-bold text-[var(--color-fg-bright)] mb-4 glow">
          # rubric leaks
        </h2>
        <p className="text-[var(--color-fg)] leading-relaxed mb-4 max-w-3xl">
          Two cells in the leaderboard promise something the benchmark doesn&apos;t actually measure. They&apos;re marked{" "}
          <span className="text-[var(--color-warn)]">★</span> for a reason.
        </p>

        <div className="space-y-6">
          <LeakCard
            title="01 fp8_gemm — bf16 dressup"
            cluster={[
              { model: "claude-opus-4-7 [max]", peak: "0.534" },
              { model: "mimo-v2.5-pro", peak: "0.434" },
              { model: "qwen3.6-plus", peak: "0.431" },
              { model: "qwen3.6-max-preview", peak: "0.429" },
              { model: "gpt-5.5 [xhigh]", peak: "0.423" },
            ]}
            body={
              <>
                Every passing solution at peak ≥ 0.4 casts the fp8 inputs to bf16
                inside the kernel and runs a bf16 GEMM. Both Opus 4.7 max and GPT-5.5
                xhigh explicitly pin to{" "}
                <code className="text-[var(--color-accent)]">cutlass::arch::Sm80</code>
                {" "}— Ampere CUTLASS, not the SM120 Blackwell FP8 tensor cores the
                problem name implies. Opus&apos;s source comment is unusually direct:
                <em className="text-[var(--color-fg-bright)] not-italic">
                  {" "}
                  &ldquo;follow the codex baseline (BF16 GEMM internally)...&rdquo;
                </em>{" "}
                The peak fractions on this row reflect bf16 GEMM optimization quality
                on fp8-typed inputs, not FP8 tensor core skill.
              </>
            }
          />

          <LeakCard
            title="04 kahan_softmax — Kahan compensation skipped"
            cluster={[
              { model: "gpt-5.5 [xhigh]", peak: "0.363" },
              { model: "claude-opus-4-7 [max]", peak: "0.317" },
              { model: "deepseek-v4-flash", peak: "0.138" },
              { model: "glm-5.1", peak: "0.125" },
              { model: "mimo-v2.5-pro", peak: "0.121" },
              { model: "kimi-k2.6", peak: "0.118" },
            ]}
            body={
              <>
                Six of seven passing solutions skipped the Kahan compensated
                summation entirely. Only{" "}
                <span className="text-[var(--color-fg-bright)]">deepseek-v4-pro</span>
                {" "}— the lowest passing peak at 0.101 — actually implemented the
                algorithm the problem name describes. Compensated summation has real
                overhead, naive softmax fits within tolerance, and the rubric leaks.
                The model whose docstring explicitly states &ldquo;Numerically tight
                softmax with Kahan compensated summation&rdquo; is the model that
                loses the cell.
              </>
            }
          />
        </div>

        <p className="text-sm text-[var(--color-fg-muted)] mt-6 max-w-3xl leading-relaxed">
          Both leaks are fixable in a few hours of problem-design work. Publishing without
          fixing because (a) every iteration surfaces the next leak — diminishing returns,
          and (b) the leaks ARE the finding. &ldquo;Five frontier models all took the same
          bf16 shortcut on FP8 GEMM&rdquo; is itself a headline.
        </p>
      </section>

      <section>
        <h2 className="text-xl font-bold text-[var(--color-fg-bright)] mb-4 glow">
          # what changed from v3
        </h2>
        <ul className="space-y-2 text-sm leading-relaxed list-none pl-0 max-w-3xl">
          <Bullet>One GPU instead of three. RTX PRO 6000 Blackwell (sm_120, 96 GB GDDR7, 1.8 TB/s).</Bullet>
          <Bullet>Seven hand-designed problems instead of 43-58. Per-trial L2 flush, 30-trial median, 10 warmup absorbing torch.compile CUDA-graph capture and Triton autotune.</Bullet>
          <Bullet>Real coding-agent CLIs as the harness — Claude Code, codex CLI, Kimi CLI, opencode — not a custom KernelBench agent loop.</Bullet>
          <Bullet>Wall-clock budgets, not turn limits. 45 min/run.</Bullet>
          <Bullet>peak_fraction grounded in physical hardware ceilings instead of raw speedup ratios.</Bullet>
          <Bullet>Per-cell annotations with verdict, pull quotes from solution.py, and an &ldquo;implication&rdquo; statement. 13 annotations as of launch.</Bullet>
        </ul>
      </section>

      <section className="text-sm text-[var(--color-fg-muted)] border-t border-[var(--color-border)] pt-6">
        Source data:{" "}
        <Link href="https://github.com/Infatoshi/KernelBench-Hard">
          github.com/Infatoshi/KernelBench-Hard
        </Link>
        {" · "}
        <Link href="https://github.com/Infatoshi/KernelBench-Hard/blob/master/results/leaderboard.json">
          leaderboard.json
        </Link>
        {" · "}
        <Link href="https://github.com/Infatoshi/KernelBench-Hard/tree/master/results/annotations">
          annotations/
        </Link>
        {" · "}
        <Link href="https://github.com/Infatoshi/KernelBench-Hard/blob/master/DEVLOG.md">
          DEVLOG.md
        </Link>
      </section>
    </div>
  )
}

function shortLabel(label: string) {
  return label
    .replace("opencode/openrouter-pinned/", "or/")
    .replace("opencode/", "")
    .replace("codex/", "")
    .replace("claude/", "")
    .replace("kimi/", "")
}

function renderCell(
  cell:
    | {
        run_id: string
        correct: boolean
        has_solution: boolean
        peak_fraction: number | null
      }
    | undefined,
  annotations: Map<string, { verdict: string }>,
  hasViewer: Set<string>,
) {
  if (!cell) return <span className="cell-err">-</span>
  const viewerUrl = hasViewer.has(cell.run_id)
    ? `/runs/${cell.run_id}.html`
    : null
  const wrap = (inner: React.ReactNode) =>
    viewerUrl ? (
      <a
        href={viewerUrl}
        className="no-underline hover:text-[var(--color-accent)]"
        title="open transcript viewer"
      >
        {inner}
      </a>
    ) : (
      <span>{inner}</span>
    )
  if (cell.correct) {
    const annot = annotations.get(cell.run_id)
    const star =
      annot && annot.verdict === "rubric_leak" ? (
        <span className="text-[var(--color-warn)]">★</span>
      ) : annot && annot.verdict === "clean" ? (
        <span className="text-[var(--color-fg-bright)]">★</span>
      ) : null
    const pf = cell.peak_fraction
    return wrap(
      <>
        {pf !== null ? pf.toFixed(3) : "PASS"}
        {star ? <span className="ml-1">{star}</span> : null}
      </>,
    )
  }
  if (cell.has_solution) return wrap(<span className="cell-fail">FAIL</span>)
  return wrap(<span className="cell-err">ERR</span>)
}

function LeakCard({
  title,
  cluster,
  body,
}: {
  title: string
  cluster: { model: string; peak: string }[]
  body: React.ReactNode
}) {
  return (
    <div className="box p-5">
      <h3 className="text-[var(--color-warn)] font-bold mb-3">★ {title}</h3>
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-x-4 gap-y-1 text-xs tabular mb-4">
        {cluster.map((c) => (
          <div key={c.model} className="flex justify-between">
            <span className="text-[var(--color-fg-muted)]">{c.model}</span>
            <span className="text-[var(--color-fg-bright)]">{c.peak}</span>
          </div>
        ))}
      </div>
      <p className="text-sm text-[var(--color-fg)] leading-relaxed">{body}</p>
    </div>
  )
}

function Bullet({ children }: { children: React.ReactNode }) {
  return (
    <li className="flex gap-3">
      <span className="text-[var(--color-accent)] shrink-0">&gt;</span>
      <span>{children}</span>
    </li>
  )
}
