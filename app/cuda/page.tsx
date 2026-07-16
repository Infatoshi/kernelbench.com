import Link from "next/link"
import { LeaderboardTable, type HardRunRecord } from "../hard/leaderboard-table"

// KernelBench-CUDA: CUDA-only writing deck. Triton/DSL fail the language gate.
// Hard/Mega stay frozen. Four problems — no published runs yet.

const REPO_TREE =
  "https://github.com/Infatoshi/kernelbench.com/blob/master/benchmarks/cuda"

const REPO_DIR =
  "https://github.com/Infatoshi/kernelbench.com/tree/master/benchmarks/cuda"

const CUDA_PROBLEMS = [
  {
    key: "01_glm52_fused_moe",
    label: "GLM-5.2 Fused MoE",
    blurb:
      "256 routed + 1 shared, top-8, fused silu_and_mul; serving-shape geomean incl. T=4127",
  },
  {
    key: "02_deepseek_nsa",
    label: "DeepSeek NSA",
    blurb:
      "Block importance → top-n select ∪ sliding window → sparse causal attention",
  },
  {
    key: "03_megaqwen_decode",
    label: "MegaQwen Decode",
    blurb:
      "Qwen3-0.6B geometry; prefill untimed; decode-only at 2k/8k/32k/128k; improve MegaQwen",
  },
  {
    key: "04_grid_mingru_sps",
    label: "Grid + MinGRU SPS",
    blurb: "RL sim env + 3×MinGRU(h=256); fusion optional; SPS metric",
  },
] as const

const NO_RUN = { label: "no run", tone: "muted" as const }

function placeholderRow(problem: {
  key: string
  label: string
}): HardRunRecord {
  return {
    key: `${problem.key}:pending`,
    runId: null,
    model: "—",
    harness: "—",
    gpu: "RTX PRO 6000",
    problem: problem.label,
    problemKey: problem.key,
    date: null,
    time: null,
    compiled: NO_RUN,
    correct: NO_RUN,
    auditFlags: [],
    explanation: null,
    peakFraction: null,
    speedPct: null,
    isWinner: false,
    referenceUrl: `${REPO_TREE}/problems-rtxpro6000/${problem.key}/reference.py`,
    solutionUrl: null,
    transcriptUrl: null,
    scored: "0/0",
    note: "awaiting first run",
    title: "no run yet",
    searchText: `${problem.label} ${problem.key} no run`,
  }
}

export default function CudaPage() {
  const rows = CUDA_PROBLEMS.map(placeholderRow)

  return (
    <div className="hard-page space-y-12">
      <section>
        <p className="text-xs text-[var(--color-fg-muted)] mb-2">
          <Link href="/" className="underline underline-offset-2">
            home
          </Link>
          {" · "}
          <Link href="/hard" className="underline underline-offset-2">
            hard
          </Link>
          {" · "}
          <Link href="/mega" className="underline underline-offset-2">
            mega
          </Link>
          {" · "}
          <span className="text-[var(--color-fg)]">cuda</span>
        </p>
        <h1 className="text-3xl font-semibold tracking-tight text-[var(--color-fg-bright)] mb-3">
          cuda
        </h1>
        <p className="text-sm text-[var(--color-fg)] mb-2">
          RTX PRO 6000 Blackwell (SM120 · GDDR7 · 1.8 TB/s)
          <span className="ml-2 text-xs font-semibold text-[var(--color-accent)]">
            ● CUDA-only language gate · Triton / DSL = fail
          </span>
        </p>
        <p className="text-xs text-[var(--color-fg-muted)] mb-4 max-w-4xl leading-relaxed">
          Isolated CUDA-writing deck. Hard and Mega stay frozen for labs. Four
          hard cells only: GLM-5.2 fused MoE (serving hot path), DeepSeek NSA
          (Chinese sparse attention), MegaQwen improve-baseline decode (tok/s at
          2k–128k), and RL grid+MinGRU SPS. Shape sweeps use Hard-style
          geomean with misalignment tails. Language gate requires real CUDA C++
          / PTX; pure PyTorch without a kernel fails.
        </p>
        <div className="flex flex-wrap gap-3 text-xs text-[var(--color-fg-muted)] mb-6">
          <span className="rounded border border-[var(--color-border)] px-2 py-1">
            4 problems
          </span>
          <span className="rounded border border-[var(--color-border)] px-2 py-1">
            framework + triton_cheat sidecars
          </span>
          <span className="rounded border border-[var(--color-border)] px-2 py-1 text-[var(--color-accent)]">
            coming soon · no published cells yet
          </span>
          <a
            href={REPO_DIR}
            className="rounded border border-[var(--color-border)] px-2 py-1 underline underline-offset-2"
          >
            benchmarks/cuda
          </a>
        </div>
      </section>

      <section>
        <h2 className="text-sm font-semibold text-[var(--color-fg-bright)] mb-3 uppercase tracking-wide">
          deck
        </h2>
        <ul className="text-sm text-[var(--color-fg)] space-y-3 max-w-3xl">
          {CUDA_PROBLEMS.map((p, i) => (
            <li key={p.key}>
              <span className="text-[var(--color-accent)] font-mono text-xs mr-2">
                {String(i + 1).padStart(2, "0")}
              </span>
              <span className="font-medium">{p.label}</span>
              <span className="text-[var(--color-fg-muted)]">
                {" — "}
                {p.blurb}
              </span>
              {" "}
              <a
                href={`${REPO_TREE}/problems-rtxpro6000/${p.key}/reference.py`}
                className="text-xs underline underline-offset-2 text-[var(--color-fg-muted)]"
              >
                reference
              </a>
            </li>
          ))}
        </ul>
      </section>

      <section>
        <h2 className="text-sm font-semibold text-[var(--color-fg-bright)] mb-3 uppercase tracking-wide">
          methodology
        </h2>
        <ul className="text-xs text-[var(--color-fg-muted)] space-y-2 max-w-3xl leading-relaxed list-disc pl-5">
          <li>
            Language gate: Triton, CuteDSL, TileLang, and pure torch without
            CUDA evidence hard-fail. Sidecars:{" "}
            <code className="text-[var(--color-fg)]">cuda_language.json</code>,{" "}
            <code className="text-[var(--color-fg)]">framework.txt</code>.
          </li>
          <li>
            Shape score = geometric mean across the sweep (aligned + misaligned
            + decode microbatch), same idea as Hard FP8{" "}
            <code className="text-[var(--color-fg)]">K=4127</code>.
          </li>
          <li>
            MegaQwen: prefill builds KV (untimed); only decode tok/s at 2k / 8k
            / 32k / 128k is graded. Numeric{" "}
            <code className="text-[var(--color-fg)]">last_hidden</code> match —
            no tokenizer.
          </li>
          <li>
            Sibling harness to{" "}
            <Link href="/hard" className="underline underline-offset-2">
              hard
            </Link>
            ; does not change Hard or Mega prompts.
          </li>
        </ul>
      </section>

      <section>
        <h2 className="text-sm font-semibold text-[var(--color-fg-bright)] mb-3 uppercase tracking-wide">
          leaderboard
        </h2>
        <p className="text-xs text-[var(--color-fg-muted)] mb-4 max-w-3xl">
          Placeholder rows until the first audited sweep lands. Run archives
          will fill cells the same way as{" "}
          <Link href="/hard" className="underline underline-offset-2">
            /hard
          </Link>
          .
        </p>
        <LeaderboardTable rows={rows} />
      </section>
    </div>
  )
}
