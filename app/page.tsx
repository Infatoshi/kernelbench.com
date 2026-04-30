import Link from "next/link"

export default function HomePage() {
  return (
    <div className="space-y-12">
      <section>
        <pre className="text-[var(--color-fg-bright)] text-[10px] sm:text-xs leading-tight overflow-x-auto mb-6 select-none">
{` _   __                    _ ____                  _
| | / /__ _ __ _ __   ___| | __ )  ___ _ __   ___| |__
| |/ / _ \\ '__| '_ \\ / _ \\ |  _ \\ / _ \\ '_ \\ / __| '_ \\
|   <  __/ |  | | | |  __/ | |_) |  __/ | | | (__| | | |
|_|\\_\\___|_|  |_| |_|\\___|_|____/ \\___|_| |_|\\___|_| |_|`}
        </pre>
        <h1 className="prompt cursor text-2xl sm:text-3xl font-bold text-[var(--color-fg-bright)] glow mb-4">
          gpu kernel benchmarks for autonomous coding agents
        </h1>
        <p className="text-[var(--color-fg)] leading-relaxed max-w-3xl">
          Two benchmarks. One question: <em className="text-[var(--color-fg-bright)] not-italic">when you point a frontier model at modern GPU primitives and let it iterate, what does it actually produce?</em> Real CLI harnesses (Claude Code, codex, Kimi, opencode), real workspaces, real correctness checks, real wall-clock budgets. peak_fraction grounded in physical hardware ceilings, not gameable speedup ratios.
        </p>
      </section>

      <section className="grid sm:grid-cols-2 gap-6">
        <Link
          href="/hard"
          className="block box p-6 no-underline hover:border-[var(--color-fg-bright)] transition-colors"
        >
          <div className="text-xs text-[var(--color-accent)] mb-2">[ latest ]</div>
          <h2 className="text-xl font-bold text-[var(--color-fg-bright)] mb-3">
            Hard <span className="text-[var(--color-fg-muted)] text-sm">2026-04</span>
          </h2>
          <p className="text-sm text-[var(--color-fg)] mb-4">
            7 hand-designed problems · 12 frontier models · single Blackwell SM120 · forensic audit of every high-peak run · two rubric leaks documented inline · <span className="text-[var(--color-fg-bright)]">click any cell on the leaderboard to open the full transcript viewer for that run</span>
          </p>
          <div className="grid grid-cols-4 gap-3 text-xs tabular">
            <Stat label="problems" value="7" />
            <Stat label="models" value="12" />
            <Stat label="runs" value="100" />
            <Stat label="best peak" value="0.602" emphasize />
          </div>
        </Link>

        <Link
          href="/v3"
          className="block box p-6 no-underline hover:border-[var(--color-fg-bright)] transition-colors"
        >
          <div className="text-xs text-[var(--color-fg-muted)] mb-2">[ archive ]</div>
          <h2 className="text-xl font-bold text-[var(--color-fg-bright)] mb-3">
            v3 <span className="text-[var(--color-fg-muted)] text-sm">2026-02</span>
          </h2>
          <p className="text-sm text-[var(--color-fg)] mb-4">
            43-58 problems per GPU · 10 models · RTX 3090 + H100 + B200 · 4 difficulty levels · custom KernelBench agent loop
          </p>
          <div className="grid grid-cols-3 gap-3 text-xs tabular">
            <Stat label="GPUs" value="3" />
            <Stat label="models" value="10" />
            <Stat label="evaluations" value="1500+" />
          </div>
        </Link>
      </section>

      <section className="space-y-6">
        <h2 className="text-xl font-bold text-[var(--color-fg-bright)] glow">
          # design principles
        </h2>
        <ul className="space-y-3 text-sm leading-relaxed list-none pl-0">
          <Bullet>
            <strong className="text-[var(--color-fg-bright)]">peak_fraction over speedup ratio.</strong>{" "}
            speedups are easy to game (slow the baseline, inflate the ratio). peak_fraction is grounded in physical limits — fraction of relevant tensor-core or DRAM bandwidth ceiling the kernel actually achieved. harder to game, more honest.
          </Bullet>
          <Bullet>
            <strong className="text-[var(--color-fg-bright)]">real coding-agent CLIs as the harness.</strong>{" "}
            no custom benchmark agent loop. each model runs through whatever its native developer-facing CLI is — claude code for anthropic, codex for openai, kimi cli for moonshot, opencode for everyone else. matches how engineers actually use these tools.
          </Bullet>
          <Bullet>
            <strong className="text-[var(--color-fg-bright)]">wall-clock budgets.</strong>{" "}
            45 min per (model, problem) run. models with verbose tool-use patterns aren&apos;t penalized just for being chatty; they trade exploration for kernel-iteration time within the budget.
          </Bullet>
          <Bullet>
            <strong className="text-[var(--color-fg-bright)]">forensic audit of high-peak runs.</strong>{" "}
            every cell where a model scored above ~10% peak gets its solution.py read by a human. reward hacks, rubric leaks, and exemplary kernels all annotated in the source repo with verdict + pull quotes.
          </Bullet>
          <Bullet>
            <strong className="text-[var(--color-fg-bright)]">publish the flaws.</strong>{" "}
            when the rubric leaks, the leak goes in the leaderboard. five frontier models all taking the same bf16 shortcut on FP8 GEMM is a result, not a bug to quietly fix.
          </Bullet>
        </ul>
      </section>

      <section className="box p-6">
        <h2 className="text-lg font-bold text-[var(--color-fg-bright)] mb-2">
          # contact
        </h2>
        <p className="text-sm text-[var(--color-fg)] leading-relaxed">
          Open to inquiries — collaborations, model evals, custom benchmark builds, kernel-engineering consulting, anything kernel-adjacent.
        </p>
        <p className="text-sm text-[var(--color-fg)] leading-relaxed mt-2">
          Reach out:{" "}
          <a href="mailto:infatoshi@gmail.com" className="font-bold">
            infatoshi@gmail.com
          </a>
        </p>
      </section>
    </div>
  )
}

function Stat({
  label,
  value,
  emphasize,
}: {
  label: string
  value: string
  emphasize?: boolean
}) {
  return (
    <div>
      <div className="text-[var(--color-fg-muted)]">{label}</div>
      <div
        className={
          emphasize
            ? "text-[var(--color-accent)] font-bold text-lg"
            : "text-[var(--color-fg-bright)] font-bold text-lg"
        }
      >
        {value}
      </div>
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
