import Link from "next/link"
import { loadBaselines, loadLeaderboard } from "@/app/_lib/data"

const PROBLEMS = [
  { key: "01_fp8_gemm", label: "FP8 GEMM" },
  { key: "02_kda_cutlass", label: "KDA CUTLASS" },
  { key: "03_paged_attention", label: "Paged Attention" },
  { key: "05_topk_bitonic", label: "TopK Bitonic" },
  { key: "06_sonic_moe_swiglu", label: "Sonic MoE SwiGLU" },
  { key: "07_w4a16_gemm", label: "W4A16 GEMM" },
]

const FP8_CONSTRAINT_FIXED_RUNS = [
  ["Claude Opus 4.6", "claude", "FAIL", "18.5m", "large_input stress failed, max_abs_diff=4"],
  ["Claude Opus 4.7", "claude", "FAIL", "21.6m", "check_failed under real FP8 constraint"],
  ["Claude Opus 4.8", "claude", "FAIL", "41.8m", "large_input K=4127 failed, max_abs_diff=4"],
  ["GPT-5.5", "codex", "FAIL", "6.8m", "nominal tolerance failed on first fixed run"],
  ["DeepSeek V4 Flash", "opencode", "FAIL", "4.1m", "nominal tolerance failed, max_abs_diff around 0.53"],
  ["DeepSeek V4 Pro", "opencode", "FAIL", "5.9m", "first run had Triton fp8 load cast error"],
  ["OpenCode GLM-5.1", "opencode", "EARLY", "11.5m", "provider early-stop/no solution on opencode route"],
  ["Kimi K2.6", "kimi", "ERR", "4s", "invalid or expired API key"],
  ["MiniMax/Qwen/MiMo via OpenRouter", "opencode", "ERR", "1-2s", "provider_insufficient_credits"],
]

const FP8_RECOVERY_RUNS = [
  ["GLM-5.1", "zai-claude", "FAIL", "11.0m", "direct ZAI route worked, nominal max_abs_diff=0.5625"],
  ["DeepSeek V4 Pro", "opencode", "FAIL", "9.8m", "second attempt reached verifier, nominal max_abs_diff=0.539"],
  ["DeepSeek V4 Flash", "opencode", "FAIL", "3.2m", "second attempt reached verifier, nominal max_abs_diff=0.539"],
  ["GPT-5.5", "codex", "FAIL", "8.1m", "Triton resource failure: 147456B shared memory > 101376B limit"],
]

export default async function HardBlogPage() {
  const [lb, baselines] = await Promise.all([loadLeaderboard(), loadBaselines()])

  return (
    <article className="space-y-12">
      <section className="max-w-3xl">
        <p className="text-xs uppercase tracking-wide text-[var(--color-fg-muted)] mb-2">
          hard notes
        </p>
        <h1 className="text-3xl font-semibold tracking-tight text-[var(--color-fg-bright)] mb-4">
          hard benchmark design notes
        </h1>
        <p className="text-sm text-[var(--color-fg)] leading-relaxed">
          Background notes for the Hard leaderboard: problem ceilings, known
          caveats, and the FP8 constraint rerun. The operational leaderboard
          lives on <Link href="/hard">/hard</Link>.
        </p>
      </section>

      <section>
        <h2 className="text-xl font-semibold text-[var(--color-fg-bright)] mb-3">
          per-problem ceilings
        </h2>
        <p className="text-sm text-[var(--color-fg)] mb-3 max-w-4xl leading-relaxed">
          eager / compiled = PyTorch reference timings. SOTA = the existing best-known kernel
          for the problem when one exists on this hardware. best peak = the model
          that pushed furthest above the reference line.
        </p>
        <div className="box overflow-x-auto">
          <table className="term tabular text-sm">
            <thead>
              <tr>
                <th>problem</th>
                <th className="text-right">eager ms</th>
                <th className="text-right">compiled ms</th>
                <th className="text-right">SOTA ms</th>
                <th className="text-right">best peak</th>
                <th>best model</th>
                <th className="text-right">n scored</th>
              </tr>
            </thead>
            <tbody>
              {PROBLEMS.map((p) => {
                const pp = lb.per_problem[p.key]
                const bl = baselines?.problems[p.key] ?? {}
                return (
                  <tr key={p.key}>
                    <td>{p.label}</td>
                    <td className="text-right">{fmtMs(bl.eager)}</td>
                    <td className="text-right">{fmtMs(bl.compiled)}</td>
                    <td className="text-right">{fmtMs(bl.sota)}</td>
                    <td className="text-right text-[var(--color-fg-bright)]">
                      {pp.best_peak_fraction ? pp.best_peak_fraction.toFixed(3) : <span className="cell-missing">-</span>}
                    </td>
                    <td>
                      {pp.best_model ? <span className="table-value">{shortLabel(pp.best_model)}</span> : <span className="cell-missing">-</span>}
                    </td>
                    <td className="text-right">
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
        <h2 className="text-xl font-semibold text-[var(--color-fg-bright)] mb-3">
          rubric caveat
        </h2>
        <div className="box p-5 max-w-4xl">
          <h3 className="text-[var(--color-warn)] font-semibold mb-3">
            01 fp8_gemm: bf16 dressup
          </h3>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-x-4 gap-y-1 text-xs tabular mb-4">
            {[
              ["claude-opus-4-7 [max]", "0.534"],
              ["mimo-v2.5-pro", "0.434"],
              ["qwen3.6-plus", "0.431"],
              ["qwen3.6-max-preview", "0.429"],
              ["gpt-5.5 [xhigh]", "0.423"],
            ].map(([model, peak]) => (
              <div key={model} className="flex justify-between gap-3">
                <span className="text-[var(--color-fg)]">{model}</span>
                <span className="text-[var(--color-fg-bright)]">{peak}</span>
              </div>
            ))}
          </div>
          <p className="text-sm text-[var(--color-fg)] leading-relaxed">
            Every passing solution at peak {">="} 0.4 casts the fp8 inputs to bf16
            inside the kernel and runs a bf16 GEMM. The peak fractions on this
            row reflect bf16 GEMM optimization quality on fp8-typed inputs, not
            FP8 tensor core skill.
          </p>
        </div>
      </section>

      <section>
        <h2 className="text-xl font-semibold text-[var(--color-fg-bright)] mb-3">
          fp8 constraint rerun
        </h2>
        <p className="text-sm text-[var(--color-fg)] mb-4 max-w-4xl leading-relaxed">
          On June 5, 2026, the FP8 GEMM verifier was tightened to reject the
          bf16-dressup shortcut and require an FP8-looking execution path. Once
          the shortcut was blocked, every available model either failed
          correctness, failed the provider path, or could not run because of
          credits/key issues.
        </p>
        <div className="grid gap-4 lg:grid-cols-2">
          <div className="box overflow-x-auto">
            <h3 className="text-sm font-semibold text-[var(--color-fg-bright)] mb-3">
              fixed-tolerance rerun
            </h3>
            <Fp8ConstraintTable rows={FP8_CONSTRAINT_FIXED_RUNS} />
          </div>
          <div className="box overflow-x-auto">
            <h3 className="text-sm font-semibold text-[var(--color-fg-bright)] mb-3">
              recovery smokes
            </h3>
            <Fp8ConstraintTable rows={FP8_RECOVERY_RUNS} />
          </div>
        </div>
        <div className="grid gap-4 mt-5 lg:grid-cols-3">
          <Figure
            src="/blog-hard/fp8-constraint-rerun/fp8_token_burn_stacked.png"
            alt="Stacked token burn for FP8 constraint rerun"
            caption="Token burn by model on the FP8 constraint run."
          />
          <Figure
            src="/blog-hard/fp8-constraint-rerun/fp8_tokens_vs_effective_peak.png"
            alt="Tokens versus effective peak for FP8 constraint rerun"
            caption="All effective peaks collapse to zero under the strict verifier."
          />
          <Figure
            src="/blog-hard/fp8-constraint-rerun/fp8_cost_before_outcome.png"
            alt="Cost before outcome for FP8 constraint rerun"
            caption="Spend and wall time before each failing outcome."
          />
        </div>
      </section>

      <section>
        <h2 className="text-xl font-semibold text-[var(--color-fg-bright)] mb-3">
          design choices
        </h2>
        <ul className="space-y-2 text-sm leading-relaxed list-none pl-0 max-w-3xl">
          <Bullet>One primary GPU deck (RTX PRO 6000 Blackwell), plus H100/B200 boards with the same problems.</Bullet>
          <Bullet>A small hand-designed problem deck, not dozens of ops per GPU.</Bullet>
          <Bullet>Real coding-agent CLIs as the harness: claude code, codex, opencode, droid, kimi, cursor, gemini-cli, grok build.</Bullet>
          <Bullet>Unlimited agent sessions; peak_fraction grounded in physical hardware ceilings.</Bullet>
          <Bullet>Per-cell annotations with verdict, quotes from solution.py, and implication notes.</Bullet>
        </ul>
      </section>
    </article>
  )
}

function Fp8ConstraintTable({ rows }: { rows: string[][] }) {
  return (
    <table className="term tabular text-xs">
      <thead>
        <tr>
          <th>model</th>
          <th>route</th>
          <th>outcome</th>
          <th className="text-right">elapsed</th>
          <th>note</th>
        </tr>
      </thead>
      <tbody>
        {rows.map(([model, route, result, elapsed, note]) => (
          <tr key={`${model}-${route}`}>
            <td className="text-[var(--color-fg-bright)] whitespace-nowrap">{model}</td>
            <td className="text-[var(--color-fg-muted)] whitespace-nowrap">{route}</td>
            <td>
              <span className={result === "FAIL" ? "cell-fail" : "cell-err"}>{result}</span>
            </td>
            <td className="text-right text-[var(--color-fg-muted)] whitespace-nowrap">{elapsed}</td>
            <td className="min-w-64">{note}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function Figure({
  src,
  alt,
  caption,
}: {
  src: string
  alt: string
  caption: string
}) {
  return (
    <figure className="box">
      <img src={src} alt={alt} className="w-full border border-[var(--color-border)]" />
      <figcaption className="text-xs text-[var(--color-fg-muted)] mt-2 leading-relaxed">
        {caption}
      </figcaption>
    </figure>
  )
}

function Bullet({ children }: { children: React.ReactNode }) {
  return (
    <li className="flex gap-3">
      <span className="text-[var(--color-fg)] shrink-0">•</span>
      <span>{children}</span>
    </li>
  )
}

function fmtMs(t: { ms: number } | undefined) {
  return t ? <span className="table-value">{t.ms.toFixed(3)}</span> : <span className="cell-missing">-</span>
}

function shortLabel(label: string) {
  return label
    .replace("codex/gpt-5.5 [2026-05-28 finish xhigh]", "GPT-5.5")
    .replace("claude/claude-opus-4-7 [2026-05-28 finish max]", "Claude Opus 4.7")
    .replace("claude/claude-opus-4-8 [2026-05-28 opus48-grok max]", "Claude Opus 4.8")
    .replace("claude/claude-opus-4-6 [2026-06-04 opus46 max]", "Claude Opus 4.6")
    .replace("kimi/kimi-k2.6", "Kimi K2.6")
    .replace("opencode/deepseek/deepseek-v4-flash", "DeepSeek V4 Flash")
    .replace("opencode/deepseek/deepseek-v4-pro", "DeepSeek V4 Pro")
    .replace("opencode/", "")
}
