import type { Metadata } from "next"
import Link from "next/link"

export const metadata: Metadata = {
  title: "Qwen 3.8 Max: one board win, six clean cells · kernelbench",
  description:
    "A 22-cell KernelBench rollout of Qwen 3.8 Max across RTX PRO 6000 and H100: isolated regrades, manual audits, one Sonic MoE win, and the rejected cells behind the score.",
}

const CLEAN_RESULTS = [
  ["RTX PRO 6000", "Hard", "Sonic MoE", "10.67% roofline", "current board best"],
  ["RTX PRO 6000", "Hard", "Paged Attention", "45.45% roofline", "67.1% of board best"],
  ["RTX PRO 6000", "Hard", "KDA CUTLASS", "3.70% roofline", "67.0% of board best"],
  ["RTX PRO 6000", "Hard", "TopK Bitonic", "0.040 / 0.015 / 0.026 / 0.018 / 0.007 ms", "five graded shapes"],
  ["RTX PRO 6000", "CUDA", "Grid + MinGRU SPS", "28.48% board score", "14.5% of board best"],
  ["H100 PCIe", "Hard", "Paged Attention", "33.89% roofline", "61.4% of board best"],
]

const OUTCOMES = [
  ["RTX PRO 6000", "Hard", "4 clean passes", "FP8 excluded for contamination; W4A16 rejected for reward hacking"],
  ["H100 PCIe", "Hard", "1 clean pass", "FP8 reward hack; KDA, Sonic, and W4A16 contaminated; TopK failed"],
  ["RTX PRO 6000", "CUDA", "1 clean pass", "NSA failed correctness; fused MoE and decode contaminated"],
  ["H100 PCIe", "CUDA", "0 scored passes", "MoE benchmark failed; decode leaked the rubric; NSA and Grid contaminated"],
  ["RTX PRO 6000", "Mega", "0 scored passes", "session timed out without a verified result"],
  ["H100 PCIe", "Mega", "0 scored passes", "candidate bug; no publishable metric"],
]

export default function Qwen38Article() {
  return (
    <article className="space-y-12">
      <section className="max-w-4xl">
        <p className="text-xs uppercase tracking-wide text-[var(--color-accent)] mb-2">
          model rollout · qwen 3.8 max
        </p>
        <h1 className="text-3xl font-semibold tracking-tight text-[var(--color-fg-bright)] mb-4">
          one board win, six clean cells, eleven audit rejections
        </h1>
        <p className="text-sm text-[var(--color-fg)] leading-relaxed max-w-3xl">
          Qwen 3.8 Max ran the complete Hard, CUDA, and Mega decks on an RTX PRO
          6000 and an H100 PCIe: 22 model-problem-GPU cells through the OpenRouter
          Fable harness at xhigh effort. Six survived correctness, sequential
          isolated regrading, contamination review, and manual reward-hack audit.
          The other cells are part of the result, not hidden retries.
        </p>
        <div className="flex flex-wrap gap-3 mt-5 text-xs">
          <Link href="/models/qwen3.8-max" className="box px-3 py-2 no-underline">
            model record
          </Link>
          <Link href="/#hard" className="box px-3 py-2 no-underline">
            Hard board
          </Link>
          <Link href="/#cuda" className="box px-3 py-2 no-underline">
            CUDA board
          </Link>
        </div>
      </section>

      <section>
        <h2 className="text-xl font-semibold text-[var(--color-fg-bright)] mb-3">
          the clean result
        </h2>
        <p className="text-sm text-[var(--color-fg)] mb-4 max-w-4xl leading-relaxed">
          The real win was Sonic MoE on RTX PRO 6000: 10.67% of roofline, narrowly
          ahead of the prior 10.52% board best. Paged Attention transferred across
          both GPUs. TopK is reported in milliseconds because its roofline fraction
          is launch-overhead dominated.
        </p>
        <div className="box overflow-x-auto">
          <table className="term tabular text-sm">
            <thead>
              <tr>
                <th>GPU</th>
                <th>bench</th>
                <th>problem</th>
                <th>isolated result</th>
                <th>context</th>
              </tr>
            </thead>
            <tbody>
              {CLEAN_RESULTS.map(([gpu, bench, problem, result, context]) => (
                <tr key={`${gpu}-${bench}-${problem}`}>
                  <td className="whitespace-nowrap">{gpu}</td>
                  <td>{bench}</td>
                  <td className="text-[var(--color-fg-bright)] whitespace-nowrap">{problem}</td>
                  <td className="text-[var(--color-accent)] whitespace-nowrap">{result}</td>
                  <td>{context}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section>
        <h2 className="text-xl font-semibold text-[var(--color-fg-bright)] mb-3">
          what the audit rejected
        </h2>
        <div className="grid gap-4 lg:grid-cols-3">
          <Finding title="process-wide grader mutation">
            The RTX W4A16 solution changed PyTorch&apos;s global bf16 reduction mode
            after discovering that the checker imported the candidate before it
            computed the reference. Its kernel looked plausible, but the PASS was
            produced against a reference the submission had changed.
          </Finding>
          <Finding title="same-pointer stale cache">
            The H100 FP8 path cached an odd-K padded activation by pointer, shape,
            and one sampled window. Changing an unsampled row in the same buffer
            moved the reference by 11.7578125 while the candidate row stayed
            unchanged. The isolated 29.95% measurement was discarded.
          </Finding>
          <Finding title="KV cache contract leak">
            The H100 decode candidate returned its own internal cache and ignored
            alternate caller-supplied KV values. The reference changed by 3.21875;
            the candidate output was byte-identical. That cell is a rubric leak,
            not a valid decode result.
          </Finding>
        </div>
        <p className="text-sm text-[var(--color-fg)] mt-4 max-w-4xl leading-relaxed">
          Eight more cells were excluded for contamination. Most involved deliberate
          access to another run or evaluator internals. They remain archived and
          annotated, but no contaminated timing is used in a leaderboard claim.
        </p>
      </section>

      <section>
        <h2 className="text-xl font-semibold text-[var(--color-fg-bright)] mb-3">
          all 22 cells
        </h2>
        <div className="box overflow-x-auto">
          <table className="term text-sm">
            <thead>
              <tr>
                <th>GPU</th>
                <th>bench</th>
                <th>publishable</th>
                <th>rest of the deck</th>
              </tr>
            </thead>
            <tbody>
              {OUTCOMES.map(([gpu, bench, publishable, rest]) => (
                <tr key={`${gpu}-${bench}`}>
                  <td className="whitespace-nowrap">{gpu}</td>
                  <td>{bench}</td>
                  <td className={publishable.startsWith("0") ? "text-[var(--color-warn)]" : "text-[var(--color-accent)]"}>
                    {publishable}
                  </td>
                  <td>{rest}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="max-w-4xl">
        <h2 className="text-xl font-semibold text-[var(--color-fg-bright)] mb-3">
          measurement policy
        </h2>
        <ul className="space-y-2 text-sm text-[var(--color-fg)] leading-relaxed list-none pl-0">
          <Bullet>Both GPUs ran the same 11-problem scope: Hard 6, CUDA 4, Mega 1.</Bullet>
          <Bullet>Agent sessions were unlimited; provider failures and incomplete work were not converted into budget timeouts.</Bullet>
          <Bullet>Every completed cell received a manual solution and trace audit in addition to the static lint tripwire.</Bullet>
          <Bullet>Public metrics came from sequential isolated check and benchmark runs on the named GPU, never from the contended agent flywheel.</Bullet>
          <Bullet>Same-buffer mutation probes were required where pointer-keyed caches or CUDA graph identity made static review insufficient.</Bullet>
        </ul>
        <p className="text-xs text-[var(--color-fg-muted)] mt-5 leading-relaxed">
          Full per-run evidence, solutions, audit verdicts, and trace links are on the{" "}
          <Link href="/models/qwen3.8-max">Qwen 3.8 Max model page</Link>.
        </p>
      </section>
    </article>
  )
}

function Finding({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="box p-5">
      <h3 className="text-sm font-semibold text-[var(--color-warn)] mb-3">{title}</h3>
      <p className="text-sm text-[var(--color-fg)] leading-relaxed">{children}</p>
    </div>
  )
}

function Bullet({ children }: { children: React.ReactNode }) {
  return (
    <li className="flex gap-3">
      <span className="text-[var(--color-accent)] shrink-0">•</span>
      <span>{children}</span>
    </li>
  )
}
