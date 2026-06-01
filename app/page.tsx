import Link from "next/link"

const HUGGING_FACE_LOGO =
  "https://huggingface.co/front/assets/huggingface_logo-noborder.svg"

const benchmarks = [
  {
    href: "/hard",
    title: "Hard",
    description:
      "Small hard CUDA deck, curated frontier-model comparison, single Blackwell SM120, clickable transcript viewers for scored runs.",
    stats: [
      ["problems", "8"],
      ["models", "13"],
      ["GPU", "SM120"],
    ],
    hfHref: "https://huggingface.co/datasets/Infatoshi/kernelbench-hard-runs",
  },
  {
    href: "/v3",
    title: "v3",
    description:
      "43-58 problems per GPU, 10 models, RTX 3090, H100, B200, four difficulty levels, and the original KernelBench agent loop.",
    stats: [
      ["problems", "43-58"],
      ["GPUs", "3"],
      ["models", "10"],
      ["evaluations", "1500+"],
    ],
    hfHref: "https://huggingface.co/datasets/Infatoshi/kernelbench-v3-runs",
  },
]

export default function HomePage() {
  return (
    <div className="space-y-12">
      <h1 className="sr-only">KernelBench benchmarks</h1>
      <section aria-label="Benchmarks" className="space-y-4">
        {benchmarks.map((benchmark) => (
          <article key={benchmark.href} className="benchmark-card">
            <Link
              href={benchmark.href}
              className="benchmark-main no-underline"
              aria-label={`Open KernelBench ${benchmark.title}`}
            >
              <div className="flex flex-col gap-5 sm:flex-row sm:items-start sm:justify-between">
                <div className="max-w-2xl">
                  <h2 className="text-2xl sm:text-3xl font-semibold tracking-tight text-[var(--color-fg-bright)]">
                    {benchmark.title}
                  </h2>
                  <p className="mt-3 text-sm sm:text-base text-[var(--color-fg)] leading-relaxed">
                    {benchmark.description}
                  </p>
                </div>
                <span className="benchmark-open" aria-hidden="true">
                  <svg
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="1.8"
                  >
                    <path d="M7 17 17 7" />
                    <path d="M9 7h8v8" />
                  </svg>
                </span>
              </div>
              <div className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-4">
                {benchmark.stats.map(([label, value]) => (
                  <Stat key={label} label={label} value={value} />
                ))}
              </div>
            </Link>
            <div className="benchmark-footer">
              <a
                href={benchmark.hfHref}
                target="_blank"
                rel="noreferrer"
                className="hf-link no-underline"
                aria-label={`Open KernelBench ${benchmark.title} runs on Hugging Face`}
              >
                <img
                  src={HUGGING_FACE_LOGO}
                  alt=""
                  width="22"
                  height="22"
                  loading="lazy"
                />
                <span>Hugging Face runs</span>
              </a>
            </div>
          </article>
        ))}
      </section>

      <section className="space-y-5">
        <h2 className="text-xl font-semibold text-[var(--color-fg-bright)]">
          Design principles
        </h2>
        <ul className="space-y-3 text-sm leading-relaxed list-none pl-0">
          <Bullet>
            <strong className="text-[var(--color-fg-bright)]">
              Percent of theoretical maximum, not speedup over PyTorch.
            </strong>{" "}
            Scores are grounded in hardware ceilings instead of baseline quirks.
          </Bullet>
          <Bullet>
            <strong className="text-[var(--color-fg-bright)]">
              Modern coding-agent harnesses.
            </strong>{" "}
            Runs use Claude Code, Codex CLI, Cursor, Gemini CLI, Kimi CLI,
            OpenCode, Grok, and MiniMax where those are the natural interfaces.
          </Bullet>
          <Bullet>
            <strong className="text-[var(--color-fg-bright)]">
              Public transcript viewers.
            </strong>{" "}
            Browse{" "}
            <Link href="/runs" className="underlined-link">
              the run index
            </Link>{" "}
            or open{" "}
            <Link
              href="/runs/20260601_124343_minimax-claude_MiniMax-M3_07_w4a16_gemm.html"
              className="underlined-link"
            >
              a scored Hard run
            </Link>{" "}
            to inspect tool calls, solution files, checks, timing, and costs.
          </Bullet>
          <Bullet>
            <strong className="text-[var(--color-fg-bright)]">
              Judge-assisted audit.
            </strong>{" "}
            A judge model helps flag reward hacking, rubric leaks, and suspicious
            shortcuts for human review.
          </Bullet>
        </ul>
      </section>

      <section className="box p-6">
        <h2 className="text-lg font-semibold text-[var(--color-fg-bright)] mb-2">
          Contact
        </h2>
        <p className="text-sm text-[var(--color-fg)] leading-relaxed">
          Open to inquiries: collaborations, model evals, custom benchmark
          builds, kernel-engineering consulting, anything kernel-adjacent.
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

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="stat-box">
      <div className="text-[var(--color-fg-muted)]">{label}</div>
      <div className="text-[var(--color-fg-bright)] font-semibold text-lg">
        {value}
      </div>
    </div>
  )
}

function Bullet({ children }: { children: React.ReactNode }) {
  return (
    <li className="flex gap-3">
      <span className="text-[var(--color-fg-muted)] shrink-0">•</span>
      <span>{children}</span>
    </li>
  )
}
