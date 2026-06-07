import Link from "next/link"

const HUGGING_FACE_LOGO =
  "https://huggingface.co/front/assets/huggingface_logo-noborder.svg"

const citationGraph = {
  "@context": "https://schema.org",
  "@graph": [
    {
      "@type": "WebSite",
      "@id": "https://kernelbench.com/#website",
      name: "kernelbench.com",
      url: "https://kernelbench.com",
      author: {
        "@type": "Person",
        name: "Elliot Arledge",
        url: "https://elliotarledge.com",
      },
      description:
        "Open agentic GPU kernel benchmark results, run transcripts, source repositories, and datasets.",
      citation: [
        "https://github.com/Infatoshi/kernelbench.com",
        "https://github.com/Infatoshi/KernelBench-v3",
        "https://github.com/Infatoshi/KernelBench-Hard",
        "https://huggingface.co/datasets/Infatoshi/kernelbench-v3-runs",
        "https://huggingface.co/datasets/Infatoshi/kernelbench-hard-runs",
      ],
    },
    {
      "@type": "Dataset",
      "@id": "https://huggingface.co/datasets/Infatoshi/kernelbench-hard-runs",
      name: "KernelBench-Hard run artifacts",
      url: "https://huggingface.co/datasets/Infatoshi/kernelbench-hard-runs",
      creator: { "@type": "Person", name: "Elliot Arledge" },
    },
    {
      "@type": "Dataset",
      "@id": "https://huggingface.co/datasets/Infatoshi/kernelbench-v3-runs",
      name: "KernelBench-v3 run artifacts",
      url: "https://huggingface.co/datasets/Infatoshi/kernelbench-v3-runs",
      creator: { "@type": "Person", name: "Elliot Arledge" },
    },
    {
      "@type": "SoftwareSourceCode",
      "@id": "https://github.com/Infatoshi/KernelBench-Hard",
      name: "Hard result suite repository",
      codeRepository: "https://github.com/Infatoshi/KernelBench-Hard",
      author: { "@type": "Person", name: "Elliot Arledge" },
    },
    {
      "@type": "SoftwareSourceCode",
      "@id": "https://github.com/Infatoshi/KernelBench-v3",
      name: "v3 result suite repository",
      codeRepository: "https://github.com/Infatoshi/KernelBench-v3",
      author: { "@type": "Person", name: "Elliot Arledge" },
    },
  ],
}

const benchmarks = [
  {
    href: "/hard",
    title: "Hard",
    description:
      "Small hard CUDA deck, curated frontier-model comparison, single Blackwell SM120, clickable transcript viewers for scored runs.",
    stats: [
      ["problems", "6"],
      ["models", "13"],
      ["GPU", "SM120"],
    ],
    hfHref: "https://huggingface.co/datasets/Infatoshi/kernelbench-hard-runs",
  },
  {
    href: "/v3",
    title: "v3",
    description:
      "43-58 problems per GPU, 10 models, RTX 3090, H100, B200, four difficulty levels, and a custom v3 agent loop.",
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
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(citationGraph) }}
      />
      <h1 className="sr-only">Agentic GPU kernel benchmark results</h1>
      <section aria-label="Benchmarks" className="space-y-4">
        {benchmarks.map((benchmark) => (
          <article key={benchmark.href} className="benchmark-card">
            <Link
              href={benchmark.href}
              className="benchmark-main no-underline"
              aria-label={`Open ${benchmark.title} benchmark`}
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
                aria-label={`Open ${benchmark.title} benchmark runs on Hugging Face`}
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

      <section id="cite" className="citation-box scroll-mt-24">
        <div>
          <p className="text-xs uppercase tracking-wide text-[var(--color-fg-muted)] mb-2">
            Citation
          </p>
          <h2 className="text-xl font-semibold text-[var(--color-fg-bright)]">
            Cite this benchmark suite
          </h2>
          <p className="mt-3 text-sm leading-relaxed text-[var(--color-fg)] max-w-3xl">
            If you use these results, cite the website for the public benchmark
            view, the relevant benchmark repository for problem definitions and
            harness code, and the Hugging Face dataset for the run transcripts.
            This is a results and artifact site, not Stanford&apos;s original
            KernelBench benchmark.
          </p>
        </div>

        <div className="citation-links">
          <ArtifactLink href="https://kernelbench.com" label="Website" />
          <ArtifactLink
            href="https://github.com/Infatoshi/kernelbench.com"
            label="Website repository"
          />
          <ArtifactLink
            href="https://github.com/Infatoshi/KernelBench-Hard"
            label="Hard repository"
          />
          <ArtifactLink
            href="https://github.com/Infatoshi/KernelBench-v3"
            label="v3 repository"
          />
          <ArtifactLink
            href="https://huggingface.co/datasets/Infatoshi/kernelbench-hard-runs"
            label="Hard HF dataset"
          />
          <ArtifactLink
            href="https://huggingface.co/datasets/Infatoshi/kernelbench-v3-runs"
            label="v3 HF dataset"
          />
        </div>

        <pre className="bibtex-block">
{`@misc{arledge2026kernelbenchcom,
  title        = {kernelbench.com: Agentic GPU Kernel Benchmark Results and Run Artifacts},
  author       = {Arledge, Elliot},
  year         = {2026},
  howpublished = {\\url{https://kernelbench.com}},
  note         = {Website, benchmark results, transcript viewers, and citation index}
}

@misc{arledge2026hard,
  title        = {Hard: Agentic CUDA Kernel Result Suite},
  author       = {Arledge, Elliot},
  year         = {2026},
  howpublished = {\\url{https://github.com/Infatoshi/KernelBench-Hard}},
  note         = {CUDA benchmark suite, harness, results, and annotations}
}

@misc{arledge2026v3,
  title        = {v3: Multi-GPU Agentic Kernel Result Suite},
  author       = {Arledge, Elliot},
  year         = {2026},
  howpublished = {\\url{https://github.com/Infatoshi/KernelBench-v3}},
  note         = {Multi-GPU benchmark suite, harness, and result artifacts}
}

@misc{arledge2026hardruns,
  title        = {KernelBench-Hard Run Artifacts},
  author       = {Arledge, Elliot},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\\url{https://huggingface.co/datasets/Infatoshi/kernelbench-hard-runs}},
  note         = {Run transcripts, solutions, checks, timing, and cost metadata}
}

@misc{arledge2026v3runs,
  title        = {KernelBench-v3 Run Artifacts},
  author       = {Arledge, Elliot},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\\url{https://huggingface.co/datasets/Infatoshi/kernelbench-v3-runs}},
  note         = {Run artifacts and benchmark result data}
}`}
        </pre>
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

function ArtifactLink({ href, label }: { href: string; label: string }) {
  return (
    <a href={href} target="_blank" rel="noreferrer" className="artifact-link">
      {label}
    </a>
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
