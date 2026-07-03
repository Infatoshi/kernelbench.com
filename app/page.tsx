import Link from "next/link"
import { GroupedBars } from "./home-charts"
import { EfficiencyChart } from "./efficiency-chart"
import { loadMegaChart, loadHardChart, loadEfficiency } from "@/app/_lib/charts"

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
        "https://github.com/Infatoshi/kernelbench.com/tree/master/benchmarks/mega",
        "https://github.com/Infatoshi/kernelbench.com/tree/master/benchmarks/hard",
        "https://huggingface.co/datasets/Infatoshi/kernelbench-mega-traces",
        "https://huggingface.co/datasets/Infatoshi/kernelbench-hard-traces",
      ],
    },
    {
      "@type": "Dataset",
      "@id": "https://huggingface.co/datasets/Infatoshi/kernelbench-mega-traces",
      name: "KernelBench-Mega agent traces",
      url: "https://huggingface.co/datasets/Infatoshi/kernelbench-mega-traces",
      creator: { "@type": "Person", name: "Elliot Arledge" },
    },
    {
      "@type": "Dataset",
      "@id": "https://huggingface.co/datasets/Infatoshi/kernelbench-hard-traces",
      name: "KernelBench-Hard agent traces",
      url: "https://huggingface.co/datasets/Infatoshi/kernelbench-hard-traces",
      creator: { "@type": "Person", name: "Elliot Arledge" },
    },
    {
      "@type": "SoftwareSourceCode",
      "@id": "https://github.com/Infatoshi/kernelbench.com/tree/master/benchmarks/mega",
      name: "Mega result suite repository",
      codeRepository: "https://github.com/Infatoshi/kernelbench.com/tree/master/benchmarks/mega",
      author: { "@type": "Person", name: "Elliot Arledge" },
    },
    {
      "@type": "SoftwareSourceCode",
      "@id": "https://github.com/Infatoshi/kernelbench.com/tree/master/benchmarks/hard",
      name: "Hard result suite repository",
      codeRepository: "https://github.com/Infatoshi/kernelbench.com/tree/master/benchmarks/hard",
      author: { "@type": "Person", name: "Elliot Arledge" },
    },
  ],
}

const benchmarks = [
  {
    href: "/mega",
    title: "Mega",
    description:
      "Full GPU megakernel build (W4A16 linear decode), 8 frontier models across Blackwell / H100 / B200, scored as speedup over the reference. Contamination-audited in a sandboxed harness.",
    stats: [
      ["problem", "megakernel"],
      ["models", "8"],
      ["GPUs", "3"],
    ],
    hfHref: "https://huggingface.co/datasets/Infatoshi/kernelbench-mega-traces",
    ghHref:
      "https://github.com/Infatoshi/kernelbench.com/tree/master/benchmarks/mega",
    comingSoon: false,
  },
  {
    href: "/hard",
    title: "Hard",
    description:
      "Curated hard CUDA/Triton deck, frontier-model comparison across RTX PRO 6000 Blackwell and B200, roofline-graded with per-run agent traces.",
    stats: [
      ["problems", "6"],
      ["models", "8"],
      ["GPUs", "2"],
    ],
    hfHref: "https://huggingface.co/datasets/Infatoshi/kernelbench-hard-traces",
    ghHref:
      "https://github.com/Infatoshi/kernelbench.com/tree/master/benchmarks/hard",
    comingSoon: false,
  },
  {
    href: "/multi",
    title: "Multi",
    description:
      "The multi-GPU (NVLink) sibling of Hard: agents rewrite PyTorch + NCCL collectives as fine-grained NVLink kernels on an 8×H100 SXM (NVSwitch) node, graded on busbw — bus-bandwidth efficiency, not TFLOPS. Six communication-dominated problems.",
    stats: [
      ["problems", "6"],
      ["GPUs", "8×H100"],
      ["metric", "busbw"],
    ],
    hfHref: null,
    ghHref:
      "https://github.com/Infatoshi/kernelbench.com/tree/master/benchmarks/multi",
    comingSoon: true,
  },
]

export default async function HomePage() {
  const [megaChart, hardChart, efficiency] = await Promise.all([
    loadMegaChart(),
    loadHardChart(),
    loadEfficiency(),
  ])
  const charts: Record<string, React.ReactNode> = {
    "/mega": (
      <GroupedBars
        title="Speedup over reference megakernel"
        subtitle="Kimi linear-decode megakernel, per model. Pick a GPU."
        data={megaChart}
      />
    ),
    "/hard": (
      <GroupedBars
        title="Percent of hardware roofline"
        subtitle="Average across solved problems, relative to the GPU's own roofline. Pick a GPU."
        data={hardChart}
      />
    ),
  }
  return (
    <div className="space-y-12">
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(citationGraph) }}
      />
      <h1 className="sr-only">Agentic GPU kernel benchmark results</h1>
      <p className="text-sm text-[var(--color-fg-muted)]">
        For business inquiries reach out to{" "}
        <a href="mailto:elliot@arledge.net" className="font-bold text-[var(--color-fg-bright)]">
          elliot@arledge.net
        </a>
        .
      </p>
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
                  <h2 className="text-2xl sm:text-3xl font-semibold tracking-tight text-[var(--color-fg-bright)] flex items-center gap-3">
                    {benchmark.title}
                    {benchmark.comingSoon && (
                      <span className="text-xs font-semibold uppercase tracking-wide px-2 py-1 rounded border border-[var(--color-accent)] text-[var(--color-accent)]">
                        coming soon
                      </span>
                    )}
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
            {charts[benchmark.href]}
            <div className="benchmark-footer flex flex-wrap items-center gap-4">
              <a
                href={benchmark.ghHref}
                target="_blank"
                rel="noreferrer"
                className="hf-link no-underline"
                aria-label={`Open ${benchmark.title} benchmark source on GitHub`}
              >
                <svg
                  viewBox="0 0 24 24"
                  aria-hidden="true"
                  width="20"
                  height="20"
                  fill="currentColor"
                >
                  <path d="M12 2C6.48 2 2 6.58 2 12.26c0 4.53 2.87 8.37 6.84 9.73.5.1.68-.22.68-.49 0-.24-.01-1.04-.01-1.89-2.78.62-3.37-1.22-3.37-1.22-.45-1.19-1.11-1.5-1.11-1.5-.91-.64.07-.63.07-.63 1 .07 1.53 1.06 1.53 1.06.9 1.57 2.35 1.12 2.92.86.09-.67.35-1.12.63-1.38-2.22-.26-4.55-1.14-4.55-5.07 0-1.12.39-2.03 1.03-2.75-.1-.26-.45-1.3.1-2.71 0 0 .84-.28 2.75 1.05A9.29 9.29 0 0 1 12 6.98c.85 0 1.71.12 2.51.34 1.91-1.33 2.75-1.05 2.75-1.05.55 1.41.2 2.45.1 2.71.64.72 1.03 1.63 1.03 2.75 0 3.94-2.34 4.81-4.57 5.06.36.32.68.94.68 1.9 0 1.38-.01 2.49-.01 2.83 0 .27.18.59.69.49A10.05 10.05 0 0 0 22 12.26C22 6.58 17.52 2 12 2Z" />
                </svg>
                <span>GitHub source</span>
              </a>
              {benchmark.hfHref && (
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
              )}
            </div>
          </article>
        ))}
      </section>

      <section aria-label="Performance vs compute" className="box p-6">
        <EfficiencyChart mega={efficiency.mega} hard={efficiency.hard} />
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
            or open the{" "}
            <Link
              href="https://huggingface.co/datasets/Infatoshi/kernelbench-hard-traces"
              className="underlined-link"
            >
              Hard agent traces on Hugging Face
            </Link>{" "}
            to inspect tool calls, reasoning, solution files, and timing.
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
            href="https://github.com/Infatoshi/kernelbench.com/tree/master/benchmarks/mega"
            label="Mega repository"
          />
          <ArtifactLink
            href="https://github.com/Infatoshi/kernelbench.com/tree/master/benchmarks/hard"
            label="Hard repository"
          />
          <ArtifactLink
            href="https://huggingface.co/datasets/Infatoshi/kernelbench-mega-traces"
            label="Mega HF traces"
          />
          <ArtifactLink
            href="https://huggingface.co/datasets/Infatoshi/kernelbench-hard-traces"
            label="Hard HF traces"
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
  howpublished = {\\url{https://github.com/Infatoshi/kernelbench.com/tree/master/benchmarks/hard}},
  note         = {CUDA benchmark suite, harness, results, and annotations}
}

@misc{arledge2026mega,
  title        = {Mega: Agentic GPU Megakernel Result Suite},
  author       = {Arledge, Elliot},
  year         = {2026},
  howpublished = {\\url{https://github.com/Infatoshi/kernelbench.com/tree/master/benchmarks/mega}},
  note         = {Megakernel benchmark suite, sandboxed harness, and result artifacts}
}

@misc{arledge2026hardtraces,
  title        = {KernelBench-Hard Agent Traces},
  author       = {Arledge, Elliot},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\\url{https://huggingface.co/datasets/Infatoshi/kernelbench-hard-traces}},
  note         = {Per-run agent transcripts: messages, tool calls, reasoning}
}

@misc{arledge2026megatraces,
  title        = {KernelBench-Mega Agent Traces},
  author       = {Arledge, Elliot},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\\url{https://huggingface.co/datasets/Infatoshi/kernelbench-mega-traces}},
  note         = {Per-run agent transcripts for the megakernel suite}
}`}
        </pre>
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
