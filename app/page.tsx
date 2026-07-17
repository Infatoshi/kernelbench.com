import Link from "next/link"
import { EfficiencyChart } from "./efficiency-chart"
import { loadEfficiency } from "@/app/_lib/charts"
import { columnOrder, columnsForBench, columnsForCorrectness } from "@/app/_lib/models"
import { loadModelIndex } from "@/app/_lib/models.server"
import { ModelScoreboards } from "@/app/_components/model-columns"

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
        "https://github.com/Infatoshi/kernelbench.com/tree/master/benchmarks/cuda",
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

// Compact bench tiles: the model charts above carry the numbers, so each
// bench gets a one-liner, GPU deep-links into its board, and the artifact
// links (GitHub source, HF traces).
const benchmarks = [
  {
    href: "/mega",
    title: "Mega",
    tagline:
      "Whole-block megakernels — best decode speedup over an optimized-PyTorch baseline.",
    gpus: [
      { label: "RTX PRO 6000", href: "/mega" },
      { label: "H100", href: "/mega?gpu=h100" },
      { label: "B200", href: "/mega?gpu=b200" },
    ],
    hfHref: "https://huggingface.co/datasets/Infatoshi/kernelbench-mega-traces",
    ghHref:
      "https://github.com/Infatoshi/kernelbench.com/tree/master/benchmarks/mega",
    comingSoon: false,
  },
  {
    href: "/hard",
    title: "Hard",
    tagline:
      "Six-problem CUDA/Triton deck, roofline-graded, one unlimited-time agent session per cell.",
    gpus: [
      { label: "RTX PRO 6000", href: "/hard" },
      { label: "H100", href: "/hard?gpu=h100" },
      { label: "B200", href: "/hard?gpu=b200" },
      { label: "RTX 3090", href: "/hard?gpu=rtx3090" },
    ],
    hfHref: "https://huggingface.co/datasets/Infatoshi/kernelbench-hard-traces",
    ghHref:
      "https://github.com/Infatoshi/kernelbench.com/tree/master/benchmarks/hard",
    comingSoon: false,
  },
  {
    href: "/cuda",
    title: "CUDA",
    tagline:
      "CUDA-only writing deck — Triton and kernel DSLs fail the language gate.",
    gpus: [{ label: "RTX PRO 6000", href: "/cuda" }],
    hfHref: "https://huggingface.co/datasets/Infatoshi/kernelbench-cuda-traces",
    ghHref:
      "https://github.com/Infatoshi/kernelbench.com/tree/master/benchmarks/cuda",
    comingSoon: false,
  },
  {
    href: "/multi",
    title: "Multi",
    tagline:
      "NVLink collectives rewritten as kernels on 8×H100 SXM, graded on busbw.",
    gpus: [{ label: "8×H100 SXM", href: "/multi" }],
    hfHref: null,
    ghHref:
      "https://github.com/Infatoshi/kernelbench.com/tree/master/benchmarks/multi",
    comingSoon: true,
  },
]

export default async function HomePage() {
  const [efficiency, modelIdx] = await Promise.all([
    loadEfficiency(),
    loadModelIndex(),
  ])
  // columnOrder applies the curated roster filter; each chart then re-ranks
  // its columns by its own metric (highest left -> lowest right) and drops
  // models with no result on that bench.
  const ordered = columnOrder(modelIdx)
  const perfCharts = (["mega", "hard", "cuda"] as const).map((b) =>
    columnsForBench(modelIdx, b, ordered),
  )
  const correctnessChart = columnsForCorrectness(modelIdx, ordered)
  return (
    <div className="space-y-12">
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(citationGraph) }}
      />
      <h1 className="sr-only">Agentic GPU kernel benchmark results</h1>
      <section aria-label="Models" className="space-y-4">
        <div>
          <h2 className="text-xl font-semibold text-[var(--color-fg-bright)]">
            Models
          </h2>
          <p className="text-xs text-[var(--color-fg-muted)] mt-1.5 leading-relaxed">
            Frontier coding models on the kernel decks, AA-style. Performance
            is disaggregated per benchmark (Mega / Hard / CUDA, bars colored by
            lab), and the last chart is compiled correctness: the percentage of
            published problems each model gets correct across the benches it
            attempted. Each chart ranks models by its own score, best on the
            left. Scores use plain, on-page math: mean peak fraction over
            valid cells, best decode speedup, passed&nbsp;/&nbsp;total. Click a
            column for per-problem cells, audit chips, and the model&apos;s
            integrity record.
          </p>
        </div>
        <ModelScoreboards perf={perfCharts} correctness={correctnessChart} />
      </section>

      <section aria-label="Benchmarks" className="bench-grid">
        {benchmarks.map((benchmark) => (
          <article key={benchmark.href} className="bench-tile">
            <div className="bench-tile-head">
              <Link
                href={benchmark.href}
                className="bench-tile-title no-underline"
                aria-label={`Open ${benchmark.title} benchmark`}
              >
                {benchmark.title}
                <svg
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  aria-hidden="true"
                >
                  <path d="M7 17 17 7" />
                  <path d="M9 7h8v8" />
                </svg>
              </Link>
              {benchmark.comingSoon && (
                <span className="bench-tile-soon">coming soon</span>
              )}
            </div>
            <p className="bench-tile-tag">{benchmark.tagline}</p>
            <div className="bench-tile-gpus" aria-label={`${benchmark.title} GPU boards`}>
              {benchmark.gpus.map((g) => (
                <Link key={g.label} href={g.href} className="bench-gpu-chip no-underline">
                  {g.label}
                </Link>
              ))}
            </div>
            <div className="bench-tile-links">
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
                  width="16"
                  height="16"
                  fill="currentColor"
                >
                  <path d="M12 2C6.48 2 2 6.58 2 12.26c0 4.53 2.87 8.37 6.84 9.73.5.1.68-.22.68-.49 0-.24-.01-1.04-.01-1.89-2.78.62-3.37-1.22-3.37-1.22-.45-1.19-1.11-1.5-1.11-1.5-.91-.64.07-.63.07-.63 1 .07 1.53 1.06 1.53 1.06.9 1.57 2.35 1.12 2.92.86.09-.67.35-1.12.63-1.38-2.22-.26-4.55-1.14-4.55-5.07 0-1.12.39-2.03 1.03-2.75-.1-.26-.45-1.3.1-2.71 0 0 .84-.28 2.75 1.05A9.29 9.29 0 0 1 12 6.98c.85 0 1.71.12 2.51.34 1.91-1.33 2.75-1.05 2.75-1.05.55 1.41.2 2.45.1 2.71.64.72 1.03 1.63 1.03 2.75 0 3.94-2.34 4.81-4.57 5.06.36.32.68.94.68 1.9 0 1.38-.01 2.49-.01 2.83 0 .27.18.59.69.49A10.05 10.05 0 0 0 22 12.26C22 6.58 17.52 2 12 2Z" />
                </svg>
                <span>GitHub</span>
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
                    width="16"
                    height="16"
                    loading="lazy"
                  />
                  <span>Traces</span>
                </a>
              )}
              <Link href="/runs" className="hf-link no-underline">
                <span>Run index</span>
              </Link>
            </div>
          </article>
        ))}
      </section>

      <section aria-label="Performance vs compute" className="box p-6">
        <EfficiencyChart mega={efficiency.mega} hard={efficiency.hard} />
      </section>

      <section className="space-y-5">
        <p className="text-sm text-[var(--color-fg-muted)]">
          For business inquiries reach out to{" "}
          <a href="mailto:elliot@arledge.net" className="font-bold text-[var(--color-fg-bright)]">
            elliot@arledge.net
          </a>
          .
        </p>
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
            Runs use Claude Code, Codex CLI, Cursor, Kimi CLI,
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

function Bullet({ children }: { children: React.ReactNode }) {
  return (
    <li className="flex gap-3">
      <span className="text-[var(--color-fg-muted)] shrink-0">•</span>
      <span>{children}</span>
    </li>
  )
}
