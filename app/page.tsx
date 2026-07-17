import Link from "next/link"
import { EfficiencyChart } from "./efficiency-chart"
import { loadEfficiency } from "@/app/_lib/charts"
import { SITE_HIDDEN_GPUS, columnOrder, columnsForBench, columnsForCorrectness } from "@/app/_lib/models"
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
    tagline: "Whole-block fused megakernels, graded on decode speedup over optimized PyTorch.",
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
    tagline: "Six-op CUDA/Triton deck, roofline-graded, one unlimited agent session per cell.",
    gpus: [
      { label: "RTX PRO 6000", href: "/hard" },
      { label: "H100", href: "/hard?gpu=h100" },
      { label: "B200", href: "/hard?gpu=b200" },
    ],
    hfHref: "https://huggingface.co/datasets/Infatoshi/kernelbench-hard-traces",
    ghHref:
      "https://github.com/Infatoshi/kernelbench.com/tree/master/benchmarks/hard",
    comingSoon: false,
  },
  {
    href: "/cuda",
    title: "CUDA",
    tagline: "CUDA-only writing deck — Triton and kernel DSLs fail the language gate.",
    gpus: [{ label: "RTX PRO 6000", href: "/cuda" }],
    hfHref: "https://huggingface.co/datasets/Infatoshi/kernelbench-cuda-traces",
    ghHref:
      "https://github.com/Infatoshi/kernelbench.com/tree/master/benchmarks/cuda",
    comingSoon: false,
  },
  {
    href: "/multi",
    title: "Multi",
    tagline: "NVLink collectives rewritten as kernels on 8×H100 SXM, graded on busbw.",
    gpus: [{ label: "8×H100 SXM", href: "/multi" }],
    hfHref: null,
    ghHref:
      "https://github.com/Infatoshi/kernelbench.com/tree/master/benchmarks/multi",
    comingSoon: true,
  },
]

// Decorative roofline: bandwidth slope into compute roof, dot pulsing at the
// knee. Pure SVG, drawn once on load.
function RooflineMotif() {
  return (
    <div className="hero-art-panel">
      <p className="hero-art-cap">the roofline</p>
      <svg
        className="hero-art"
        viewBox="0 0 300 170"
        aria-hidden="true"
        fill="none"
      >
      <defs>
        <filter id="roof-glow" x="-40%" y="-40%" width="180%" height="180%">
          <feGaussianBlur stdDeviation="3.2" result="b" />
          <feMerge>
            <feMergeNode in="b" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>
      {/* grid */}
      {[40, 80, 120, 160, 200, 240, 280].map((x) => (
        <line key={`v${x}`} x1={x} y1="8" x2={x} y2="150" stroke="#2b2b2b" strokeWidth="1" />
      ))}
      {[30, 70, 110, 150].map((y) => (
        <line key={`h${y}`} x1="16" y1={y} x2="292" y2={y} stroke="#2b2b2b" strokeWidth="1" />
      ))}
      {/* baseline */}
      <line x1="16" y1="150" x2="292" y2="150" stroke="#444444" strokeWidth="1" />
      {/* stragglers (dim dots below the roof) */}
      {[
        [60, 120], [95, 96], [130, 118], [175, 92], [215, 104], [250, 78],
      ].map(([cx, cy]) => (
        <circle key={`${cx}-${cy}`} cx={cx} cy={cy} r="2.6" fill="#6b7480" />
      ))}
      {/* the roof */}
      <path
        className="roof-path"
        d="M 16 150 L 128 38 L 292 38"
        stroke="#76b900"
        strokeWidth="2.4"
        strokeLinejoin="round"
        filter="url(#roof-glow)"
      />
      {/* knee of the roof — the frontier */}
      <circle className="roof-dot" cx="128" cy="38" r="3.6" fill="#76b900" filter="url(#roof-glow)" />
      <circle className="roof-dot" cx="128" cy="38" r="8" fill="none" stroke="#76b900" strokeWidth="1" opacity="0.5" />
      </svg>
    </div>
  )
}

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
  const boards = (["mega", "hard", "cuda"] as const).reduce((n, b) => {
    const g = modelIdx.benches[b]?.gpu_labels
    return n + (g ? Object.keys(g).filter((k) => !SITE_HIDDEN_GPUS.has(k)).length : 1)
  }, 0)
  return (
    <div className="space-y-12">
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(citationGraph) }}
      />
      <section className="hero" aria-label="About">
        <div className="hero-copy">
          <p className="hero-kicker">
            Agentic GPU kernel benchmarks
            <span className="hero-cursor" aria-hidden="true" />
          </p>
          <h1 className="hero-line">
            Frontier coding agents write the kernels.
            <br />
            <span className="dim">The roofline keeps score.</span>
          </h1>
          <ul className="hero-stats">
            <li>
              <strong>{ordered.length}</strong> models ranked
            </li>
            <li>
              <strong>3</strong> bench decks <span className="accent">+ multi soon</span>
            </li>
            <li>
              <strong>{boards}</strong> GPU boards
            </li>
            <li>
              updated <strong>{new Date(modelIdx.generated).toISOString().slice(0, 10)}</strong>
            </li>
          </ul>
        </div>
        <RooflineMotif />
      </section>

      <section aria-label="Models">
        <div className="section-head">
          <h2 className="section-title">Rankings</h2>
          <span className="section-note">
            each chart ranks by its own score, best left — click a column for
            cells, audits, and integrity record
          </span>
        </div>
        <ModelScoreboards perf={perfCharts} correctness={correctnessChart} />
      </section>

      <section aria-label="Benchmarks">
        <div className="section-head">
          <h2 className="section-title">The decks</h2>
          <span className="section-note">
            pick a GPU board — frozen decks, public harnesses, traces on
            Hugging&nbsp;Face
          </span>
        </div>
        <div className="bench-grid">
          {benchmarks.map((benchmark) => (
            <article
              key={benchmark.href}
              className={`bench-tile${benchmark.comingSoon ? " bench-tile-ghost" : ""}`}
            >
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
              </div>
            </article>
          ))}
        </div>
      </section>

      <section aria-label="Performance vs compute" className="box p-6">
        <EfficiencyChart mega={efficiency.mega} hard={efficiency.hard} />
      </section>

      <section aria-label="Method">
        <div className="section-head">
          <h2 className="section-title">Method</h2>
        </div>
        <div className="method-grid">
          <div className="method-card">
            <span className="method-num">01</span>
            <h3>Roofline, not speedup</h3>
            <p>Scores ground in hardware ceilings; baseline quirks can&apos;t move them.</p>
          </div>
          <div className="method-card">
            <span className="method-num">02</span>
            <h3>Real agent harnesses</h3>
            <p>Claude Code, Codex, Cursor, Kimi, OpenCode, Grok — the tools labs actually ship.</p>
          </div>
          <div className="method-card">
            <span className="method-num">03</span>
            <h3>Public transcripts</h3>
            <p>
              Every run — tools, reasoning, diffs — on{" "}
              <Link href="/runs">the run index</Link> and Hugging Face.
            </p>
          </div>
          <div className="method-card">
            <span className="method-num">04</span>
            <h3>Judge-assisted audit</h3>
            <p>Reward hacks and rubric leaks get flagged, published, and linked per cell.</p>
          </div>
        </div>
      </section>

      <details className="cite-details scroll-mt-24" id="cite">
        <summary>Cite this benchmark suite</summary>
        <div className="cite-body">
          <div className="citation-links" style={{ marginTop: 0 }}>
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
        </div>
      </details>

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
