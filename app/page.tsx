import type { Metadata } from "next"
import Link from "next/link"
import {
  BENCH_LABELS,
  DEFAULT_GPU,
  HOME_GPU_TABS,
  reportCardForBench,
} from "@/app/_lib/models"
import { loadModelIndex } from "@/app/_lib/models.server"
import {
  buildSiteAuditSummary,
  SITE_URL,
  TRACES_HF,
} from "@/app/_lib/seo.server"
import { HomeDecks, type HomeDeck } from "@/app/_components/home-decks"

export const metadata: Metadata = {
  alternates: {
    canonical: "/",
  },
}

export default async function HomePage() {
  const [modelIdx, summary] = await Promise.all([
    loadModelIndex(),
    buildSiteAuditSummary(),
  ])

  const megaByGpu: NonNullable<HomeDeck["byGpu"]> = {}
  for (const g of HOME_GPU_TABS) {
    megaByGpu[g.key] = reportCardForBench(
      modelIdx,
      "mega",
      g.key === "rtxpro6000" ? undefined : g.key,
    )
  }

  const cudaReport = reportCardForBench(modelIdx, "cuda")

  const decks: HomeDeck[] = [
    {
      key: "mega",
      title: "Mega",
      accent: "#76b900",
      byGpu: megaByGpu,
      gpus: HOME_GPU_TABS,
      defaultGpu: DEFAULT_GPU,
    },
    {
      key: "cuda",
      title: "CUDA",
      accent: "#38bdf8",
      byGpu: { rtxpro6000: cudaReport },
      gpus: [{ key: "rtxpro6000", label: "RTX PRO 6000" }],
      defaultGpu: "rtxpro6000",
    },
    {
      key: "multi",
      title: "Multi",
      accent: "#fbbf24",
      byGpu: null,
      gpus: [],
      defaultGpu: "",
    },
  ]

  const megaTop = summary.megaLeaders[0]
  const cudaTop = summary.cudaLeaders[0]
  const leaderSentence = [
    megaTop
      ? `On KernelBench Mega (Kimi-Linear Decode, RTX PRO 6000), ${megaTop.name} leads at ${megaTop.metric} decode speedup.`
      : null,
    cudaTop
      ? `On KernelBench CUDA (GLM-5.3 Fused MoE, DeepSeek NSA, MegaQwen Decode), ${cudaTop.name} leads across ${cudaTop.passed}/${cudaTop.total} workloads (${cudaTop.summary}).`
      : null,
  ]
    .filter(Boolean)
    .join(" ")

  const structuredGraph = {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "WebSite",
        "@id": `${SITE_URL}/#website`,
        name: "kernelbench.com",
        alternateName: ["KernelBench", "KernelBench Mega", "KernelBench CUDA"],
        url: SITE_URL,
        author: {
          "@type": "Person",
          "@id": `${SITE_URL}/#author`,
          name: "Elliot Arledge",
          url: "https://elliotarledge.com",
          sameAs: [
            "https://github.com/Infatoshi",
            "https://x.com/elliotarledge",
          ],
        },
        description:
          "Independent, reward-hack audited benchmark for autonomous coding agents optimizing GPU kernels and CUDA inference operators, with full kernel source, per-shape hardware roofline breakdowns, and HuggingFace agent transcripts.",
      },
      {
        "@type": "Dataset",
        "@id": `${SITE_URL}/#dataset`,
        name: "kernelbench.com — Audited Agentic GPU Kernel & CUDA Optimization Benchmark",
        alternateName: ["KernelBench Mega", "KernelBench CUDA"],
        url: SITE_URL,
        description:
          "Qualitative, audit-first GPU kernel optimization benchmark evaluating frontier AI coding agents on real inference engine workloads (fused MoE, DeepSeek Native Sparse Attention, MegaQwen Decode, and Kimi-Linear Decode megakernel). Every published cell undergoes isolated hardware regrading and manual reward-hack, contamination, and rubric-leak auditing with full source code and transcripts.",
        isAccessibleForFree: true,
        dateModified: summary.generated,
        creator: { "@id": `${SITE_URL}/#author` },
        publisher: { "@id": `${SITE_URL}/#author` },
        sameAs: [
          "https://github.com/Infatoshi/kernelbench.com",
          TRACES_HF.mega,
          TRACES_HF.cuda,
        ],
        keywords: [
          "GPU kernel optimization",
          "CUDA benchmark",
          "inference engine kernels",
          "agentic coding benchmark",
          "reward-hack audit",
          "hardware roofline",
          "fused MoE",
          "DeepSeek NSA",
          "Kimi-Linear Decode",
        ],
        measurementTechnique:
          "Isolated sequential GPU regrade on NVIDIA RTX PRO 6000 Blackwell (sm_120a) and H100 PCIe measuring multi-shape geomean fraction of governing hardware roofline (compute TFLOPS or HBM bandwidth), geomean millisecond latency, or full-model decode speedup vs optimized PyTorch, gated by a mandatory per-cell manual and subagent reward-hack audit YAML.",
        variableMeasured: [
          "Peak fraction of governing GPU hardware roofline (%)",
          "Geomean inference latency across tensor shapes (ms)",
          "Full-model decode speedup vs optimized PyTorch baseline (x)",
          "Manual reward-hack, contamination, and rubric-leak audit verdict",
        ],
        distribution: [
          {
            "@type": "DataDownload",
            name: "Structured Model & Audit Index (JSON)",
            encodingFormat: "application/json",
            contentUrl: `${SITE_URL}/data/models.json`,
          },
          {
            "@type": "DataDownload",
            name: "Machine-Readable Benchmark Summary (llms.txt)",
            encodingFormat: "text/plain",
            contentUrl: `${SITE_URL}/llms.txt`,
          },
          {
            "@type": "DataDownload",
            name: "KernelBench CUDA Agent Transcripts (HuggingFace)",
            encodingFormat: "application/x-ndjson",
            contentUrl: TRACES_HF.cuda,
          },
          {
            "@type": "DataDownload",
            name: "KernelBench Mega Agent Transcripts (HuggingFace)",
            encodingFormat: "application/x-ndjson",
            contentUrl: TRACES_HF.mega,
          },
        ],
      },
      {
        "@type": "ItemList",
        "@id": `${SITE_URL}/#cuda-leaders`,
        name: "Top Frontier Models on KernelBench CUDA (RTX PRO 6000)",
        itemListOrder: "https://schema.org/ItemListOrderDescending",
        numberOfItems: summary.cudaLeaders.length,
        itemListElement: summary.cudaLeaders.map((m, i) => ({
          "@type": "ListItem",
          position: i + 1,
          name: `${m.name} (${m.summary})`,
          url: `${SITE_URL}/models/${m.slug}`,
        })),
      },
      {
        "@type": "FAQPage",
        "@id": `${SITE_URL}/#faq`,
        mainEntity: [
          {
            "@type": "Question",
            name: "Which frontier models lead agentic GPU kernel and CUDA optimization on kernelbench.com?",
            acceptedAnswer: {
              "@type": "Answer",
              text: `${leaderSentence} Every cell links to the redacted CUDA kernel source, per-shape hardware utilization breakdown, and full HuggingFace agent transcript so engineers can inspect the actual optimization strategy rather than trusting an opaque average.`,
            },
          },
          {
            "@type": "Question",
            name: "How does kernelbench.com prevent GPU kernel reward hacking?",
            acceptedAnswer: {
              "@type": "Answer",
              text: `Every published cell must pass a sequential isolated regrade on an idle GPU (RTX PRO 6000 Blackwell or H100 PCIe) across multiple tensor shapes and a mandatory manual/subagent audit YAML inspecting both the final kernel source and the full agent transcript for check.py sniffing, tolerance edits, stale-buffer or CUDA-graph caching, rubric leaks, or cross-run contamination. Across ${summary.auditedSessions} audited active sessions, ${summary.flaggedSessions} flagged runs were caught, excluded from rankings, and kept publicly documented on each model page.`,
            },
          },
          {
            "@type": "Question",
            name: "Why is kernelbench.com qualitative and audit-first instead of a high-volume automated speedup benchmark?",
            acceptedAnswer: {
              "@type": "Answer",
              text: "Large automated kernel suites that score raw speedup over naive PyTorch eager baselines suffer heavily from proxy gaming, stream mismatches, loose numerical tolerances, and unverified wrappers. kernelbench.com intentionally trades raw problem count for deep qualitative auditability on real inference engine workloads (fused MoE, sparse attention, full-model decode megakernels), publishing every kernel, per-shape roofline measurement, failure mode, and raw agent trace.",
            },
          },
          {
            "@type": "Question",
            name: "Is kernelbench.com affiliated with Stanford KernelBench or BenchLM?",
            acceptedAnswer: {
              "@type": "Answer",
              text: "No. kernelbench.com is an independent benchmark and open evidence archive built and maintained by Elliot Arledge, unaffiliated with Stanford KernelBench. Third-party aggregators such as BenchLM mirror only historical snapshots (such as the deprecated Hard deck), whereas live audited Mega and CUDA boards, kernels, and traces are published directly on kernelbench.com.",
            },
          },
        ],
      },
    ],
  }

  return (
    <div className="space-y-8">
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(structuredGraph) }}
      />
      <div className="home-intro">
        <h1 className="home-intro-title">
          Audited Agentic GPU Kernel &amp; CUDA Optimization Benchmark
        </h1>
        <p className="home-intro-sub">
          Qualitative, reward-hack audited leaderboard for frontier coding agents optimizing inference engine kernels on real GPUs. Every cell links to its redacted kernel source, per-shape roofline breakdown, audit verdict, and full agent trace.
        </p>
      </div>
      <HomeDecks decks={decks} />
      <p className="hd-legend" aria-label="Outcome legend">
        <span>
          <b>pass</b> bar = share of best · click a cell for solution / trace
        </span>
        <span>
          <b className="hd-leg-blank">blank</b> no audited pass (failed, unaudited or not run; the model page lists every outcome)
        </span>
      </p>

      <section className="home-method" aria-labelledby="home-method-heading">
        <div className="home-method-grid">
          <div className="home-method-card">
            <h2 id="home-method-heading" className="home-method-h">
              Audit-first, qualitative GPU kernel evaluation
            </h2>
            <p className="home-method-p">
              High-volume automated kernel benchmarks often report noisy speedups
              against naive eager PyTorch baselines that agents routinely game via
              stale buffers, timer overwrites, mismatched CUDA streams, or loose
              tolerances. <strong>kernelbench.com</strong> flips that tradeoff:
              a small deck of real inference engine workloads graded against
              hardware rooflines (compute TFLOPS / HBM bandwidth), geomean
              milliseconds, or full-model decode speedup on isolated{" "}
              <strong>RTX PRO 6000 Blackwell</strong> and{" "}
              <strong>H100 PCIe</strong> GPUs.
            </p>
            <p className="home-method-p">
              Every scored cell requires a sequential idle-GPU regrade and a
              per-run reward-hack, contamination, and rubric-leak audit YAML
              reading the kernel source and transcript end to end ({summary.auditedSessions}{" "}
              audited active sessions; {summary.flaggedSessions} flagged and
              excluded from rankings, kept public on each model page).
            </p>
          </div>

          <div className="home-method-card">
            <h2 className="home-method-h">Current audited leaders (RTX PRO 6000)</h2>
            <ul className="home-leader-list">
              {summary.problemLeaders.map((p) => (
                <li key={`${p.bench}:${p.problem}`} className="home-leader-item">
                  <span className="home-leader-prob">
                    {BENCH_LABELS[p.bench]} · {p.problemName}
                  </span>
                  <span className="home-leader-val tabular">
                    <Link href={`/models/${p.modelSlug}`} className="no-underline">
                      <strong>{p.modelName}</strong>
                    </Link>{" "}
                    {p.runUrl ? (
                      <Link href={p.runUrl}>{p.metric}</Link>
                    ) : (
                      p.metric
                    )}
                    {p.runnerUpName && p.runnerUpMetric && (
                      <span className="home-leader-runner">
                        {" "}
                        (next: {p.runnerUpName} {p.runnerUpMetric})
                      </span>
                    )}
                  </span>
                </li>
              ))}
            </ul>
            <p className="home-method-links">
              <Link href="/models">Full ranked model directory ({summary.liveModelCount} live)</Link>
              {" · "}
              <Link href="/runs">Complete audited run index</Link>
              {" · "}
              <Link href="/hard">Deprecated Hard archive</Link>
              {" · "}
              <a href="/llms.txt">llms.txt</a>
              {" · "}
              <a href={TRACES_HF.cuda} target="_blank" rel="noreferrer">
                CUDA traces ↗
              </a>
              {" · "}
              <a href={TRACES_HF.mega} target="_blank" rel="noreferrer">
                Mega traces ↗
              </a>
            </p>
          </div>
        </div>
      </section>
    </div>
  )
}
