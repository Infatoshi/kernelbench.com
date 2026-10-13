import {
  DEFAULT_GPU,
  HOME_GPU_TABS,
  reportCardForBench,
} from "@/app/_lib/models"
import { loadModelIndex } from "@/app/_lib/models.server"
import { HomeDecks, type HomeDeck } from "@/app/_components/home-decks"

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
    },
  ],
}

export default async function HomePage() {
  const modelIdx = await loadModelIndex()

  const megaByGpu: NonNullable<HomeDeck["byGpu"]> = {}
  const hardByGpu: NonNullable<HomeDeck["byGpu"]> = {}
  for (const g of HOME_GPU_TABS) {
    megaByGpu[g.key] = reportCardForBench(
      modelIdx,
      "mega",
      g.key === "rtxpro6000" ? undefined : g.key,
    )
    hardByGpu[g.key] = reportCardForBench(
      modelIdx,
      "hard",
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
      key: "hard",
      title: "Hard",
      accent: "#a78bfa",
      byGpu: hardByGpu,
      gpus: HOME_GPU_TABS,
      defaultGpu: DEFAULT_GPU,
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

  return (
    <div className="space-y-8">
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(citationGraph) }}
      />
      <HomeDecks decks={decks} />
      <p className="hd-legend" aria-label="Outcome legend">
        <span>
          <b>pass</b> bar = share of best · click a cell for solution / trace
        </span>
        <span>
          <b className="hd-leg-blank">blank</b> no audited pass (failed, unaudited or not run; the model page lists every outcome)
        </span>
      </p>
    </div>
  )
}
