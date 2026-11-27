import type { Metadata } from "next"
import Link from "next/link"
import {
  DEFAULT_GPU,
  HOME_GPU_TABS,
  reportCardForBench,
} from "@/app/_lib/models"
import { loadModelIndex } from "@/app/_lib/models.server"
import { SITE_URL } from "@/app/_lib/seo.server"
import { HomeDecks, type HomeDeck } from "@/app/_components/home-decks"
import { PageHead } from "@/app/_components/page-head"

// KernelBench Hard was deprecated on 2026-11-23 and removed from the homepage
// and active model rankings. Keep /hard as a frozen historical archive so
// existing links from posts and articles remain readable.

export const metadata: Metadata = {
  title: "KernelBench Hard (Deprecated Archive)",
  description:
    "Frozen historical archive of KernelBench Hard (6 per-op CUDA/Triton kernels on H100 PCIe and RTX PRO 6000), deprecated on 2026-11-23 and superseded by KernelBench Mega and KernelBench CUDA.",
  alternates: { canonical: "/hard" },
  openGraph: {
    title: "KernelBench Hard (Deprecated Archive) · kernelbench.com",
    description:
      "Frozen historical archive of KernelBench Hard, superseded by KernelBench Mega and CUDA.",
    url: `${SITE_URL}/hard`,
  },
}

const HARD_TRACES_HF =
  "https://huggingface.co/datasets/Infatoshi/kernelbench-hard-traces"

export default async function HardArchivePage() {
  const modelIdx = await loadModelIndex()

  const hardByGpu: NonNullable<HomeDeck["byGpu"]> = {}
  for (const g of HOME_GPU_TABS) {
    hardByGpu[g.key] = reportCardForBench(
      modelIdx,
      "hard",
      g.key === "rtxpro6000" ? undefined : g.key,
    )
  }

  const decks: HomeDeck[] = [
    {
      key: "hard",
      title: "Hard (archived)",
      accent: "#a78bfa",
      byGpu: hardByGpu,
      gpus: HOME_GPU_TABS,
      defaultGpu: DEFAULT_GPU,
    },
  ]

  return (
    <div className="space-y-8">
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{
          __html: JSON.stringify({
            "@context": "https://schema.org",
            "@type": "BreadcrumbList",
            itemListElement: [
              { "@type": "ListItem", position: 1, name: "kernelbench.com", item: SITE_URL },
              { "@type": "ListItem", position: 2, name: "Hard (archived)", item: `${SITE_URL}/hard` },
            ],
          }),
        }}
      />
      <PageHead
        kicker="Deprecated · Frozen Archive"
        title="KernelBench Hard"
        sub={
          <>
            <span className="status-pill status-pill-warn mr-2">deprecated</span>
            Superseded by{" "}
            <Link href="/#mega">KernelBench Mega</Link> and{" "}
            <Link href="/#cuda">KernelBench CUDA</Link>. No longer updated for
            new models.
          </>
        }
        notes={
          <p>
            KernelBench Hard (6 per-op Triton/CUDA kernels) was retired from the
            active homepage and model rankings on 2026-11-23. Historical boards,
            per-run pages, and HuggingFace transcripts in{" "}
            <a href={HARD_TRACES_HF} target="_blank" rel="noreferrer">
              kernelbench-hard-traces
            </a>{" "}
            stay archived here for reference.
          </p>
        }
      />

      <HomeDecks decks={decks} />

      <p className="hd-legend" aria-label="Outcome legend">
        <span>
          <b>frozen archive</b> scores reflect the final Hard snapshot · click a
          cell for solution / trace
        </span>
      </p>
    </div>
  )
}
