import { loadSketchpadPayload } from "@/app/_lib/sketchpad.server"
import { buildSketches } from "@/app/_lib/sketches"
import { KernelBenchBoard } from "@/app/_components/kernelbench-board"

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
  const [payload, sketches] = await Promise.all([
    loadSketchpadPayload(),
    Promise.resolve(buildSketches()),
  ])

  return (
    <>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(citationGraph) }}
      />
      <KernelBenchBoard data={payload} sketches={sketches} />
    </>
  )
}
