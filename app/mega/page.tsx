import Link from "next/link"
import { barsForBench } from "@/app/_lib/models"
import { loadModelIndex } from "@/app/_lib/models.server"
import { ModelGpuBoard, type GpuView } from "@/app/_components/model-board"
import { PageHead } from "@/app/_components/page-head"

// KernelBench-Mega: whole-block megakernels. Ranked model list per GPU board.

const REFERENCE_URL = `/code?f=${encodeURIComponent(
  "/data/mega/code/02_kimi_linear_decode.reference.py.txt",
)}`
const BASELINE_URL = `/code?f=${encodeURIComponent(
  "/data/mega/code/02_kimi_linear_decode.baseline.py.txt",
)}`

export default async function MegaPage({
  searchParams,
}: {
  searchParams: Promise<{ gpu?: string }>
}) {
  const { gpu } = await searchParams
  const idx = await loadModelIndex()
  const gpuLabels = idx.benches.mega?.gpu_labels ?? {}

  const mk = (gpu?: string): Pick<GpuView, "bars"> => ({
    bars: barsForBench(idx, "mega", gpu),
  })

  const views: GpuView[] = [
    {
      key: "rtxpro6000",
      label: gpuLabels.rtxpro6000 ?? "RTX PRO 6000 Blackwell",
      ...mk(),
    },
    { key: "h100", label: gpuLabels.h100 ?? "H100", ...mk("h100") },
    { key: "b200", label: gpuLabels.b200 ?? "B200", ...mk("b200") },
  ]

  return (
    <div className="space-y-6">
      <PageHead
        kicker="Benchmark · Mega"
        title="The megakernel deck"
        sub={
          <>
            The agent fuses an entire model block into one kernel —{" "}
            <code>02_kimi_linear_decode</code> is a Kimi-Linear W4A16 hybrid
            decode — graded on <strong>decode speedup over an optimized-PyTorch
            baseline</strong>, not a roofline fraction. One unlimited session
            per cell on RTX PRO 6000, H100, and B200.
          </>
        }
        notes={
          <>
            <p>
              <strong>The metric.</strong> <code>19.35x</code> means 19x faster
              than the reference baseline; <code>tok/s</code> is decode tokens
              per second. Higher is better for both, reported per GPU. The
              transcript is the headline artifact: the model&apos;s full
              optimization journey from baseline to final megakernel.
            </p>
            <p>
              <strong>Ordering.</strong> Each run is a single autonomous
              session with unlimited wall-clock; models self-terminate well
              under three hours. Models group by valid passes, then order by
              best decode speedup. The flagged badge counts audited sessions
              that failed the reward-hack or megakernel-authenticity review; it
              never changes the order.
            </p>
            <p>
              <strong>Artifacts.</strong> The problem&apos;s{" "}
              <Link href={REFERENCE_URL}>reference</Link> and{" "}
              <Link href={BASELINE_URL}>baseline</Link> are self-hosted. Browse
              the <Link href="/runs">run index</Link> for transcripts and
              solutions, or the{" "}
              <Link href="https://github.com/Infatoshi/kernelbench.com/tree/master/benchmarks/mega">
                mega benchmark source
              </Link>
              .
            </p>
          </>
        }
      />
      <ModelGpuBoard views={views} initialGpu={gpu} />
    </div>
  )
}
