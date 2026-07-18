import Link from "next/link"
import { DEFAULT_GPU, reportCardForBench } from "@/app/_lib/models"
import { loadModelIndex } from "@/app/_lib/models.server"
import { ModelGpuBoard, type GpuView } from "@/app/_components/model-board"
import { PageHead } from "@/app/_components/page-head"

// KernelBench-Hard: per-problem report card per GPU board.
// Rank by passes only; each cell is score or fail reason (not a blended mean).

export default async function HardPage({
  searchParams,
}: {
  searchParams: Promise<{ gpu?: string }>
}) {
  const { gpu } = await searchParams
  const idx = await loadModelIndex()
  const gpuLabels = idx.benches.hard?.gpu_labels ?? {}

  const mk = (gpu?: string): Pick<GpuView, "report"> => ({
    report: reportCardForBench(idx, "hard", gpu),
  })

  // Tab order H100 → RTX PRO 6000 → B200; default selection is DEFAULT_GPU.
  const views: GpuView[] = [
    {
      key: "h100",
      label: gpuLabels.h100 ?? "H100 PCIe",
      ...mk("h100"),
    },
    {
      key: "rtxpro6000",
      label: gpuLabels.rtxpro6000 ?? "RTX PRO 6000",
      ...mk(),
    },
    {
      key: "b200",
      label: gpuLabels.b200 ?? "B200",
      ...mk("b200"),
    },
  ]

  return (
    <div className="space-y-6">
      <PageHead
        kicker="Benchmark · Hard"
        title="The per-op kernel deck"
        sub={
          <>
            Six problems × one unlimited agent session. Ranked by how many
            problems passed; each problem shows peak fraction or why it failed
            — fails are not averaged into a fake speed score.
          </>
        }
        notes={
          <>
            <p>
              <strong>Reading the board.</strong> Order is pass count only.
              Green chips are peak fraction of that GPU&apos;s roofline.{" "}
              <code>check</code> = solution failed correctness;{" "}
              <code>empty</code> = no kernel written; <code>hack</code> = audit
              flag. &quot;when ok&quot; is mean peak fraction among passes only
              (footnote, not a rank key). Click a model for audits and traces.{" "}
              <Link href="/runs">Run index</Link>.
            </p>
          </>
        }
      />
      <ModelGpuBoard views={views} initialGpu={gpu ?? DEFAULT_GPU} />
    </div>
  )
}
