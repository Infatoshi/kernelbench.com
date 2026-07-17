import Link from "next/link"
import { barsForBench } from "@/app/_lib/models"
import { loadModelIndex } from "@/app/_lib/models.server"
import { ModelGpuBoard, type GpuView } from "@/app/_components/model-board"
import { PageHead } from "@/app/_components/page-head"

// KernelBench-Hard: per-op kernel deck. AA-style model bar chart per GPU
// board; run-level forensics live on /runs and the per-model pages.

export default async function HardPage({
  searchParams,
}: {
  searchParams: Promise<{ gpu?: string }>
}) {
  const { gpu } = await searchParams
  const idx = await loadModelIndex()
  const gpuLabels = idx.benches.hard?.gpu_labels ?? {}

  const mk = (gpu?: string): Pick<GpuView, "bars"> => ({
    bars: barsForBench(idx, "hard", gpu),
  })

  const views: GpuView[] = [
    {
      key: "rtxpro6000",
      label: gpuLabels.rtxpro6000 ?? "RTX PRO 6000",
      blurb:
        "Unlimited time per problem. The frozen lab board — every published cell is contamination-checked and reward-hack audited.",
      ...mk(),
    },
    {
      key: "h100",
      label: gpuLabels.h100 ?? "H100 PCIe",
      ...mk("h100"),
    },
    {
      key: "b200",
      label: gpuLabels.b200 ?? "B200",
      ...mk("b200"),
    },
    {
      key: "rtx3090",
      label: gpuLabels.rtx3090 ?? "RTX 3090",
      ...mk("rtx3090"),
    },
  ]

  return (
    <div className="space-y-6">
      <PageHead
        kicker="Benchmark · Hard"
        title="The per-op kernel deck"
        sub={
          <>
            Six CUDA/Triton problems, one unlimited agent session per cell,
            graded against each GPU&apos;s own roofline — comparable within a
            GPU, not across GPUs. Correctness groups the board; flagged audits
            never reorder it.
          </>
        }
        notes={
          <>
            <p>
              <strong>Ordering.</strong> Models group by valid passes
              (audited-clean correct cells), then order by mean peak fraction.
              The flagged count lists audited sessions that failed the
              reward-hack review; it never changes the order. Click a model for
              its per-problem cells, audit chips, and integrity record. Browse
              the <Link href="/runs">run index</Link> for transcripts, submitted
              solutions, checks, timing, and costs.
            </p>
            <p>
              <strong>RTX PRO 6000.</strong> Frontier coding agents on one
              unlimited autonomous session per problem: Opus 4.8, Fable 5, Grok
              4.5, GPT-5.6 Sol, GLM-5.2, MiniMax-M3, DeepSeek V4 Pro, LongCat
              2.0, and more. Roofline-graded; every published cell is
              contamination-checked and reward-hack audited.
            </p>
            <p>
              <strong>H100 PCIe.</strong> Opus 4.8, Fable 5, GLM-5.2,
              MiniMax-M3, DeepSeek V4 Pro, and LongCat 2.0 on a single H100
              PCIe with the same containerized harness and roofline grading as
              the Blackwell deck; peak fraction is measured against H100 dense
              peaks.
            </p>
            <p>
              <strong>B200.</strong> The same models on a single NVIDIA B200
              (SM100 Blackwell, HBM3e) with the identical containerized harness
              and roofline grading; peak fraction is measured against B200
              dense peaks (fp8 4500, bf16 2250 TFLOPS), so the same kernel
              reads as a smaller fraction of a much higher ceiling.
            </p>
            <p>
              <strong>RTX 3090.</strong> The first consumer/Ampere part in the
              set: a single RTX 3090 (SM86, GDDR6X, 936 GB/s). Ampere has no
              FP8 tensor cores, so the FP8 GEMM problem is dropped entirely —
              five problems, not six. The memory-bound cells (paged attention,
              top-k, W4A16) port cleanly and lead the board, with
              paged-attention decode reaching ~66% of the 936 GB/s ceiling; the
              bf16 compute cells (KDA, sonic-MoE) run against Ampere&apos;s
              much lower bf16 peak. GLM-5.2 is a clean 5/5 and its
              paged-attention kernel holds the board&apos;s top bandwidth
              fraction (~66%). Only MiniMax-M3 is partial — its sweep was cut
              short by provider rate limits.
            </p>
          </>
        }
      />
      <ModelGpuBoard views={views} initialGpu={gpu} />
    </div>
  )
}
