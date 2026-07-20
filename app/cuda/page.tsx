import Link from "next/link"
import { barsForBench } from "@/app/_lib/models"
import { loadModelIndex } from "@/app/_lib/models.server"
import { ModelGpuBoard, type GpuView } from "@/app/_components/model-board"
import { PageHead } from "@/app/_components/page-head"

// KernelBench-CUDA: CUDA-only sibling of hard. Triton/DSL fail the language
// gate. Four-problem deck; PRO is the primary board, B200 when published.

const REPO_TREE =
  "https://github.com/Infatoshi/kernelbench.com/blob/master/benchmarks/cuda"

export default async function CudaPage({
  searchParams,
}: {
  searchParams: Promise<{ gpu?: string }>
}) {
  const { gpu } = await searchParams
  const idx = await loadModelIndex()
  const gpuLabels = idx.benches.cuda?.gpu_labels ?? {}

  const mk = (g?: string): Pick<GpuView, "bars"> => ({
    bars: barsForBench(idx, "cuda", g),
  })

  // PRO first (canonical), then B200 / H100 when those boards exist.
  const views: GpuView[] = [
    {
      key: "rtxpro6000",
      label: gpuLabels.rtxpro6000 ?? "RTX PRO 6000",
      ...mk(),
    },
  ]
  if (idx.benches.cuda?.gpus?.includes("b200")) {
    views.push({
      key: "b200",
      label: gpuLabels.b200 ?? "B200",
      ...mk("b200"),
    })
  }
  if (idx.benches.cuda?.gpus?.includes("h100")) {
    views.push({
      key: "h100",
      label: gpuLabels.h100 ?? "H100",
      ...mk("h100"),
    })
  }

  const hasRuns = views.some((v) => (v.bars?.rows.length ?? 0) > 0)

  return (
    <div className="space-y-6">
      <PageHead
        kicker="Benchmark · CUDA"
        title="The CUDA-only deck"
        sub={
          <>
            Hard&apos;s isolated sibling with a language gate: Triton and
            kernel DSLs fail. Four problems — GLM-5.2 fused MoE, DeepSeek NSA,
            MegaQwen decode at 2k–128k, grid+MinGRU SPS. Primary board is RTX
            PRO 6000; B200 cells use datasheet-validated peaks (bf16 2250 TF,
            fp8 4500 TF, HBM 8 TB/s).
            {!hasRuns && <strong> No runs yet.</strong>}
          </>
        }
        notes={
          <>
            <p>
              <strong>Why separate.</strong> Hard and Mega prompts are frozen
              lab boards. CUDA exists to force CUDA evidence and grade
              Triton/DSL cheats without moving their goalposts. Methodology and
              the full problem deck live in the{" "}
              <Link href={`${REPO_TREE}/SPEC.md`}>spec</Link>.
            </p>
            {hasRuns && (
              <p>
                <strong>Artifacts.</strong> Every published cell is
                contamination-checked and reward-hack audited; full agent
                traces are on HuggingFace. Browse the{" "}
                <Link href="/runs">run index</Link> for transcripts, submitted
                solutions, checks, and timing.
              </p>
            )}
          </>
        }
      />
      {hasRuns ? (
        <ModelGpuBoard
          views={views}
          initialGpu={gpu ?? "rtxpro6000"}
        />
      ) : (
        <p className="text-xs text-[var(--color-fg-muted)]">
          Results will populate here as sweeps complete.
        </p>
      )}
    </div>
  )
}
