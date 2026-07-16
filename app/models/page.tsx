import { barsForBench, rowsForIndex } from "@/app/_lib/models"
import { loadModelIndex } from "@/app/_lib/models.server"
import { ModelMetricChart } from "@/app/_components/model-metric-chart"
import { ModelList } from "@/app/_components/model-list"

// All models, AA-style bar charts across every published board, with the
// flagged/invalid sink below.

export const metadata = { title: "models · kernelbench" }

export default async function ModelsPage() {
  const idx = await loadModelIndex()
  const { sink } = rowsForIndex(idx)
  const barViews = {
    mega: barsForBench(idx, "mega"),
    hard: barsForBench(idx, "hard"),
    cuda: barsForBench(idx, "cuda"),
  }
  return (
    <div className="hard-page space-y-12">
      <section>
        <h1 className="text-3xl font-semibold tracking-tight text-[var(--color-fg-bright)] mb-3">
          models
        </h1>
        <p className="text-xs text-[var(--color-fg-muted)] max-w-4xl leading-relaxed">
          Frontier coding models on the kernel decks (Mega / Hard / CUDA): one
          bar per model, colored by lab. Correctness first: models group by
          valid passes (audited-clean correct cells), then order by score. The
          flagged count lists audited sessions that failed the reward-hack
          review (over all audited sessions for the model); it never changes
          the order. Models with no valid published results sit below the
          chart. Click any model for per-problem cells, audit chips, and its
          full integrity record.
        </p>
      </section>
      <section>
        <ModelMetricChart views={barViews} />
        {sink.length > 0 && (
          <div className="model-sink-section">
            <p className="model-sink-label">
              No valid published results — audited sessions below were flagged or invalid
            </p>
            <ModelList rows={sink} sink showBadges />
          </div>
        )}
      </section>
    </div>
  )
}
