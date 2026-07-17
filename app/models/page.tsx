import { columnOrder, columnsForBench, columnsForCorrectness } from "@/app/_lib/models"
import { loadModelIndex } from "@/app/_lib/models.server"
import { ModelScoreboards } from "@/app/_components/model-columns"
import { PageHead } from "@/app/_components/page-head"

// All models, AA-style vertical column charts: one performance chart per
// published bench, Multi (coming soon), then the compiled correctness chart.

export const metadata = { title: "models · kernelbench" }

export default async function ModelsPage() {
  const idx = await loadModelIndex()
  const ordered = columnOrder(idx)
  const perfCharts = (["mega", "hard", "cuda"] as const).map((b) =>
    columnsForBench(idx, b, ordered),
  )
  const correctnessChart = columnsForCorrectness(idx, ordered)
  return (
    <div className="space-y-6">
      <PageHead
        kicker="Index"
        title="Models"
        sub={
          <>
            Every ranked model across the decks — performance per bench,
            compiled correctness, and a full integrity record one click deep.
          </>
        }
      />
      <ModelScoreboards perf={perfCharts} correctness={correctnessChart} />
    </div>
  )
}
