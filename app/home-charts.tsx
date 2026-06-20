import type { ChartData } from "@/lib/charts"

// Horizontal grouped bars, server-rendered, themed. One row per model, one bar
// per GPU. No client JS — pure CSS widths driven by the data.
export function GroupedBars({
  title,
  subtitle,
  data,
}: {
  title: string
  subtitle: string
  data: ChartData
}) {
  const { series, groups, max, format } = data
  return (
    <div className="chart">
      <div className="chart-head">
        <div>
          <h3 className="chart-title">{title}</h3>
          <p className="chart-subtitle">{subtitle}</p>
        </div>
        <ul className="chart-legend">
          {series.map((s) => (
            <li key={s.key}>
              <span className="chart-swatch" style={{ background: s.color }} />
              {s.key}
            </li>
          ))}
        </ul>
      </div>
      <div className="chart-body">
        {groups.map((g) => (
          <div key={g.label} className="chart-row">
            <div className="chart-row-label">{g.label}</div>
            <div className="chart-row-bars">
              {g.values.map((v, i) => (
                <div key={series[i].key} className="chart-track">
                  <div className="chart-bar-wrap">
                    <div
                      className="chart-bar"
                      style={{
                        width: v == null ? "0%" : `${Math.max((v / max) * 100, 1.5)}%`,
                        background: series[i].color,
                      }}
                    />
                  </div>
                  <span className={v == null ? "chart-val chart-val-dnf" : "chart-val"}>
                    {v == null ? "DNF" : format(v)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
