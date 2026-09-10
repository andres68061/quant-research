import Plot from "react-plotly.js";

import type { SeasonalityTable } from "@/lib/types.ts";

import { MONITOR_LAYOUT, PLOT_CONFIG } from "./plotTheme.ts";

const DIVERGING: [number, string][] = [
  [0, "#f87171"],
  [0.5, "#18181b"],
  [1, "#34d399"],
];

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

interface Props {
  table: SeasonalityTable;
  isReturn: boolean;
  height?: number;
}

/** Year × month heatmap plus the per-month mean with n and hit-rate underneath. */
export default function SeasonalityHeatmap({ table, isReturn, height = 380 }: Props) {
  const { years, matrix, month_stats } = table;
  const fmt = isReturn ? ".1%" : ".2f";
  const absMax = Math.max(
    1e-9,
    ...matrix.flat().map((v) => (v === null ? 0 : Math.abs(v))),
  );
  return (
    <div className="grid grid-cols-1 gap-2">
      <Plot
        data={[
          {
            z: matrix,
            x: MONTHS,
            y: years.map(String),
            type: "heatmap",
            colorscale: DIVERGING,
            zmin: -absMax,
            zmax: absMax,
            showscale: false,
            hoverongaps: false,
            hovertemplate: `%{y} %{x}<br>%{z:${fmt}}<extra></extra>`,
            xgap: 1,
            ygap: 1,
          },
        ]}
        layout={{
          ...MONITOR_LAYOUT,
          height,
          margin: { t: 8, r: 12, b: 28, l: 48 },
          yaxis: { ...MONITOR_LAYOUT.yaxis, autorange: "reversed", dtick: 1, tickfont: { size: 8 } },
          xaxis: { ...MONITOR_LAYOUT.xaxis, side: "top" },
        }}
        config={PLOT_CONFIG}
        useResizeHandler
        style={{ width: "100%" }}
      />
      <div className="grid grid-cols-12 gap-px text-[9px] font-mono tabular-nums">
        {month_stats.map((m) => {
          const mean = m.mean;
          const cls =
            mean === null ? "text-zinc-600" : mean > 0 ? "text-emerald-400" : "text-red-400";
          return (
            <div key={m.month} className="bg-zinc-900 px-1 py-1 text-center">
              <div className={cls}>
                {mean === null ? "–" : isReturn ? `${(mean * 100).toFixed(1)}%` : mean.toFixed(2)}
              </div>
              <div className="text-zinc-500">
                {m.hit_rate === null ? `n=${m.n}` : `${(m.hit_rate * 100).toFixed(0)}% · n=${m.n}`}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
