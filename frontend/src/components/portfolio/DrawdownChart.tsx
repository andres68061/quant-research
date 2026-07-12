/**
 * Drawdown chart with optional benchmark overlay.
 *
 * Computes drawdowns from NAV (and benchmark cumulative returns) using the
 * pure helpers in ``lib/portfolio/drawdown``.
 */
import Plot from "react-plotly.js";

import type { BenchmarkResponse, SimulateResponse } from "@/lib/types.ts";
import { DARK_PLOTLY_LAYOUT } from "@/lib/plotlyTheme.ts";
import { drawdownFromCum, drawdownFromNav } from "@/lib/portfolio/drawdown.ts";

interface Props {
  sim: SimulateResponse;
  benchmark?: BenchmarkResponse;
  height?: number;
}

export default function DrawdownChart({ sim, benchmark, height = 420 }: Props) {
  const portfDD = drawdownFromNav(sim.nav);
  const benchDD = benchmark
    ? drawdownFromCum(benchmark.dates, benchmark.cumulative_returns)
    : [];

  const data: Plotly.Data[] = [
    {
      x: portfDD.map((p) => p.date),
      y: portfDD.map((p) => p.dd),
      type: "scatter",
      mode: "lines",
      fill: "tozeroy",
      name: "Portfolio DD",
      line: { color: "#ef4444", width: 1 },
      fillcolor: "rgba(239,68,68,0.15)",
    },
  ];

  if (benchmark) {
    data.push({
      x: benchDD.map((p) => p.date),
      y: benchDD.map((p) => p.dd),
      type: "scatter",
      mode: "lines",
      name: `${benchmark.benchmark_name} DD`,
      line: { color: "#a1a1aa", width: 1, dash: "dot" },
    });
  }

  return (
    <Plot
      data={data}
      layout={{
        ...DARK_PLOTLY_LAYOUT,
        yaxis: {
          ...DARK_PLOTLY_LAYOUT.yaxis,
          title: { text: "Drawdown", font: { size: 10, color: "#71717a" } } as Partial<
            Plotly.LayoutAxis["title"]
          >,
          tickformat: ".0%",
        },
        showlegend: true,
      }}
      config={{ displayModeBar: false, responsive: true }}
      className="w-full"
      style={{ height }}
    />
  );
}
