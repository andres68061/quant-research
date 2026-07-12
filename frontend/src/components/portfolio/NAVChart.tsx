/**
 * Cumulative NAV chart with optional benchmark overlay.
 *
 * Both the ETF Optimizer and the Custom Portfolio page render the simulator
 * NAV the same way: a single blue line for the portfolio plus, optionally, a
 * dotted gray line for the benchmark's wealth index.
 */
import Plot from "react-plotly.js";

import type { BenchmarkResponse, SimulateResponse } from "@/lib/types.ts";
import { DARK_PLOTLY_LAYOUT } from "@/lib/plotlyTheme.ts";

interface Props {
  sim: SimulateResponse;
  benchmark?: BenchmarkResponse;
  /** Pixel height; defaults match prior pages. */
  height?: number;
}

export default function NAVChart({ sim, benchmark, height = 420 }: Props) {
  const data: Plotly.Data[] = [
    {
      x: sim.nav.map((p) => p.date),
      y: sim.nav.map((p) => p.value),
      type: "scatter",
      mode: "lines",
      name: "Portfolio",
      line: { color: "#3b82f6", width: 1.5 },
      fill: benchmark ? undefined : "tozeroy",
      fillcolor: benchmark ? undefined : "rgba(59,130,246,0.05)",
    },
  ];

  if (benchmark) {
    data.push({
      x: benchmark.dates,
      y: benchmark.cumulative_returns,
      type: "scatter",
      mode: "lines",
      name: benchmark.benchmark_name,
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
          title: { text: "NAV", font: { size: 10, color: "#71717a" } } as Partial<
            Plotly.LayoutAxis["title"]
          >,
        },
        showlegend: true,
      }}
      config={{ displayModeBar: false, responsive: true }}
      className="w-full"
      style={{ height }}
    />
  );
}
