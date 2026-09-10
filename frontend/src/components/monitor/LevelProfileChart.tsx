import Plot from "react-plotly.js";

import type { MonitorLevelPoint } from "@/lib/types.ts";

import { BLUE, MONITOR_LAYOUT, PLOT_CONFIG, ZINC } from "./plotTheme.ts";

interface Props {
  points: MonitorLevelPoint[];
  unit: string;
  windowLabel: string;
  height?: number;
}

/** Level with its rolling mean and ±2σ band: "is today unusual against its own recent past?" */
export default function LevelProfileChart({ points, unit, windowLabel, height = 300 }: Props) {
  const x = points.map((p) => p.date);
  return (
    <Plot
      data={[
        {
          x,
          y: points.map((p) => p.upper),
          type: "scatter",
          mode: "lines",
          line: { width: 0 },
          hoverinfo: "skip",
          showlegend: false,
        },
        {
          x,
          y: points.map((p) => p.lower),
          type: "scatter",
          mode: "lines",
          line: { width: 0 },
          fill: "tonexty",
          fillcolor: "rgba(59,130,246,0.08)",
          hoverinfo: "skip",
          showlegend: false,
        },
        {
          x,
          y: points.map((p) => p.rolling_mean),
          type: "scatter",
          mode: "lines",
          line: { color: ZINC, width: 1, dash: "dot" },
          name: `rolling mean (${windowLabel})`,
          hovertemplate: "%{x}<br>mean %{y:.3f}<extra></extra>",
        },
        {
          x,
          y: points.map((p) => p.level),
          type: "scatter",
          mode: "lines",
          line: { color: BLUE, width: 1.4 },
          name: "level",
          hovertemplate: "%{x}<br>%{y:.3f}<extra></extra>",
        },
      ]}
      layout={{
        ...MONITOR_LAYOUT,
        height,
        yaxis: { ...MONITOR_LAYOUT.yaxis, title: { text: unit, font: { size: 9 } } },
      }}
      config={PLOT_CONFIG}
      useResizeHandler
      style={{ width: "100%" }}
    />
  );
}
