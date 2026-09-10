import Plot from "react-plotly.js";

import type { MonitorHistogram } from "@/lib/types.ts";

import { BLUE, MONITOR_LAYOUT, PLOT_CONFIG, RED } from "./plotTheme.ts";

interface Props {
  histogram: MonitorHistogram;
  transformLabel: string;
  percentile: number | null;
  height?: number;
}

/** Histogram of the transformed series with the latest value marked. */
export default function DistributionHistogram({
  histogram,
  transformLabel,
  percentile,
  height = 260,
}: Props) {
  const { edges, counts, mark } = histogram;
  const centers = counts.map((_, i) => (edges[i] + edges[i + 1]) / 2);
  const width = edges.length > 1 ? edges[1] - edges[0] : 1;
  const shapes: Partial<Plotly.Shape>[] =
    mark === null
      ? []
      : [
          {
            type: "line",
            x0: mark,
            x1: mark,
            y0: 0,
            y1: 1,
            yref: "paper",
            line: { color: RED, width: 1.5 },
          },
        ];
  const annotations: Partial<Plotly.Annotations>[] =
    mark === null || percentile === null
      ? []
      : [
          {
            x: mark,
            y: 1,
            yref: "paper",
            text: `latest · p${percentile.toFixed(0)}`,
            showarrow: false,
            font: { color: RED, size: 9 },
            xanchor: mark > centers[Math.floor(centers.length / 2)] ? "right" : "left",
            yanchor: "top",
          },
        ];
  return (
    <Plot
      data={[
        {
          x: centers,
          y: counts,
          type: "bar",
          width,
          marker: { color: BLUE, opacity: 0.75, line: { width: 0 } },
          hovertemplate: "%{x:.4f}<br>n=%{y}<extra></extra>",
        },
      ]}
      layout={{
        ...MONITOR_LAYOUT,
        height,
        bargap: 0.05,
        shapes,
        annotations,
        xaxis: { ...MONITOR_LAYOUT.xaxis, title: { text: transformLabel, font: { size: 9 } } },
        yaxis: { ...MONITOR_LAYOUT.yaxis, title: { text: "count", font: { size: 9 } } },
      }}
      config={PLOT_CONFIG}
      useResizeHandler
      style={{ width: "100%" }}
    />
  );
}
