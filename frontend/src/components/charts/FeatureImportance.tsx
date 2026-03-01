import Plot from "react-plotly.js";

import type { FeatureImportanceItem } from "@/lib/types.ts";

interface Props {
  data: FeatureImportanceItem[];
  topN?: number;
  height?: number;
}

export default function FeatureImportance({ data, topN = 15, height = 360 }: Props) {
  const sorted = [...data].sort((a, b) => a.importance - b.importance).slice(-topN);

  return (
    <Plot
      data={[
        {
          x: sorted.map((d) => d.importance),
          y: sorted.map((d) => d.feature),
          type: "bar",
          orientation: "h",
          marker: { color: "#3b82f6" },
          hovertemplate: "%{y}: %{x:.4f}<extra></extra>",
        },
      ]}
      layout={{
        paper_bgcolor: "transparent",
        plot_bgcolor: "transparent",
        font: { color: "#a1a1aa", size: 10, family: "JetBrains Mono, monospace" },
        title: { text: `Top ${sorted.length} Features`, font: { size: 12, color: "#71717a" }, x: 0 } as Partial<Plotly.Layout["title"]>,
        xaxis: { gridcolor: "#27272a", linecolor: "#27272a", zeroline: false },
        yaxis: { gridcolor: "#27272a", linecolor: "#27272a" },
        margin: { l: 140, r: 16, t: 32, b: 32 },
        height,
        autosize: true,
      }}
      config={{ displayModeBar: false, responsive: true }}
      className="w-full"
      useResizeHandler
      style={{ width: "100%" }}
    />
  );
}
