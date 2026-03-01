import Plot from "react-plotly.js";

import type { FoldResult } from "@/lib/types.ts";

interface Props {
  folds: FoldResult[];
  overallAccuracy?: number;
  height?: number;
}

export default function FoldAccuracy({ folds, overallAccuracy, height = 280 }: Props) {
  const labels = folds.map((f) => `F${f.fold}`);
  const accs = folds.map((f) => f.accuracy);

  const shapes: Partial<Plotly.Shape>[] = [];
  const annotations: Partial<Plotly.Annotations>[] = [];

  if (overallAccuracy != null) {
    shapes.push({
      type: "line", x0: -0.5, x1: folds.length - 0.5,
      y0: overallAccuracy, y1: overallAccuracy,
      line: { color: "#f4f4f5", dash: "dash", width: 1 },
    });
    annotations.push({
      x: folds.length - 1, y: overallAccuracy,
      text: `Overall ${(overallAccuracy * 100).toFixed(1)}%`,
      showarrow: false, font: { size: 10, color: "#a1a1aa" },
      xanchor: "right", yanchor: "bottom",
    });
  }

  shapes.push({
    type: "line", x0: -0.5, x1: folds.length - 0.5,
    y0: 0.5, y1: 0.5,
    line: { color: "#3f3f46", dash: "dot", width: 1 },
  });

  return (
    <Plot
      data={[
        {
          x: labels,
          y: accs,
          type: "bar",
          marker: { color: "#8b5cf6" },
          text: accs.map((a) => `${(a * 100).toFixed(1)}%`),
          textposition: "outside",
          textfont: { size: 9, color: "#a1a1aa" },
          hovertemplate: "%{x}<br>Accuracy: %{y:.1%}<extra></extra>",
        },
      ]}
      layout={{
        paper_bgcolor: "transparent",
        plot_bgcolor: "transparent",
        font: { color: "#a1a1aa", size: 10, family: "JetBrains Mono, monospace" },
        title: { text: "Fold Accuracy", font: { size: 12, color: "#71717a" }, x: 0 } as Partial<Plotly.Layout["title"]>,
        xaxis: { gridcolor: "#27272a" },
        yaxis: { gridcolor: "#27272a", range: [0, 1], tickformat: ".0%" },
        margin: { l: 44, r: 16, t: 32, b: 32 },
        shapes,
        annotations,
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
