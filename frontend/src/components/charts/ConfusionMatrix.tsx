import Plot from "react-plotly.js";

import type { ConfusionMatrixResult } from "@/lib/types.ts";

interface Props {
  data: ConfusionMatrixResult;
  height?: number;
}

export default function ConfusionMatrix({ data, height = 320 }: Props) {
  const { true_negatives: tn, false_positives: fp, false_negatives: fn, true_positives: tp } = data;

  const z = [[tn, fp], [fn, tp]];
  const labels = ["Down (0)", "Up (1)"];

  return (
    <Plot
      data={[
        {
          z,
          x: labels,
          y: labels,
          type: "heatmap",
          texttemplate: "%{z}",
          colorscale: [
            [0, "#18181b"],
            [1, "#1d4ed8"],
          ],
          showscale: false,
          hovertemplate: "Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
        },
      ]}
      layout={{
        paper_bgcolor: "transparent",
        plot_bgcolor: "transparent",
        font: { color: "#a1a1aa", size: 11, family: "JetBrains Mono, monospace" },
        title: { text: "Confusion Matrix", font: { size: 12, color: "#71717a" }, x: 0 } as Partial<Plotly.Layout["title"]>,
        xaxis: { title: { text: "Predicted" }, side: "bottom", dtick: 1 },
        yaxis: { title: { text: "Actual" }, autorange: "reversed", dtick: 1 },
        margin: { l: 80, r: 16, t: 32, b: 56 },
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
