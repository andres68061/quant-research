import Plot from "react-plotly.js";

import type { EquityCurvePoint } from "@/lib/types.ts";

interface Props {
  data: EquityCurvePoint[];
  title?: string;
  height?: number;
}

const dark = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "transparent",
  font: { color: "#a1a1aa", size: 11, family: "JetBrains Mono, monospace" },
  xaxis: { gridcolor: "#27272a", linecolor: "#27272a", zeroline: false },
  yaxis: {
    gridcolor: "#27272a",
    linecolor: "#27272a",
    zeroline: true,
    zerolinecolor: "#3f3f46",
    tickformat: ".1%",
  },
  margin: { l: 56, r: 16, t: 32, b: 32 },
};

export default function EquityCurve({ data, title = "Equity Curve", height = 340 }: Props) {
  return (
    <Plot
      data={[
        {
          x: data.map((p) => p.date),
          y: data.map((p) => p.cumulative_return),
          type: "scatter",
          mode: "lines",
          line: { color: "#3b82f6", width: 1.5 },
          fill: "tozeroy",
          fillcolor: "rgba(59,130,246,0.06)",
          hovertemplate: "%{x}<br>%{y:.2%}<extra></extra>",
        },
      ]}
      layout={{
        ...dark,
        title: { text: title, font: { size: 12, color: "#71717a" }, x: 0 } as Partial<Plotly.Layout["title"]>,
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
