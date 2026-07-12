/** Shared dark-terminal Plotly layout used across portfolio / analytics pages. */
export const DARK_PLOTLY_LAYOUT: Partial<Plotly.Layout> = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "transparent",
  font: { color: "#a1a1aa", family: "JetBrains Mono, monospace", size: 11 },
  margin: { t: 28, r: 16, b: 44, l: 52 },
  xaxis: { gridcolor: "#27272a", zerolinecolor: "#3f3f46" },
  yaxis: { gridcolor: "#27272a", zerolinecolor: "#3f3f46" },
  legend: { x: 0.02, y: 0.98, bgcolor: "transparent", font: { size: 10 } },
};
