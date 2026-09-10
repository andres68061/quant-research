/** Shared Plotly layout for the Data Monitor: transparent, zinc axes, mono font. */
export const MONITOR_LAYOUT: Partial<Plotly.Layout> = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "transparent",
  font: { color: "#a1a1aa", family: "JetBrains Mono, monospace", size: 10 },
  margin: { t: 24, r: 12, b: 32, l: 52 },
  xaxis: { gridcolor: "#27272a", linecolor: "#27272a", zeroline: false },
  yaxis: { gridcolor: "#27272a", linecolor: "#27272a", zeroline: false },
  showlegend: false,
};

export const BLUE = "#3b82f6";
export const EMERALD = "#34d399";
export const RED = "#f87171";
export const ZINC = "#71717a";

export const PLOT_CONFIG = { displayModeBar: false, responsive: true } as const;
