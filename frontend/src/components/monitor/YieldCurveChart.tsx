import Plot from "react-plotly.js";

import type { CurveShapePoint, CurveSnapshot } from "@/lib/types.ts";

import { BLUE, EMERALD, MONITOR_LAYOUT, PLOT_CONFIG, RED, ZINC } from "./plotTheme.ts";

interface SnapshotProps {
  snapshots: CurveSnapshot[];
  height?: number;
}

const SNAPSHOT_COLORS = [BLUE, "rgba(161,161,170,0.9)", "rgba(161,161,170,0.6)", "rgba(161,161,170,0.4)", "rgba(161,161,170,0.25)"];

/** The term structure on several dates; the most recent drawn in blue. */
export function YieldCurveSnapshots({ snapshots, height = 300 }: SnapshotProps) {
  return (
    <Plot
      data={snapshots.map((s, i) => ({
        x: s.points.map((p) => p.tenor_years),
        y: s.points.map((p) => p.yield),
        type: "scatter",
        mode: "lines+markers",
        name: s.date,
        line: { color: SNAPSHOT_COLORS[i % SNAPSHOT_COLORS.length], width: i === 0 ? 2 : 1 },
        marker: { size: i === 0 ? 5 : 3 },
        hovertemplate: `${s.date}<br>%{x}y · %{y:.2f}%<extra></extra>`,
      }))}
      layout={{
        ...MONITOR_LAYOUT,
        height,
        showlegend: true,
        legend: { orientation: "h", font: { size: 9 }, y: -0.2 },
        xaxis: {
          ...MONITOR_LAYOUT.xaxis,
          type: "log",
          title: { text: "tenor (years, log)", font: { size: 9 } },
          tickvals: [0.083, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30],
          ticktext: ["1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"],
        },
        yaxis: { ...MONITOR_LAYOUT.yaxis, title: { text: "yield %", font: { size: 9 } } },
      }}
      config={PLOT_CONFIG}
      useResizeHandler
      style={{ width: "100%" }}
    />
  );
}

interface ShapeProps {
  history: CurveShapePoint[];
  height?: number;
}

/** Spreads through time; a spread below zero (inversion) is shaded. */
export function CurveShapeHistory({ history, height = 260 }: ShapeProps) {
  const x = history.map((p) => p.date);
  return (
    <Plot
      data={[
        {
          x,
          y: history.map((p) => p.spread_2s10s ?? null),
          type: "scatter",
          mode: "lines",
          name: "2s10s",
          line: { color: BLUE, width: 1.3 },
          hovertemplate: "%{x}<br>2s10s %{y:.2f}<extra></extra>",
        },
        {
          x,
          y: history.map((p) => p.spread_3m10y ?? null),
          type: "scatter",
          mode: "lines",
          name: "3m10y",
          line: { color: EMERALD, width: 1 },
          hovertemplate: "%{x}<br>3m10y %{y:.2f}<extra></extra>",
        },
        {
          x,
          y: history.map((p) => p.butterfly_2_5_10 ?? null),
          type: "scatter",
          mode: "lines",
          name: "2-5-10 fly",
          line: { color: ZINC, width: 1, dash: "dot" },
          hovertemplate: "%{x}<br>fly %{y:.2f}<extra></extra>",
        },
      ]}
      layout={{
        ...MONITOR_LAYOUT,
        height,
        showlegend: true,
        legend: { orientation: "h", font: { size: 9 }, y: -0.2 },
        yaxis: {
          ...MONITOR_LAYOUT.yaxis,
          zeroline: true,
          zerolinecolor: RED,
          zerolinewidth: 1,
          title: { text: "pct points", font: { size: 9 } },
        },
      }}
      config={PLOT_CONFIG}
      useResizeHandler
      style={{ width: "100%" }}
    />
  );
}
