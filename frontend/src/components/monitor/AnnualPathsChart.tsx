import Plot from "react-plotly.js";

import type { AnnualPaths } from "@/lib/types.ts";

import { BLUE, MONITOR_LAYOUT, PLOT_CONFIG } from "./plotTheme.ts";

interface Props {
  paths: AnnualPaths;
  unit: string;
  height?: number;
}

/** Every year overlaid on a day-of-year axis; the current year drawn on top. */
export default function AnnualPathsChart({ paths, unit, height = 300 }: Props) {
  const years = paths.years;
  const latest = years[years.length - 1];
  const traces: Partial<Plotly.PlotData>[] = years.map((year) => {
    const pts = paths.paths[String(year)] ?? [];
    const isLatest = year === latest;
    const age = (latest - year) / Math.max(years.length - 1, 1);
    return {
      x: pts.map((p) => p.doy),
      y: pts.map((p) => p.value),
      type: "scatter",
      mode: "lines",
      name: String(year),
      line: {
        color: isLatest ? BLUE : `rgba(161,161,170,${(0.55 - 0.4 * age).toFixed(2)})`,
        width: isLatest ? 2 : 1,
      },
      hovertemplate: `${year} · day %{x}<br>%{y:.2f}<extra></extra>`,
    };
  });
  return (
    <Plot
      data={traces}
      layout={{
        ...MONITOR_LAYOUT,
        height,
        showlegend: true,
        legend: { orientation: "h", font: { size: 8 }, y: -0.18 },
        xaxis: { ...MONITOR_LAYOUT.xaxis, title: { text: "day of year", font: { size: 9 } }, range: [1, 366] },
        yaxis: {
          ...MONITOR_LAYOUT.yaxis,
          title: { text: paths.normalized ? "rebased to 100" : unit, font: { size: 9 } },
        },
      }}
      config={PLOT_CONFIG}
      useResizeHandler
      style={{ width: "100%" }}
    />
  );
}
