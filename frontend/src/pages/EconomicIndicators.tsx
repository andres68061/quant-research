import { useMutation, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import Plot from "react-plotly.js";

import KPICard from "@/components/cards/KPICard.tsx";
import RunButton from "@/components/controls/RunButton.tsx";
import AppLayout from "@/components/layout/AppLayout.tsx";
import BottomPanel from "@/components/layout/BottomPanel.tsx";
import LeftSidebar from "@/components/layout/LeftSidebar.tsx";
import RightSidebar from "@/components/layout/RightSidebar.tsx";
import { api } from "@/lib/api.ts";
import type { FREDSeriesResponse, RecessionPeriod } from "@/lib/types.ts";
import { cn, fmtRatio } from "@/lib/utils.ts";

const PLOTLY_LAYOUT: Partial<Plotly.Layout> = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "transparent",
  font: { color: "#a1a1aa", family: "JetBrains Mono, monospace", size: 11 },
  margin: { t: 28, r: 16, b: 36, l: 52 },
  xaxis: { gridcolor: "#27272a", zerolinecolor: "#3f3f46" },
  yaxis: { gridcolor: "#27272a", zerolinecolor: "#3f3f46" },
};

const COLORS = ["#3b82f6", "#22c55e", "#f59e0b", "#ef4444", "#a855f7", "#06b6d4"];

export default function EconomicIndicators() {
  const [selectedIds, setSelectedIds] = useState<string[]>(["DFF", "DGS10"]);
  const [period, setPeriod] = useState(10);
  const [showYoY, setShowYoY] = useState(false);
  const [showRecessions, setShowRecessions] = useState(true);

  const catalogQuery = useQuery({ queryKey: ["fred-catalog"], queryFn: api.getFREDCatalog });
  const categories = catalogQuery.data?.categories ?? [];

  const start = new Date();
  start.setFullYear(start.getFullYear() - period);
  const startStr = start.toISOString().split("T")[0];

  const seriesMut = useMutation({
    mutationFn: () => api.getFREDSeries(selectedIds, startStr, undefined, showYoY),
  });

  const recessionMut = useMutation({
    mutationFn: () => api.getRecessions(startStr),
  });

  const loading = seriesMut.isPending || recessionMut.isPending;
  const data: FREDSeriesResponse | undefined = seriesMut.data;
  const recessions: RecessionPeriod[] = recessionMut.data?.periods ?? [];

  const handleRun = () => {
    if (selectedIds.length === 0) return;
    seriesMut.mutate();
    if (showRecessions) recessionMut.mutate();
  };

  const toggleIndicator = (id: string) => {
    setSelectedIds((prev) =>
      prev.includes(id) ? prev.filter((s) => s !== id) : [...prev, id],
    );
  };

  const latestValues: { id: string; name: string; value: number | null; unit: string }[] =
    data
      ? selectedIds.map((id) => {
          const meta = data.metadata[id] ?? { name: id, unit: "" };
          const last = [...data.series].reverse().find((row) => row[id] != null);
          return {
            id,
            name: meta.name,
            value: last ? (last[id] as number) : null,
            unit: meta.unit,
          };
        })
      : [];

  return (
    <AppLayout
      left={
        <LeftSidebar>
          <Field label="Indicators">
            <div className="max-h-48 overflow-y-auto border border-zinc-800 rounded">
              {categories.map((cat) => (
                <div key={cat.category}>
                  <div className="text-[9px] uppercase tracking-wider text-zinc-600 px-2 pt-1.5">
                    {cat.category}
                  </div>
                  {cat.indicators.map((ind) => (
                    <label
                      key={ind.id}
                      className="flex items-center gap-1.5 px-2 py-0.5 text-[11px] font-mono text-zinc-300 hover:bg-zinc-800/50 cursor-pointer"
                    >
                      <input
                        type="checkbox"
                        checked={selectedIds.includes(ind.id)}
                        onChange={() => toggleIndicator(ind.id)}
                        className="accent-blue-500"
                      />
                      <span className="truncate">{ind.id} - {ind.name}</span>
                    </label>
                  ))}
                </div>
              ))}
            </div>
          </Field>

          <Field label={`Period: ${period} years`}>
            <input
              type="range"
              min={1}
              max={50}
              value={period}
              onChange={(e) => setPeriod(Number(e.target.value))}
              className="w-full accent-blue-500"
            />
          </Field>

          <label className="flex items-center gap-2 text-xs text-zinc-400 cursor-pointer">
            <input
              type="checkbox"
              checked={showYoY}
              onChange={(e) => setShowYoY(e.target.checked)}
              className="accent-blue-500"
            />
            YoY % Change
          </label>

          <label className="flex items-center gap-2 text-xs text-zinc-400 cursor-pointer">
            <input
              type="checkbox"
              checked={showRecessions}
              onChange={(e) => setShowRecessions(e.target.checked)}
              className="accent-blue-500"
            />
            Show Recessions
          </label>

          <RunButton onClick={handleRun} loading={loading} disabled={selectedIds.length === 0} label="Fetch Data" />

          {seriesMut.isError && (
            <div className="text-[10px] text-red-400 bg-red-950/50 border border-red-900 rounded px-2 py-1">
              {(seriesMut.error as Error).message}
            </div>
          )}
        </LeftSidebar>
      }
      right={
        <RightSidebar>
          {latestValues.length > 0 ? (
            <div className="flex flex-col gap-2">
              <div className="text-[10px] uppercase tracking-wider text-zinc-500">Latest Values</div>
              {latestValues.map((v) => (
                <KPICard
                  key={v.id}
                  label={v.name}
                  value={v.value != null ? fmtRatio(v.value, 2) + (v.unit === "%" ? "%" : "") : "N/A"}
                />
              ))}
            </div>
          ) : (
            <div className="text-xs text-zinc-600">Fetch data to see latest values</div>
          )}
        </RightSidebar>
      }
      bottom={data ? <StatsTable data={data} ids={selectedIds} /> : undefined}
    >
      {data ? (
        <IndicatorChart
          data={data}
          ids={selectedIds}
          recessions={showRecessions ? recessions : []}
        />
      ) : (
        <EmptyState />
      )}
    </AppLayout>
  );
}

/* ── Sub-components ────────────────────────────────────────── */

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="text-[10px] uppercase tracking-wider text-zinc-500 mb-1 block">{label}</label>
      {children}
    </div>
  );
}

function EmptyState() {
  return (
    <div className="flex items-center justify-center h-full">
      <div className="text-center">
        <div className="text-zinc-600 text-sm">Select indicators and click Fetch Data</div>
        <div className="text-zinc-700 text-xs mt-1">Charts with optional recession shading will appear here</div>
      </div>
    </div>
  );
}

/* ── Chart ─────────────────────────────────────────────────── */

function IndicatorChart({
  data,
  ids,
  recessions,
}: {
  data: FREDSeriesResponse;
  ids: string[];
  recessions: RecessionPeriod[];
}) {
  const dates = data.series.map((row) => row.date);

  const traces = ids.map((id, i) => ({
    type: "scatter" as const,
    x: dates,
    y: data.series.map((row) => row[id] as number | null),
    mode: "lines" as const,
    name: data.metadata[id]?.name ?? id,
    line: { color: COLORS[i % COLORS.length], width: 1.5 },
    yaxis: ids.length > 1 && i > 0 ? `y${i + 1}` : "y",
  }));

  const recShapes: Partial<Plotly.Shape>[] = recessions.map((r) => ({
    type: "rect",
    x0: r.start,
    x1: r.end,
    y0: 0,
    y1: 1,
    yref: "paper",
    fillcolor: "rgba(239,68,68,0.08)",
    line: { width: 0 },
    layer: "below",
  }));

  const layout: Partial<Plotly.Layout> = {
    ...PLOTLY_LAYOUT,
    height: 480,
    showlegend: true,
    legend: { x: 0.02, y: 0.98, bgcolor: "transparent", font: { size: 10 } },
    shapes: recShapes,
  };

  if (ids.length === 2) {
    layout.yaxis2 = {
      overlaying: "y",
      side: "right",
      gridcolor: "#27272a",
      showgrid: false,
    };
  }

  return (
    <Plot
      data={traces}
      layout={layout}
      config={{ displayModeBar: false, responsive: true }}
      className="w-full"
    />
  );
}

/* ── Bottom panel ──────────────────────────────────────────── */

function StatsTable({ data, ids }: { data: FREDSeriesResponse; ids: string[] }) {
  const stats = ids.map((id) => {
    const vals = data.series.map((r) => r[id] as number | null).filter((v): v is number => v != null);
    const last = vals.length > 0 ? vals[vals.length - 1] : null;
    const min = vals.length > 0 ? Math.min(...vals) : null;
    const max = vals.length > 0 ? Math.max(...vals) : null;
    const mean = vals.length > 0 ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
    const name = data.metadata[id]?.name ?? id;
    const unit = data.metadata[id]?.unit ?? "";
    return { id, name, unit, last, min, max, mean, count: vals.length };
  });

  return (
    <BottomPanel>
      <div className="overflow-x-auto">
        <table className="w-full text-[11px] font-mono">
          <thead>
            <tr className="text-zinc-500 border-b border-zinc-800">
              {["Indicator", "Unit", "Latest", "Min", "Max", "Mean", "Points"].map((h) => (
                <th key={h} className="text-left px-2 py-1 font-medium">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {stats.map((s) => (
              <tr key={s.id} className="border-b border-zinc-900 hover:bg-zinc-900/50">
                <td className="px-2 py-1 text-zinc-300">{s.name}</td>
                <td className="px-2 py-1 text-zinc-500">{s.unit}</td>
                <td className="px-2 py-1 tabular-nums">{s.last?.toFixed(2) ?? "—"}</td>
                <td className="px-2 py-1 tabular-nums">{s.min?.toFixed(2) ?? "—"}</td>
                <td className="px-2 py-1 tabular-nums">{s.max?.toFixed(2) ?? "—"}</td>
                <td className="px-2 py-1 tabular-nums">{s.mean?.toFixed(2) ?? "—"}</td>
                <td className="px-2 py-1 tabular-nums text-zinc-500">{s.count}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </BottomPanel>
  );
}
