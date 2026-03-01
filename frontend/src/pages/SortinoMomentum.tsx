import { useMutation, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import Plot from "react-plotly.js";

import KPICard from "@/components/cards/KPICard.tsx";
import SignalBadge from "@/components/cards/SignalBadge.tsx";
import RunButton from "@/components/controls/RunButton.tsx";
import AppLayout from "@/components/layout/AppLayout.tsx";
import BottomPanel from "@/components/layout/BottomPanel.tsx";
import LeftSidebar from "@/components/layout/LeftSidebar.tsx";
import RightSidebar from "@/components/layout/RightSidebar.tsx";
import { api } from "@/lib/api.ts";
import type {
  BootstrapResponse,
  GridSearchResponse,
  RegimeResponse,
} from "@/lib/types.ts";
import { cn, fmtRatio } from "@/lib/utils.ts";

type Tab = "heatmap" | "bootstrap";

const PLOTLY_LAYOUT_DEFAULTS: Partial<Plotly.Layout> = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "transparent",
  font: { color: "#a1a1aa", family: "JetBrains Mono, monospace", size: 11 },
  margin: { t: 36, r: 16, b: 44, l: 52 },
};

export default function SortinoMomentum() {
  const [symbol, setSymbol] = useState("GLD");
  const [selectedX, setSelectedX] = useState(10);
  const [selectedK, setSelectedK] = useState(10);
  const [activeTab, setActiveTab] = useState<Tab>("heatmap");
  const [symbolFilter, setSymbolFilter] = useState("");

  const assetsQuery = useQuery({ queryKey: ["assets"], queryFn: api.listAssets });
  const assets = assetsQuery.data?.assets ?? [];
  const filteredAssets = symbolFilter
    ? assets.filter((a) => a.symbol.includes(symbolFilter.toUpperCase()))
    : assets;

  const gridMut = useMutation({ mutationFn: () => api.getMomentumGrid(symbol) });
  const bootstrapMut = useMutation({
    mutationFn: () => api.getMomentumBootstrap(symbol, selectedX, selectedK),
  });
  const regimeMut = useMutation({
    mutationFn: () => api.getMomentumRegime(symbol, selectedX, selectedK),
  });

  const loading = gridMut.isPending || bootstrapMut.isPending || regimeMut.isPending;

  const handleRun = () => {
    gridMut.mutate();
    bootstrapMut.mutate();
    regimeMut.mutate();
  };

  return (
    <AppLayout
      left={
        <LeftSidebar>
          <Field label="Symbol">
            <input
              type="text"
              placeholder="Search..."
              value={symbolFilter}
              onChange={(e) => setSymbolFilter(e.target.value)}
              className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5 text-xs text-zinc-200 font-mono mb-1"
            />
            <select
              value={symbol}
              onChange={(e) => setSymbol(e.target.value)}
              size={6}
              className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-0.5 text-xs text-zinc-200 font-mono"
            >
              {filteredAssets.map((a) => (
                <option key={a.symbol} value={a.symbol}>
                  {a.symbol} ({a.type})
                </option>
              ))}
            </select>
          </Field>
          <Field label="Lookback (X days)">
            <input
              type="number"
              value={selectedX}
              onChange={(e) => setSelectedX(Number(e.target.value))}
              min={1}
              className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5 text-xs text-zinc-200 font-mono tabular-nums"
            />
          </Field>
          <Field label="Forecast (K days)">
            <input
              type="number"
              value={selectedK}
              onChange={(e) => setSelectedK(Number(e.target.value))}
              min={1}
              className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5 text-xs text-zinc-200 font-mono tabular-nums"
            />
          </Field>
          <RunButton onClick={handleRun} loading={loading} label="Run Analysis" />
          {gridMut.isError && (
            <div className="text-[10px] text-red-400 bg-red-950/50 border border-red-900 rounded px-2 py-1">
              {(gridMut.error as Error).message}
            </div>
          )}
        </LeftSidebar>
      }
      right={
        <RightSidebar>
          <RegimePanel regime={regimeMut.data} />
          <BootstrapKPIs data={bootstrapMut.data} />
          <BestParams grid={gridMut.data} />
        </RightSidebar>
      }
      bottom={gridMut.data ? <GridTable data={gridMut.data} /> : undefined}
    >
      {gridMut.data || bootstrapMut.data ? (
        <div className="flex flex-col h-full">
          <TabBar active={activeTab} onChange={setActiveTab} />
          <div className="flex-1 min-h-0">
            {activeTab === "heatmap" && gridMut.data && (
              <HeatmapChart data={gridMut.data} />
            )}
            {activeTab === "bootstrap" && bootstrapMut.data && (
              <BootstrapChart data={bootstrapMut.data} />
            )}
          </div>
        </div>
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
      <label className="text-[10px] uppercase tracking-wider text-zinc-500 mb-1 block">
        {label}
      </label>
      {children}
    </div>
  );
}

function TabBar({ active, onChange }: { active: Tab; onChange: (t: Tab) => void }) {
  const tabs: { id: Tab; label: string }[] = [
    { id: "heatmap", label: "Grid Search" },
    { id: "bootstrap", label: "Bootstrap Test" },
  ];
  return (
    <div className="flex gap-1 mb-4 border-b border-zinc-800 pb-2">
      {tabs.map((t) => (
        <button
          key={t.id}
          onClick={() => onChange(t.id)}
          className={cn(
            "px-3 py-1 text-xs font-medium rounded-t transition-colors cursor-pointer",
            active === t.id
              ? "bg-zinc-800 text-zinc-200 border-b-2 border-blue-500"
              : "text-zinc-500 hover:text-zinc-300",
          )}
        >
          {t.label}
        </button>
      ))}
    </div>
  );
}

function EmptyState() {
  return (
    <div className="flex items-center justify-center h-full">
      <div className="text-center">
        <div className="text-zinc-600 text-sm">Configure and run Sortino momentum analysis</div>
        <div className="text-zinc-700 text-xs mt-1">
          Grid search and bootstrap results will appear here
        </div>
      </div>
    </div>
  );
}

/* ── Charts ────────────────────────────────────────────────── */

function HeatmapChart({ data }: { data: GridSearchResponse }) {
  const rows = data.results;
  if (rows.length === 0) return null;

  const xVals = [...new Set(rows.map((r) => r["X (lookback)"]))].sort((a, b) => a - b);
  const yVals = [...new Set(rows.map((r) => r["K (forecast)"]))].sort((a, b) => a - b);

  const zGrid = yVals.map((k) =>
    xVals.map((x) => {
      const match = rows.find((r) => r["X (lookback)"] === x && r["K (forecast)"] === k);
      return match ? match["Z (hit_rate)"] : null;
    }),
  );

  return (
    <Plot
      data={[
        {
          type: "heatmap",
          x: xVals.map(String),
          y: yVals.map(String),
          z: zGrid,
          colorscale: [
            [0, "#18181b"],
            [0.5, "#3b82f6"],
            [1, "#22c55e"],
          ],
          colorbar: { title: { text: "Hit Rate %", side: "right" } as Partial<Plotly.ColorBar["title"]>, ticksuffix: "%" },
          hoverongaps: false,
          texttemplate: "%{z:.1f}%",
          textfont: { size: 10 },
        } as Partial<Plotly.PlotData>,
      ]}
      layout={{
        ...PLOTLY_LAYOUT_DEFAULTS,
        xaxis: { title: { text: "Lookback X (days)" } as Partial<Plotly.LayoutAxis["title"]>, color: "#71717a" },
        yaxis: { title: { text: "Forecast K (days)" } as Partial<Plotly.LayoutAxis["title"]>, color: "#71717a" },
        height: 420,
      }}
      config={{ displayModeBar: false, responsive: true }}
      className="w-full"
    />
  );
}

function BootstrapChart({ data }: { data: BootstrapResponse }) {
  const dist = data.bootstrap_dist;
  if (!dist || dist.length === 0) return null;

  return (
    <Plot
      data={[
        {
          type: "histogram",
          x: dist,
          nbinsx: 30,
          marker: { color: "#3b82f6", opacity: 0.7 },
          name: "Bootstrap",
        },
        {
          type: "scatter",
          x: [data.actual_hit_rate, data.actual_hit_rate],
          y: [0, dist.length / 5],
          mode: "lines",
          line: { color: "#22c55e", width: 2, dash: "dash" },
          name: `Actual ${data.actual_hit_rate.toFixed(1)}%`,
        },
      ]}
      layout={{
        ...PLOTLY_LAYOUT_DEFAULTS,
        xaxis: { title: { text: "Hit Rate (%)" } as Partial<Plotly.LayoutAxis["title"]>, color: "#71717a" },
        yaxis: { title: { text: "Frequency" } as Partial<Plotly.LayoutAxis["title"]>, color: "#71717a" },
        height: 420,
        showlegend: true,
        legend: { x: 0.02, y: 0.98, bgcolor: "transparent", font: { size: 10 } },
        shapes: [
          {
            type: "line",
            x0: data.random_mean,
            x1: data.random_mean,
            y0: 0,
            y1: 1,
            yref: "paper",
            line: { color: "#ef4444", width: 1.5, dash: "dot" },
          },
        ] as Partial<Plotly.Shape>[],
      }}
      config={{ displayModeBar: false, responsive: true }}
      className="w-full"
    />
  );
}

/* ── Right-rail panels ─────────────────────────────────────── */

function RegimePanel({ regime }: { regime: RegimeResponse | undefined }) {
  if (!regime?.regime)
    return <div className="text-xs text-zinc-600">Run analysis to see regime</div>;

  const r = regime.regime;
  return (
    <div className="flex flex-col gap-2">
      <div className="text-[10px] uppercase tracking-wider text-zinc-500">Current Regime</div>
      <SignalBadge signal={r.strong_momentum ? "long" : "flat"} />
      <KPICard label="Sortino" value={fmtRatio(r.current_sortino)} />
      <KPICard
        label="Recent Slope"
        value={fmtRatio(r.recent_slope, 4)}
        accent={r.recent_slope > 0 ? "positive" : "negative"}
      />
      <KPICard label="Baseline Slope" value={fmtRatio(r.baseline_slope, 4)} />
    </div>
  );
}

function BootstrapKPIs({ data }: { data: BootstrapResponse | undefined }) {
  if (!data) return null;
  return (
    <div className="flex flex-col gap-2 mt-3">
      <div className="text-[10px] uppercase tracking-wider text-zinc-500">Bootstrap Test</div>
      <KPICard
        label="Significant"
        value={data.significant ? "YES" : "NO"}
        accent={data.significant ? "positive" : "negative"}
      />
      <KPICard label="p-value" value={fmtRatio(data.p_value, 4)} />
      <KPICard label="Hit Rate" value={`${data.actual_hit_rate.toFixed(1)}%`} />
      <KPICard label="Signals" value={String(data.n_signals)} />
    </div>
  );
}

function BestParams({ grid }: { grid: GridSearchResponse | undefined }) {
  if (!grid || grid.results.length === 0) return null;
  const best = grid.results[0];
  return (
    <div className="flex flex-col gap-2 mt-3">
      <div className="text-[10px] uppercase tracking-wider text-zinc-500">Best (X, K)</div>
      <KPICard label="Lookback X" value={`${best["X (lookback)"]}d`} />
      <KPICard label="Forecast K" value={`${best["K (forecast)"]}d`} />
      <KPICard
        label="Hit Rate"
        value={`${best["Z (hit_rate)"].toFixed(1)}%`}
        accent={best["Z (hit_rate)"] > 55 ? "positive" : "neutral"}
      />
    </div>
  );
}

/* ── Bottom grid table ─────────────────────────────────────── */

function GridTable({ data }: { data: GridSearchResponse }) {
  if (data.results.length === 0) return null;
  return (
    <BottomPanel>
      <div className="overflow-x-auto">
        <table className="w-full text-[11px] font-mono">
          <thead>
            <tr className="text-zinc-500 border-b border-zinc-800">
              {["X", "K", "Hit Rate", "CI Low", "CI High", "Signals", "Success", "Failed"].map(
                (h) => (
                  <th key={h} className="text-left px-2 py-1 font-medium">
                    {h}
                  </th>
                ),
              )}
            </tr>
          </thead>
          <tbody>
            {data.results.map((r, i) => (
              <tr key={i} className="border-b border-zinc-900 hover:bg-zinc-900/50">
                <td className="px-2 py-1 tabular-nums">{r["X (lookback)"]}</td>
                <td className="px-2 py-1 tabular-nums">{r["K (forecast)"]}</td>
                <td className="px-2 py-1 tabular-nums text-blue-400">
                  {r["Z (hit_rate)"].toFixed(1)}%
                </td>
                <td className="px-2 py-1 tabular-nums">{r["CI_lower"].toFixed(1)}%</td>
                <td className="px-2 py-1 tabular-nums">{r["CI_upper"].toFixed(1)}%</td>
                <td className="px-2 py-1 tabular-nums">{r["Total_signals"]}</td>
                <td className="px-2 py-1 tabular-nums text-emerald-400">{r["Successful"]}</td>
                <td className="px-2 py-1 tabular-nums text-red-400">{r["Failed"]}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </BottomPanel>
  );
}
