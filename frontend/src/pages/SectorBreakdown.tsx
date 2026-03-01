import { useMutation, useQuery } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import Plot from "react-plotly.js";

import KPICard from "@/components/cards/KPICard.tsx";
import RunButton from "@/components/controls/RunButton.tsx";
import AppLayout from "@/components/layout/AppLayout.tsx";
import BottomPanel from "@/components/layout/BottomPanel.tsx";
import LeftSidebar from "@/components/layout/LeftSidebar.tsx";
import RightSidebar from "@/components/layout/RightSidebar.tsx";
import { api } from "@/lib/api.ts";
import type { SectorSummaryResponse, SectorSymbol } from "@/lib/types.ts";
import { cn } from "@/lib/utils.ts";

const PLOTLY_LAYOUT: Partial<Plotly.Layout> = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "transparent",
  font: { color: "#a1a1aa", family: "JetBrains Mono, monospace", size: 11 },
  margin: { t: 28, r: 16, b: 44, l: 120 },
};

type Tab = "treemap" | "bar";

export default function SectorBreakdown() {
  const [activeTab, setActiveTab] = useState<Tab>("treemap");
  const [sectorFilter, setSectorFilter] = useState<string>("");
  const [search, setSearch] = useState("");

  const summaryQuery = useQuery({
    queryKey: ["sector-summary"],
    queryFn: api.getSectorSummary,
  });

  const breakdownMut = useMutation({
    mutationFn: () => api.getSectorBreakdown(sectorFilter || undefined),
  });

  const summary: SectorSummaryResponse | undefined = summaryQuery.data;
  const symbols: SectorSymbol[] = breakdownMut.data?.symbols ?? [];

  const handleLoad = () => breakdownMut.mutate();

  const filteredSymbols = useMemo(() => {
    if (!search) return symbols;
    const q = search.toUpperCase();
    return symbols.filter(
      (s) =>
        s.symbol.includes(q) ||
        s.sector.toUpperCase().includes(q) ||
        s.industry.toUpperCase().includes(q),
    );
  }, [symbols, search]);

  const sectorList = summary?.sectors ?? [];
  const uniqueTypes = useMemo(() => {
    const types = new Set(symbols.map((s) => s.type));
    return [...types];
  }, [symbols]);

  return (
    <AppLayout
      left={
        <LeftSidebar>
          <Field label="Filter by Sector">
            <select
              value={sectorFilter}
              onChange={(e) => setSectorFilter(e.target.value)}
              className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5 text-xs text-zinc-200 font-mono"
            >
              <option value="">All Sectors</option>
              {sectorList.map((s) => (
                <option key={s.sector} value={s.sector}>
                  {s.sector} ({s.count})
                </option>
              ))}
            </select>
          </Field>

          <RunButton
            onClick={handleLoad}
            loading={breakdownMut.isPending}
            label="Load Breakdown"
          />

          {symbols.length > 0 && (
            <Field label="Search">
              <input
                type="text"
                placeholder="Symbol, sector..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5 text-xs text-zinc-200 font-mono"
              />
            </Field>
          )}
        </LeftSidebar>
      }
      right={
        <RightSidebar>
          {summary ? (
            <div className="flex flex-col gap-2">
              <div className="text-[10px] uppercase tracking-wider text-zinc-500">Universe</div>
              <KPICard label="Total Symbols" value={String(summary.total_symbols)} />
              <KPICard label="Sectors" value={String(sectorList.length)} />
              {uniqueTypes.length > 0 && (
                <KPICard label="Asset Types" value={uniqueTypes.join(", ")} />
              )}
            </div>
          ) : (
            <div className="text-xs text-zinc-600">Loading summary...</div>
          )}
        </RightSidebar>
      }
      bottom={
        filteredSymbols.length > 0 ? (
          <SymbolsTable symbols={filteredSymbols} />
        ) : undefined
      }
    >
      {sectorList.length > 0 ? (
        <div className="flex flex-col h-full">
          <TabBar active={activeTab} onChange={setActiveTab} />
          <div className="flex-1 min-h-0">
            {activeTab === "treemap" && <TreemapChart sectors={sectorList} />}
            {activeTab === "bar" && <BarChart sectors={sectorList} />}
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
      <label className="text-[10px] uppercase tracking-wider text-zinc-500 mb-1 block">{label}</label>
      {children}
    </div>
  );
}

function TabBar({ active, onChange }: { active: Tab; onChange: (t: Tab) => void }) {
  const tabs: { id: Tab; label: string }[] = [
    { id: "treemap", label: "Treemap" },
    { id: "bar", label: "Distribution" },
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
        <div className="text-zinc-600 text-sm">Sector breakdown of the investment universe</div>
        <div className="text-zinc-700 text-xs mt-1">Loading summary data...</div>
      </div>
    </div>
  );
}

/* ── Charts ────────────────────────────────────────────────── */

function TreemapChart({ sectors }: { sectors: SectorSummaryResponse["sectors"] }) {
  const labels = sectors.map((s) => s.sector);
  const values = sectors.map((s) => s.count);

  return (
    <Plot
      data={[
        {
          type: "treemap",
          labels,
          parents: labels.map(() => ""),
          values,
          textinfo: "label+value+percent root",
          textfont: { size: 11, family: "JetBrains Mono, monospace" },
          marker: {
            colorscale: [
              [0, "#1e3a5f"],
              [1, "#3b82f6"],
            ],
            line: { width: 1, color: "#27272a" },
          },
        } as Partial<Plotly.PlotData>,
      ]}
      layout={{ ...PLOTLY_LAYOUT, height: 460, margin: { t: 8, r: 8, b: 8, l: 8 } }}
      config={{ displayModeBar: false, responsive: true }}
      className="w-full"
    />
  );
}

function BarChart({ sectors }: { sectors: SectorSummaryResponse["sectors"] }) {
  const sorted = [...sectors].sort((a, b) => b.count - a.count);

  return (
    <Plot
      data={[
        {
          type: "bar",
          y: sorted.map((s) => s.sector),
          x: sorted.map((s) => s.count),
          orientation: "h",
          marker: { color: "#3b82f6" },
          text: sorted.map((s) => `${s.pct}%`),
          textposition: "outside",
          textfont: { size: 10 },
        },
      ]}
      layout={{
        ...PLOTLY_LAYOUT,
        height: 460,
        xaxis: {
          gridcolor: "#27272a",
          title: { text: "Symbol Count" } as Partial<Plotly.LayoutAxis["title"]>,
        },
      }}
      config={{ displayModeBar: false, responsive: true }}
      className="w-full"
    />
  );
}

/* ── Bottom panel ──────────────────────────────────────────── */

function SymbolsTable({ symbols }: { symbols: SectorSymbol[] }) {
  return (
    <BottomPanel>
      <div className="overflow-x-auto max-h-60 overflow-y-auto">
        <table className="w-full text-[11px] font-mono">
          <thead className="sticky top-0 bg-zinc-950">
            <tr className="text-zinc-500 border-b border-zinc-800">
              {["Symbol", "Sector", "Industry", "Type"].map((h) => (
                <th key={h} className="text-left px-2 py-1 font-medium">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {symbols.slice(0, 200).map((s) => (
              <tr key={s.symbol} className="border-b border-zinc-900 hover:bg-zinc-900/50">
                <td className="px-2 py-1 text-zinc-200">{s.symbol}</td>
                <td className="px-2 py-1 text-zinc-400">{s.sector}</td>
                <td className="px-2 py-1 text-zinc-500">{s.industry}</td>
                <td className="px-2 py-1 text-zinc-600">{s.type}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {symbols.length > 200 && (
          <div className="text-[10px] text-zinc-600 text-center py-1">
            Showing 200 of {symbols.length}
          </div>
        )}
      </div>
    </BottomPanel>
  );
}
