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

type Tab = "treemap" | "bar" | "performance";

// Fixed sector -> color map (validated for the default six on the dark surface;
// colors follow the sector identity and never reassign when the selection
// changes). Legend + hover + final-value labels carry identity alongside color.
const SECTOR_COLORS: Record<string, string> = {
  Technology: "#2563eb",
  "Financial Services": "#d97706",
  Healthcare: "#0d9488",
  "Consumer Cyclical": "#7c3aed",
  Industrials: "#65a30d",
  Energy: "#db2777",
  "Communication Services": "#0284c7",
  "Consumer Defensive": "#ca8a04",
  Utilities: "#64748b",
  "Basic Materials": "#9333ea",
  "Real Estate": "#e11d48",
};
const DEFAULT_SECTORS = ["Technology", "Financial Services", "Healthcare", "Energy"];
const MAX_VISIBLE_SECTORS = 6;

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
          <div className="flex-1 min-h-0 overflow-y-auto">
            {activeTab === "treemap" && <TreemapChart sectors={sectorList} />}
            {activeTab === "bar" && <BarChart sectors={sectorList} />}
            {activeTab === "performance" && <SectorPerformancePanel />}
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
    { id: "performance", label: "Performance" },
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

function SectorPerformancePanel() {
  const [weighting, setWeighting] = useState<"cap" | "equal">("cap");
  const [sp500Only, setSp500Only] = useState(true);
  const [start, setStart] = useState("2000-01-03");
  const [visible, setVisible] = useState<string[]>(DEFAULT_SECTORS);

  const perfQuery = useQuery({
    queryKey: ["sector-performance", weighting, sp500Only, start],
    queryFn: () => api.getSectorPerformance({ weighting, sp500Only, start }),
    staleTime: 30 * 60 * 1000,
  });
  const data = perfQuery.data;

  const toggleSector = (sector: string) => {
    setVisible((current) =>
      current.includes(sector)
        ? current.filter((s) => s !== sector)
        : current.length >= MAX_VISIBLE_SECTORS
          ? current
          : [...current, sector],
    );
  };

  const shown = (data?.sectors ?? []).filter((s) => visible.includes(s.sector));

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-3 text-xs">
        <select
          value={weighting}
          onChange={(e) => setWeighting(e.target.value as "cap" | "equal")}
          className="bg-zinc-900 border border-zinc-800 rounded px-2 py-1 text-zinc-200 font-mono"
        >
          <option value="cap">Cap-weighted</option>
          <option value="equal">Equal-weighted</option>
        </select>
        <select
          value={start}
          onChange={(e) => setStart(e.target.value)}
          className="bg-zinc-900 border border-zinc-800 rounded px-2 py-1 text-zinc-200 font-mono"
        >
          <option value="2000-01-03">Since 2000</option>
          <option value="2010-01-04">Since 2010</option>
          <option value="2015-01-02">Since 2015</option>
          <option value="2020-01-02">Since 2020</option>
        </select>
        <label className="flex items-center gap-1.5 text-zinc-400 cursor-pointer">
          <input
            type="checkbox"
            checked={sp500Only}
            onChange={(e) => setSp500Only(e.target.checked)}
          />
          S&amp;P 500 members only (point-in-time)
        </label>
      </div>

      <div className="flex flex-wrap gap-1.5">
        {(data?.sectors ?? []).map((s) => {
          const active = visible.includes(s.sector);
          const capped = !active && visible.length >= MAX_VISIBLE_SECTORS;
          return (
            <button
              key={s.sector}
              onClick={() => toggleSector(s.sector)}
              disabled={capped}
              title={capped ? `Max ${MAX_VISIBLE_SECTORS} sectors at once` : undefined}
              className={cn(
                "flex items-center gap-1.5 px-2 py-1 rounded text-[11px] border transition-colors",
                active
                  ? "border-zinc-600 text-zinc-100 bg-zinc-800"
                  : "border-zinc-800 text-zinc-500 hover:text-zinc-300",
                capped && "opacity-40 cursor-not-allowed",
              )}
            >
              <span
                className="inline-block w-2 h-2 rounded-full"
                style={{ background: SECTOR_COLORS[s.sector] ?? "#71717a" }}
              />
              {s.sector}
              <span className="font-mono tabular-nums text-zinc-500">
                {s.ann_return_pct > 0 ? "+" : ""}
                {s.ann_return_pct}%/y
              </span>
            </button>
          );
        })}
      </div>

      {perfQuery.isLoading ? (
        <div className="text-xs text-zinc-500 py-8 text-center">
          Computing sector indices (first load takes ~20s)…
        </div>
      ) : null}
      {perfQuery.error ? (
        <div className="text-xs text-red-400 py-4">Failed to load sector performance.</div>
      ) : null}

      {shown.length > 0 ? (
        <Plot
          data={shown.map((s) => ({
            x: s.series.map((pt) => pt.date),
            y: s.series.map((pt) => pt.level),
            type: "scatter" as const,
            mode: "lines" as const,
            name: s.sector,
            line: { color: SECTOR_COLORS[s.sector] ?? "#71717a", width: 1.5 },
            hovertemplate: `${s.sector}<br>%{x}<br>%{y:.2f}x<extra></extra>`,
          }))}
          layout={{
            ...PLOTLY_LAYOUT,
            margin: { t: 8, r: 16, b: 40, l: 48 },
            height: 420,
            yaxis: {
              gridcolor: "#27272a",
              linecolor: "#27272a",
              type: "log",
              title: { text: "growth of $1 (log)", font: { size: 10 } },
            },
            xaxis: { gridcolor: "#27272a", linecolor: "#27272a" },
            legend: { orientation: "h", y: -0.12, font: { size: 10 } },
            hovermode: "x unified" as const,
          }}
          config={{ displayModeBar: false, responsive: true }}
          className="w-full"
          useResizeHandler
          style={{ width: "100%" }}
        />
      ) : null}

      {data ? (
        <div className="space-y-2">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-[11px]">
            <MethodFact label="Rebalance" value="Daily (continuous)" />
            <MethodFact
              label="Weighting"
              value={data.weighting === "cap" ? "Cap, prior-day" : "Equal, daily"}
            />
            <MethodFact
              label="Universe"
              value={`${data.methodology.universe_labeled_symbols.toLocaleString()} labeled${
                data.sp500_membership_filter ? ", S&P members only" : ""
              }`}
            />
            <MethodFact label="Costs" value="None (gross)" />
          </div>

          <details className="text-[11px] text-zinc-500 border border-zinc-800 rounded p-2">
            <summary className="cursor-pointer text-zinc-400">
              Full methodology &amp; {data.caveats.length} disclosed caveats
            </summary>
            <dl className="mt-2 space-y-1.5">
              <MethodRow label="Rebalance" detail={data.methodology.rebalance} />
              <MethodRow label="Membership" detail={data.methodology.membership} />
              <MethodRow label="Returns" detail={data.methodology.returns} />
              <MethodRow
                label="Minimum members"
                detail={`${data.methodology.min_members_per_day} per sector per day; below that the level holds flat`}
              />
              <MethodRow
                label="Handled biases"
                detail="Survivorship (delisted names contribute until their last traded day); weight lookahead (prior-day caps); entry/exit (daily membership)"
              />
            </dl>
            <div className="mt-3 space-y-2 border-t border-zinc-800 pt-2">
              {data.caveats.map((caveat) => (
                <div key={caveat.id}>
                  <div className="flex items-center gap-1.5">
                    <span
                      className={cn(
                        "px-1 py-0.5 rounded text-[9px] uppercase",
                        caveat.severity === "high"
                          ? "text-red-400 bg-red-950/40"
                          : caveat.severity === "medium"
                            ? "text-amber-400 bg-amber-950/30"
                            : "text-zinc-400 bg-zinc-800/60",
                      )}
                    >
                      {caveat.severity}
                    </span>
                    <span className="text-zinc-300">{caveat.title}</span>
                  </div>
                  <p className="text-zinc-500 mt-0.5 leading-relaxed">{caveat.detail}</p>
                </div>
              ))}
            </div>
          </details>
        </div>
      ) : null}
    </div>
  );
}

function MethodFact({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5">
      <div className="text-[9px] uppercase tracking-wider text-zinc-500">{label}</div>
      <div className="text-zinc-200 mt-0.5">{value}</div>
    </div>
  );
}

function MethodRow({ label, detail }: { label: string; detail: string }) {
  return (
    <div className="flex gap-2">
      <dt className="text-zinc-400 shrink-0 w-32">{label}</dt>
      <dd className="text-zinc-500 leading-relaxed">{detail}</dd>
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
