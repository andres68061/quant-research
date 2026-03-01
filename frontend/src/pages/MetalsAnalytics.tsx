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
import type {
  CommodityReturnStats,
  CommodityReturnsResponse,
  CorrelationResponse,
  SeasonalityResponse,
} from "@/lib/types.ts";
import { cn, fmtPct, fmtRatio } from "@/lib/utils.ts";

const PLOTLY_LAYOUT: Partial<Plotly.Layout> = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "transparent",
  font: { color: "#a1a1aa", family: "JetBrains Mono, monospace", size: 11 },
  margin: { t: 28, r: 16, b: 44, l: 52 },
  xaxis: { gridcolor: "#27272a", zerolinecolor: "#3f3f46" },
  yaxis: { gridcolor: "#27272a", zerolinecolor: "#3f3f46" },
  legend: { x: 0.02, y: 0.98, bgcolor: "transparent", font: { size: 10 } },
};

const CATEGORY_LABELS: Record<string, string> = {
  precious_metals: "Precious Metals",
  energy: "Energy",
  industrial: "Industrial",
  agricultural: "Agricultural",
};

type Tab = "prices" | "returns" | "correlation" | "seasonality";

export default function MetalsAnalytics() {
  const [selected, setSelected] = useState<string[]>(["GLD", "SLV"]);
  const [activeTab, setActiveTab] = useState<Tab>("prices");
  const [seasonSymbol, setSeasonSymbol] = useState("GLD");

  const listQuery = useQuery({ queryKey: ["commodities"], queryFn: api.listCommodities });
  const commodities = listQuery.data?.commodities ?? [];

  const pricesMut = useMutation({
    mutationFn: () => api.getCommodityPrices(selected),
  });

  const returnsMut = useMutation({
    mutationFn: () => api.getCommodityReturns(selected),
  });

  const corrMut = useMutation({
    mutationFn: () => api.getCommodityCorrelation(selected),
  });

  const seasonMut = useMutation({
    mutationFn: () => api.getCommoditySeasonality(seasonSymbol),
  });

  const loading = pricesMut.isPending || returnsMut.isPending || corrMut.isPending || seasonMut.isPending;

  const handleRun = () => {
    if (selected.length === 0) return;
    pricesMut.mutate();
    returnsMut.mutate();
    if (selected.length >= 2) corrMut.mutate();
    seasonMut.mutate();
  };

  const toggleSymbol = (sym: string) => {
    setSelected((prev) =>
      prev.includes(sym) ? prev.filter((s) => s !== sym) : [...prev, sym],
    );
  };

  const grouped = commodities.reduce<Record<string, typeof commodities>>((acc, c) => {
    (acc[c.category] ??= []).push(c);
    return acc;
  }, {});

  const stats: CommodityReturnStats[] = returnsMut.data?.stats ?? [];

  return (
    <AppLayout
      left={
        <LeftSidebar>
          <Field label="Commodities">
            <div className="max-h-52 overflow-y-auto border border-zinc-800 rounded">
              {Object.entries(grouped).map(([cat, items]) => (
                <div key={cat}>
                  <div className="text-[9px] uppercase tracking-wider text-zinc-600 px-2 pt-1.5">
                    {CATEGORY_LABELS[cat] ?? cat}
                  </div>
                  {items.map((c) => (
                    <label
                      key={c.symbol}
                      className="flex items-center gap-1.5 px-2 py-0.5 text-[11px] font-mono text-zinc-300 hover:bg-zinc-800/50 cursor-pointer"
                    >
                      <input
                        type="checkbox"
                        checked={selected.includes(c.symbol)}
                        onChange={() => toggleSymbol(c.symbol)}
                        className="accent-blue-500"
                      />
                      {c.symbol}
                    </label>
                  ))}
                </div>
              ))}
            </div>
            <div className="text-[10px] text-zinc-600 mt-1">{selected.length} selected</div>
          </Field>

          <Field label="Seasonality Symbol">
            <select
              value={seasonSymbol}
              onChange={(e) => setSeasonSymbol(e.target.value)}
              className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5 text-xs text-zinc-200 font-mono"
            >
              {selected.map((s) => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
          </Field>

          <RunButton onClick={handleRun} loading={loading} disabled={selected.length === 0} label="Analyze" />

          {pricesMut.isError && (
            <div className="text-[10px] text-red-400 bg-red-950/50 border border-red-900 rounded px-2 py-1">
              {(pricesMut.error as Error).message}
            </div>
          )}
        </LeftSidebar>
      }
      right={
        <RightSidebar>
          {stats.length > 0 ? (
            <div className="flex flex-col gap-2">
              <div className="text-[10px] uppercase tracking-wider text-zinc-500">Summary</div>
              {stats.map((s) => (
                <div key={s.symbol} className="border-b border-zinc-800 pb-2 mb-1">
                  <div className="text-[11px] font-mono text-zinc-300 mb-1">{s.symbol}</div>
                  <KPICard label="Price" value={`$${s.latest_price.toFixed(2)}`} />
                  <KPICard label="Ann. Return" value={fmtPct(s.annualized)} accent={s.annualized >= 0 ? "positive" : "negative"} />
                  <KPICard label="Sharpe" value={fmtRatio(s.sharpe)} />
                </div>
              ))}
            </div>
          ) : (
            <div className="text-xs text-zinc-600">Run analysis to see stats</div>
          )}
        </RightSidebar>
      }
      bottom={stats.length > 0 ? <StatsTable stats={stats} /> : undefined}
    >
      {pricesMut.data || returnsMut.data || corrMut.data || seasonMut.data ? (
        <div className="flex flex-col h-full">
          <TabBar active={activeTab} onChange={setActiveTab} />
          <div className="flex-1 min-h-0">
            {activeTab === "prices" && pricesMut.data && (
              <PriceChart data={pricesMut.data.series} />
            )}
            {activeTab === "returns" && returnsMut.data && (
              <ReturnsChart data={returnsMut.data} />
            )}
            {activeTab === "correlation" && corrMut.data && (
              <CorrelationChart data={corrMut.data} />
            )}
            {activeTab === "seasonality" && seasonMut.data && (
              <SeasonalityChart data={seasonMut.data} />
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
      <label className="text-[10px] uppercase tracking-wider text-zinc-500 mb-1 block">{label}</label>
      {children}
    </div>
  );
}

function TabBar({ active, onChange }: { active: Tab; onChange: (t: Tab) => void }) {
  const tabs: { id: Tab; label: string }[] = [
    { id: "prices", label: "Prices" },
    { id: "returns", label: "Returns" },
    { id: "correlation", label: "Correlation" },
    { id: "seasonality", label: "Seasonality" },
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
        <div className="text-zinc-600 text-sm">Select commodities and click Analyze</div>
        <div className="text-zinc-700 text-xs mt-1">Price trends, returns, and correlation will appear here</div>
      </div>
    </div>
  );
}

/* ── Charts ────────────────────────────────────────────────── */

const COLORS = ["#3b82f6", "#22c55e", "#f59e0b", "#ef4444", "#a855f7", "#06b6d4", "#f43f5e", "#84cc16"];

function PriceChart({ data }: { data: Record<string, { date: string; price: number }[]> }) {
  const traces = Object.entries(data).map(([sym, points], i) => ({
    type: "scatter" as const,
    x: points.map((p) => p.date),
    y: points.map((p) => p.price),
    mode: "lines" as const,
    name: sym,
    line: { color: COLORS[i % COLORS.length], width: 1.5 },
  }));

  return (
    <Plot
      data={traces}
      layout={{ ...PLOTLY_LAYOUT, height: 420, showlegend: true }}
      config={{ displayModeBar: false, responsive: true }}
      className="w-full"
    />
  );
}

function ReturnsChart({ data }: { data: CommodityReturnsResponse }) {
  const traces = data.stats.map((s, i) => ({
    type: "histogram" as const,
    x: data.series.map((row) => row[s.symbol] as number).filter((v) => v != null),
    name: s.symbol,
    opacity: 0.6,
    marker: { color: COLORS[i % COLORS.length] },
  }));

  return (
    <Plot
      data={traces}
      layout={{
        ...PLOTLY_LAYOUT,
        height: 420,
        barmode: "overlay",
        xaxis: { ...PLOTLY_LAYOUT.xaxis, title: { text: "Daily Return" } as Partial<Plotly.LayoutAxis["title"]> },
        showlegend: true,
      }}
      config={{ displayModeBar: false, responsive: true }}
      className="w-full"
    />
  );
}

function CorrelationChart({ data }: { data: CorrelationResponse }) {
  return (
    <Plot
      data={[
        {
          type: "heatmap",
          x: data.symbols,
          y: data.symbols,
          z: data.matrix,
          colorscale: [
            [0, "#ef4444"],
            [0.5, "#18181b"],
            [1, "#22c55e"],
          ],
          zmin: -1,
          zmax: 1,
          texttemplate: "%{z:.2f}",
          textfont: { size: 10 },
        } as Partial<Plotly.PlotData>,
      ]}
      layout={{ ...PLOTLY_LAYOUT, height: 420 }}
      config={{ displayModeBar: false, responsive: true }}
      className="w-full"
    />
  );
}

function SeasonalityChart({ data }: { data: SeasonalityResponse }) {
  const months = data.monthly_avg.map((m) => m.month);
  const avg = data.monthly_avg.map((m) => m.avg_return * 100);

  return (
    <Plot
      data={[
        {
          type: "bar",
          x: months.map((m) => ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][m - 1]),
          y: avg,
          marker: {
            color: avg.map((v) => (v >= 0 ? "#22c55e" : "#ef4444")),
          },
          name: data.symbol,
        },
      ]}
      layout={{
        ...PLOTLY_LAYOUT,
        height: 420,
        yaxis: {
          ...PLOTLY_LAYOUT.yaxis,
          title: { text: "Avg Monthly Return (%)" } as Partial<Plotly.LayoutAxis["title"]>,
          ticksuffix: "%",
        },
      }}
      config={{ displayModeBar: false, responsive: true }}
      className="w-full"
    />
  );
}

/* ── Bottom panel ──────────────────────────────────────────── */

function StatsTable({ stats }: { stats: CommodityReturnStats[] }) {
  return (
    <BottomPanel>
      <div className="overflow-x-auto">
        <table className="w-full text-[11px] font-mono">
          <thead>
            <tr className="text-zinc-500 border-b border-zinc-800">
              {["Symbol", "Price", "Ann. Return", "Volatility", "Sharpe", "Skew", "Kurtosis"].map((h) => (
                <th key={h} className="text-left px-2 py-1 font-medium">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {stats.map((s) => (
              <tr key={s.symbol} className="border-b border-zinc-900 hover:bg-zinc-900/50">
                <td className="px-2 py-1 text-zinc-300">{s.symbol}</td>
                <td className="px-2 py-1 tabular-nums">${s.latest_price.toFixed(2)}</td>
                <td className={cn("px-2 py-1 tabular-nums", s.annualized >= 0 ? "text-emerald-400" : "text-red-400")}>
                  {fmtPct(s.annualized)}
                </td>
                <td className="px-2 py-1 tabular-nums">{fmtPct(s.volatility)}</td>
                <td className="px-2 py-1 tabular-nums">{fmtRatio(s.sharpe)}</td>
                <td className="px-2 py-1 tabular-nums">{fmtRatio(s.skew)}</td>
                <td className="px-2 py-1 tabular-nums">{fmtRatio(s.kurtosis)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </BottomPanel>
  );
}
