import { useMutation } from "@tanstack/react-query";
import { useState } from "react";
import Plot from "react-plotly.js";

import KPICard from "@/components/cards/KPICard.tsx";
import RunButton from "@/components/controls/RunButton.tsx";
import AppLayout from "@/components/layout/AppLayout.tsx";
import BottomPanel from "@/components/layout/BottomPanel.tsx";
import LeftSidebar from "@/components/layout/LeftSidebar.tsx";
import RightSidebar from "@/components/layout/RightSidebar.tsx";
import { api } from "@/lib/api.ts";
import type { ExclusionSummaryResponse, StockDetailResponse } from "@/lib/types.ts";
import { cn } from "@/lib/utils.ts";

type Tab = "summary" | "detail";

const PL: Partial<Plotly.Layout> = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "transparent",
  font: { color: "#a1a1aa", family: "JetBrains Mono, monospace", size: 11 },
  margin: { t: 36, r: 16, b: 44, l: 52 },
};

export default function ExcludedStocks() {
  const [threshold, setThreshold] = useState(5.0);
  const [startDate, setStartDate] = useState("2020-01-01");
  const [endDate, setEndDate] = useState("");
  const [activeTab, setActiveTab] = useState<Tab>("summary");
  const [selectedSymbol, setSelectedSymbol] = useState("");

  const summaryMut = useMutation({
    mutationFn: () => api.getExclusionSummary(threshold, startDate || undefined, endDate || undefined),
  });

  const detailMut = useMutation({
    mutationFn: (sym: string) =>
      api.getExclusionDetail(sym, threshold, startDate || undefined, endDate || undefined),
  });

  const summary = summaryMut.data as ExclusionSummaryResponse | undefined;
  const detail = detailMut.data as StockDetailResponse | undefined;

  const handleRun = () => summaryMut.mutate();

  const handleSelectSymbol = (sym: string) => {
    setSelectedSymbol(sym);
    setActiveTab("detail");
    detailMut.mutate(sym);
  };

  const excluded = summary?.stats ?? [];

  return (
    <AppLayout
      left={
        <LeftSidebar>
          <Field label="Price Threshold ($)">
            <input
              type="range"
              min={1}
              max={20}
              step={0.5}
              value={threshold}
              onChange={(e) => setThreshold(+e.target.value)}
              className="w-full accent-blue-500"
            />
            <span className="text-xs font-mono text-zinc-400">${threshold.toFixed(1)}</span>
          </Field>

          <Field label="Start Date">
            <input
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5 text-xs text-zinc-200 font-mono"
            />
          </Field>

          <Field label="End Date">
            <input
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5 text-xs text-zinc-200 font-mono"
            />
          </Field>

          <RunButton onClick={handleRun} loading={summaryMut.isPending} label="Analyze" />

          {excluded.length > 0 && (
            <div className="border-t border-zinc-800 pt-3 mt-1">
              <div className="text-[10px] uppercase tracking-wider text-zinc-500 mb-2">
                Inspect Stock
              </div>
              <select
                value={selectedSymbol}
                onChange={(e) => handleSelectSymbol(e.target.value)}
                className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5 text-xs text-zinc-200 font-mono"
                size={Math.min(excluded.length, 8)}
              >
                {excluded.map((s) => (
                  <option key={s.symbol} value={s.symbol}>
                    {s.symbol} (${s.min_price.toFixed(2)})
                  </option>
                ))}
              </select>
            </div>
          )}
        </LeftSidebar>
      }
      right={
        <RightSidebar>
          {summary ? (
            <>
              <KPICard label="Total Stocks" value={String(summary.total)} />
              <KPICard label="Valid" value={String(summary.valid)} accent="positive" />
              <KPICard label="Excluded" value={String(summary.excluded)} accent="negative" />
              <KPICard
                label="Exclusion Rate"
                value={`${((summary.excluded / summary.total) * 100).toFixed(1)}%`}
              />
            </>
          ) : (
            <p className="text-xs text-zinc-600">Run analysis to see metrics</p>
          )}

          {detail && (
            <>
              <div className="border-t border-zinc-800 pt-3 mt-1" />
              <KPICard label={`${detail.symbol} Min`} value={`$${detail.min_price.toFixed(2)}`} />
              <KPICard label="Max" value={`$${detail.max_price.toFixed(2)}`} />
              <KPICard label="Current" value={`$${detail.current_price.toFixed(2)}`} />
              <KPICard label="Ann. Vol" value={`${detail.annualized_vol.toFixed(1)}%`} />
            </>
          )}
        </RightSidebar>
      }
      bottom={
        excluded.length > 0 ? (
          <BottomPanel>
            <ExclusionsTable stats={excluded} onSelect={handleSelectSymbol} />
          </BottomPanel>
        ) : undefined
      }
    >
      {/* Tabs */}
      <div className="flex gap-2 mb-3">
        {(["summary", "detail"] as Tab[]).map((t) => (
          <button
            key={t}
            onClick={() => setActiveTab(t)}
            className={cn(
              "px-3 py-1 text-xs rounded font-medium capitalize transition-colors",
              activeTab === t
                ? "bg-zinc-800 text-zinc-100"
                : "text-zinc-500 hover:text-zinc-300",
            )}
          >
            {t}
          </button>
        ))}
      </div>

      {activeTab === "summary" ? (
        summary ? (
          <SummaryChart stats={excluded} threshold={threshold} />
        ) : (
          <EmptyState />
        )
      ) : detail ? (
        <DetailView detail={detail} threshold={threshold} loading={detailMut.isPending} />
      ) : (
        <p className="text-xs text-zinc-600 text-center mt-20">
          Select a stock from the left sidebar to inspect.
        </p>
      )}
    </AppLayout>
  );
}

/* ── Charts ─────────────────────────────────────────────────── */

function SummaryChart({
  stats,
  threshold,
}: {
  stats: ExclusionSummaryResponse["stats"];
  threshold: number;
}) {
  return (
    <Plot
      data={[
        {
          x: stats.map((s) => s.symbol),
          y: stats.map((s) => s.min_price),
          type: "bar" as const,
          marker: {
            color: stats.map((s) =>
              s.min_price < 1 ? "#ef4444" : s.pct_below > 50 ? "#f97316" : "#60a5fa",
            ),
          },
          name: "Min Price",
        },
      ]}
      layout={{
        ...PL,
        title: { text: "Minimum Prices of Excluded Stocks", font: { size: 13, color: "#d4d4d8" } } as Partial<Plotly.Layout["title"]>,
        xaxis: { title: { text: "Symbol" } as Partial<Plotly.LayoutAxis["title"]>, tickangle: -45 },
        yaxis: { title: { text: "Min Price ($)" } as Partial<Plotly.LayoutAxis["title"]>, gridcolor: "#27272a" },
        shapes: [
          {
            type: "line",
            y0: threshold,
            y1: threshold,
            x0: -0.5,
            x1: stats.length - 0.5,
            line: { dash: "dash", color: "#ef4444", width: 1.5 },
          } as Partial<Plotly.Shape>,
        ],
      }}
      useResizeHandler
      className="w-full h-full min-h-[400px]"
      config={{ displayModeBar: false }}
    />
  );
}

function DetailView({
  detail,
  threshold,
  loading,
}: {
  detail: StockDetailResponse;
  threshold: number;
  loading: boolean;
}) {
  if (loading) {
    return <p className="text-xs text-zinc-500 text-center mt-20">Loading detail...</p>;
  }

  const allPrices = detail.prices as { date: string; price: number; below: boolean }[];
  const belowPts = allPrices.filter((p) => p.below);

  return (
    <div className="space-y-4 h-full overflow-y-auto">
      <Plot
        data={[
          {
            x: allPrices.map((p) => p.date),
            y: allPrices.map((p) => p.price),
            type: "scatter" as const,
            mode: "lines" as const,
            name: "Price",
            line: { color: "#60a5fa", width: 2 },
          },
          ...(belowPts.length > 0
            ? [
                {
                  x: belowPts.map((p) => p.date),
                  y: belowPts.map((p) => p.price),
                  type: "scatter" as const,
                  mode: "markers" as const,
                  name: `Below $${threshold}`,
                  marker: { color: "#ef4444", size: 5, symbol: "x" as const },
                },
              ]
            : []),
        ]}
        layout={{
          ...PL,
          title: { text: `${detail.symbol} Price History`, font: { size: 13, color: "#d4d4d8" } } as Partial<Plotly.Layout["title"]>,
          yaxis: {
            title: { text: "Price ($)" } as Partial<Plotly.LayoutAxis["title"]>,
            gridcolor: "#27272a",
            type: detail.min_price < 1 ? "log" : "linear",
          },
          xaxis: { gridcolor: "#27272a" },
          showlegend: true,
          legend: { font: { size: 10 } },
          shapes: [
            {
              type: "line",
              y0: threshold,
              y1: threshold,
              x0: allPrices[0]?.date,
              x1: allPrices[allPrices.length - 1]?.date,
              line: { dash: "dash", color: "#ef4444", width: 1 },
            } as Partial<Plotly.Shape>,
          ],
        }}
        useResizeHandler
        className="w-full min-h-[350px]"
        config={{ displayModeBar: false }}
      />

      <div className="grid grid-cols-4 gap-3 text-xs font-mono">
        <Stat label="Days Below" value={String(detail.days_below)} />
        <Stat label="% Below" value={`${detail.pct_below.toFixed(1)}%`} />
        <Stat label="Max Gain" value={`${detail.max_daily_gain.toFixed(2)}%`} />
        <Stat label="Max Loss" value={`${detail.max_daily_loss.toFixed(2)}%`} />
        <Stat label="Extreme Gains" value={String(detail.extreme_gains)} />
        <Stat label="Extreme Losses" value={String(detail.extreme_losses)} />
        <Stat label="Ann. Vol" value={`${detail.annualized_vol.toFixed(1)}%`} />
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5">
      <div className="text-[9px] uppercase text-zinc-500">{label}</div>
      <div className="text-zinc-200">{value}</div>
    </div>
  );
}

/* ── Table ──────────────────────────────────────────────────── */

function ExclusionsTable({
  stats,
  onSelect,
}: {
  stats: ExclusionSummaryResponse["stats"];
  onSelect: (s: string) => void;
}) {
  return (
    <table className="text-xs font-mono w-full">
      <thead>
        <tr className="text-zinc-500 text-left">
          <th className="pr-3 pb-1">Symbol</th>
          <th className="pr-3 pb-1">Min $</th>
          <th className="pr-3 pb-1">Max $</th>
          <th className="pr-3 pb-1">Current $</th>
          <th className="pr-3 pb-1">Days Below</th>
          <th className="pr-3 pb-1">% Below</th>
        </tr>
      </thead>
      <tbody>
        {stats.map((s) => (
          <tr
            key={s.symbol}
            className="text-zinc-300 border-t border-zinc-800/50 cursor-pointer hover:bg-zinc-900/50"
            onClick={() => onSelect(s.symbol)}
          >
            <td className="pr-3 py-1 text-blue-400 font-semibold">{s.symbol}</td>
            <td className="pr-3 py-1">${s.min_price.toFixed(2)}</td>
            <td className="pr-3 py-1">${s.max_price.toFixed(2)}</td>
            <td className="pr-3 py-1">${s.current_price.toFixed(2)}</td>
            <td className="pr-3 py-1">{s.days_below}</td>
            <td className="pr-3 py-1">{s.pct_below.toFixed(1)}%</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/* ── Helpers ────────────────────────────────────────────────── */

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="block text-[10px] uppercase tracking-wider text-zinc-500 mb-1">
        {label}
      </label>
      {children}
    </div>
  );
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center h-full text-zinc-600 gap-4">
      <div className="text-lg font-semibold">Excluded Stocks Viewer</div>
      <p className="text-xs max-w-md text-center leading-relaxed">
        Identify stocks automatically excluded from portfolio simulations due to
        price filters. Inspect their price history and understand why they are flagged.
      </p>
      <p className="text-xs text-zinc-700">Set a threshold and click Analyze.</p>
    </div>
  );
}
