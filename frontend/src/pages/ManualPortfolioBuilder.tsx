import { useMutation, useQuery } from "@tanstack/react-query";
import { useCallback, useMemo, useState } from "react";
import Plot from "react-plotly.js";

import KPICard from "@/components/cards/KPICard.tsx";
import RunButton from "@/components/controls/RunButton.tsx";
import AppLayout from "@/components/layout/AppLayout.tsx";
import BottomPanel from "@/components/layout/BottomPanel.tsx";
import LeftSidebar from "@/components/layout/LeftSidebar.tsx";
import RightSidebar from "@/components/layout/RightSidebar.tsx";
import { api } from "@/lib/api.ts";
import type {
  BenchmarkResponse,
  SectorBreakdownResponse,
  SectorSummaryResponse,
  SimulateResponse,
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

const BENCHMARK_TYPES = [
  "S&P 500 (^GSPC)",
  "S&P 500 Reconstructed (2020+)",
  "Equal Weight Universe",
  "Synthetic (Custom Mix)",
] as const;

const REBALANCE_OPTIONS = [
  { label: "Annual", value: "Annual" },
  { label: "Quarterly", value: "Quarterly" },
  { label: "Monthly", value: "Monthly" },
] as const;

type WeightScheme = "equal" | "manual";
type ActiveTab = "nav" | "drawdown";

export default function ManualPortfolioBuilder() {
  const [startDate, setStartDate] = useState("2020-01-01");
  const [endDate, setEndDate] = useState("2025-12-31");
  const [sectorFilter, setSectorFilter] = useState<string[]>([]);
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState<string[]>([]);
  const [weightScheme, setWeightScheme] = useState<WeightScheme>("equal");
  const [manualWeights, setManualWeights] = useState<Record<string, number>>({});
  const [rebalFreq, setRebalFreq] = useState("Annual");
  const [benchmarkType, setBenchmarkType] = useState<string>("S&P 500 (^GSPC)");
  const [activeTab, setActiveTab] = useState<ActiveTab>("nav");

  const sectorsQuery = useQuery({
    queryKey: ["sectorSummary"],
    queryFn: api.getSectorSummary,
  });
  const breakdownQuery = useQuery({
    queryKey: ["sectorBreakdown"],
    queryFn: () => api.getSectorBreakdown(),
  });

  const sectors: SectorSummaryResponse | undefined = sectorsQuery.data;
  const breakdown: SectorBreakdownResponse | undefined = breakdownQuery.data;

  const availableStocks = useMemo(() => {
    if (!breakdown?.symbols) return [];
    let stocks = breakdown.symbols;
    if (sectorFilter.length > 0) {
      stocks = stocks.filter((s) => sectorFilter.includes(s.sector));
    }
    if (search) {
      const q = search.toUpperCase();
      stocks = stocks.filter(
        (s) =>
          s.symbol.includes(q) ||
          s.sector.toUpperCase().includes(q) ||
          s.industry.toUpperCase().includes(q),
      );
    }
    return stocks;
  }, [breakdown, sectorFilter, search]);

  const computedWeights = useMemo(() => {
    if (selected.length === 0) return {};
    if (weightScheme === "equal") {
      const w = 1 / selected.length;
      return Object.fromEntries(selected.map((s) => [s, w]));
    }
    const raw: Record<string, number> = {};
    let total = 0;
    for (const sym of selected) {
      const v = manualWeights[sym] ?? 1;
      raw[sym] = v;
      total += v;
    }
    if (total === 0) return Object.fromEntries(selected.map((s) => [s, 0]));
    return Object.fromEntries(Object.entries(raw).map(([k, v]) => [k, v / total]));
  }, [selected, weightScheme, manualWeights]);

  const simMut = useMutation({
    mutationFn: () =>
      api.simulatePortfolio({
        symbols: selected,
        weights: computedWeights,
        freq: rebalFreq,
        start_date: startDate,
        end_date: endDate,
      }),
  });

  const benchMut = useMutation({
    mutationFn: () =>
      api.getBenchmarkReturns({
        benchmark_type: benchmarkType,
        start_date: startDate,
        end_date: endDate,
      }),
  });

  const handleRun = useCallback(() => {
    if (selected.length < 1) return;
    simMut.mutate();
    benchMut.mutate();
  }, [selected, simMut, benchMut]);

  const loading = simMut.isPending || benchMut.isPending;
  const sim: SimulateResponse | undefined = simMut.data;
  const bench: BenchmarkResponse | undefined = benchMut.data;

  const toggleSector = (sector: string) => {
    setSectorFilter((prev) =>
      prev.includes(sector) ? prev.filter((s) => s !== sector) : [...prev, sector],
    );
  };

  const addStock = (sym: string) => {
    if (!selected.includes(sym)) setSelected((prev) => [...prev, sym]);
  };

  const removeStock = (sym: string) => {
    setSelected((prev) => prev.filter((s) => s !== sym));
    setManualWeights((prev) => {
      const next = { ...prev };
      delete next[sym];
      return next;
    });
  };

  const drawdownFromNav = (nav: { date: string; value: number }[]) => {
    let peak = nav[0]?.value ?? 1;
    return nav.map((p) => {
      if (p.value > peak) peak = p.value;
      return { date: p.date, dd: (p.value - peak) / peak };
    });
  };

  const drawdownFromCum = (dates: string[], cum: number[]) => {
    let peak = cum[0] ?? 1;
    return dates.map((d, i) => {
      if (cum[i] > peak) peak = cum[i];
      return { date: d, dd: (cum[i] - peak) / peak };
    });
  };

  const portfDD = sim ? drawdownFromNav(sim.nav) : [];
  const benchDD = bench ? drawdownFromCum(bench.dates, bench.cumulative_returns) : [];

  const selectedStockInfo = useMemo(() => {
    if (!breakdown?.symbols) return [];
    return selected.map((sym) => {
      const info = breakdown.symbols.find((s) => s.symbol === sym);
      return {
        symbol: sym,
        sector: info?.sector ?? "—",
        industry: info?.industry ?? "—",
        weight: computedWeights[sym] ?? 0,
      };
    });
  }, [selected, breakdown, computedWeights]);

  /* ── Left sidebar ──────────────────────────────────────────── */
  const left = (
    <LeftSidebar>
      <Field label="Date Range">
        <div className="flex gap-2">
          <input
            type="date"
            value={startDate}
            onChange={(e) => setStartDate(e.target.value)}
            className="flex-1 bg-zinc-900 border border-zinc-700 rounded px-2 py-1 text-xs text-zinc-100 font-mono"
          />
          <input
            type="date"
            value={endDate}
            onChange={(e) => setEndDate(e.target.value)}
            className="flex-1 bg-zinc-900 border border-zinc-700 rounded px-2 py-1 text-xs text-zinc-100 font-mono"
          />
        </div>
      </Field>

      <Field label="Filter by Sector">
        <div className="max-h-32 overflow-y-auto space-y-1">
          {sectors?.sectors.map((s) => (
            <label
              key={s.sector}
              className="flex items-center gap-2 text-xs text-zinc-400 cursor-pointer hover:text-zinc-200"
            >
              <input
                type="checkbox"
                checked={sectorFilter.includes(s.sector)}
                onChange={() => toggleSector(s.sector)}
                className="accent-blue-500"
              />
              <span className="truncate">{s.sector}</span>
              <span className="ml-auto text-zinc-600">{s.count}</span>
            </label>
          ))}
        </div>
      </Field>

      <Field label="Search Stocks">
        <input
          placeholder="Symbol or industry..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="w-full bg-zinc-900 border border-zinc-700 rounded px-2 py-1 text-xs text-zinc-100 placeholder-zinc-600"
        />
      </Field>

      <Field label={`Available (${availableStocks.length})`}>
        <div className="max-h-40 overflow-y-auto space-y-0.5">
          {availableStocks.slice(0, 100).map((s) => (
            <button
              key={s.symbol}
              onClick={() => addStock(s.symbol)}
              disabled={selected.includes(s.symbol)}
              className={cn(
                "w-full text-left px-2 py-0.5 rounded text-xs truncate transition-colors",
                selected.includes(s.symbol)
                  ? "text-zinc-600 cursor-default"
                  : "text-zinc-300 hover:bg-zinc-800 cursor-pointer",
              )}
            >
              <span className="font-mono">{s.symbol}</span>{" "}
              <span className="text-zinc-600">{s.industry}</span>
            </button>
          ))}
          {availableStocks.length > 100 && (
            <div className="text-[10px] text-zinc-600 px-2 pt-1">
              +{availableStocks.length - 100} more (narrow search)
            </div>
          )}
        </div>
      </Field>

      <Field label={`Selected (${selected.length})`}>
        <div className="max-h-32 overflow-y-auto space-y-0.5">
          {selected.length === 0 && (
            <div className="text-[10px] text-zinc-600">No stocks selected</div>
          )}
          {selected.map((sym) => (
            <div key={sym} className="flex items-center justify-between text-xs">
              <span className="font-mono text-emerald-400">{sym}</span>
              {weightScheme === "manual" && (
                <input
                  type="number"
                  min={0}
                  step={0.1}
                  value={manualWeights[sym] ?? 1}
                  onChange={(e) =>
                    setManualWeights((prev) => ({
                      ...prev,
                      [sym]: Math.max(0, parseFloat(e.target.value) || 0),
                    }))
                  }
                  className="w-14 bg-zinc-900 border border-zinc-700 rounded px-1 py-0.5 text-[10px] font-mono text-zinc-300 text-right"
                />
              )}
              <button
                onClick={() => removeStock(sym)}
                className="text-zinc-600 hover:text-red-400 text-xs ml-1"
              >
                ×
              </button>
            </div>
          ))}
        </div>
      </Field>

      <Field label="Weighting">
        <select
          value={weightScheme}
          onChange={(e) => setWeightScheme(e.target.value as WeightScheme)}
          className="w-full bg-zinc-900 border border-zinc-700 rounded px-2 py-1 text-xs text-zinc-100"
        >
          <option value="equal">Equal Weight</option>
          <option value="manual">Manual</option>
        </select>
      </Field>

      <Field label="Rebalancing">
        <select
          value={rebalFreq}
          onChange={(e) => setRebalFreq(e.target.value)}
          className="w-full bg-zinc-900 border border-zinc-700 rounded px-2 py-1 text-xs text-zinc-100"
        >
          {REBALANCE_OPTIONS.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
      </Field>

      <Field label="Benchmark">
        <select
          value={benchmarkType}
          onChange={(e) => setBenchmarkType(e.target.value)}
          className="w-full bg-zinc-900 border border-zinc-700 rounded px-2 py-1 text-xs text-zinc-100"
        >
          {BENCHMARK_TYPES.map((b) => (
            <option key={b} value={b}>
              {b}
            </option>
          ))}
        </select>
      </Field>

      <RunButton
        onClick={handleRun}
        loading={loading}
        label="Run Simulation"
        disabled={selected.length < 1}
      />
      {(simMut.error || benchMut.error) && (
        <div className="text-red-400 text-[10px] mt-1">
          {(simMut.error as Error)?.message ?? (benchMut.error as Error)?.message}
        </div>
      )}
    </LeftSidebar>
  );

  /* ── Right sidebar: KPIs ───────────────────────────────────── */
  const right = (
    <RightSidebar>
      {sim && (
        <>
          <div className="text-[10px] uppercase tracking-wider text-zinc-500 font-semibold">
            Portfolio
          </div>
          <KPICard
            label="Total Return"
            value={fmtPct(sim.metrics.total_return)}
            accent={sim.metrics.total_return >= 0 ? "positive" : "negative"}
          />
          <KPICard
            label="Sharpe"
            value={fmtRatio(sim.metrics.sharpe_ratio)}
            accent={sim.metrics.sharpe_ratio >= 0 ? "positive" : "negative"}
          />
          <KPICard
            label="Sortino"
            value={fmtRatio(sim.metrics.sortino_ratio)}
            accent={sim.metrics.sortino_ratio >= 0 ? "positive" : "negative"}
          />
          <KPICard label="Max DD" value={fmtPct(sim.metrics.max_drawdown)} accent="negative" />
          <KPICard label="Volatility" value={fmtPct(sim.metrics.annualized_volatility)} />
          <KPICard
            label="Calmar"
            value={fmtRatio(sim.metrics.calmar_ratio)}
            accent={sim.metrics.calmar_ratio >= 0 ? "positive" : "negative"}
          />
        </>
      )}
      {bench && (
        <>
          <div className="mt-3 text-[10px] uppercase tracking-wider text-zinc-500 font-semibold">
            {bench.benchmark_name}
          </div>
          <KPICard
            label="Total Return"
            value={fmtPct(bench.total_return)}
            accent={bench.total_return >= 0 ? "positive" : "negative"}
          />
          <KPICard
            label="Sharpe"
            value={fmtRatio(bench.sharpe_ratio)}
            accent={bench.sharpe_ratio >= 0 ? "positive" : "negative"}
          />
          <KPICard label="Max DD" value={fmtPct(bench.max_drawdown)} accent="negative" />
          <KPICard label="Volatility" value={fmtPct(bench.volatility)} />
        </>
      )}
      {sim && bench && (
        <>
          <div className="mt-3 text-[10px] uppercase tracking-wider text-zinc-500 font-semibold">
            Relative
          </div>
          <KPICard
            label="Excess Return"
            value={fmtPct(sim.metrics.total_return - bench.total_return)}
            accent={sim.metrics.total_return - bench.total_return >= 0 ? "positive" : "negative"}
          />
          {sim.metrics.alpha !== undefined && (
            <KPICard
              label="Alpha"
              value={fmtPct(sim.metrics.alpha)}
              accent={sim.metrics.alpha >= 0 ? "positive" : "negative"}
            />
          )}
          {sim.metrics.beta !== undefined && (
            <KPICard label="Beta" value={fmtRatio(sim.metrics.beta)} accent="neutral" />
          )}
          {sim.metrics.information_ratio !== undefined && (
            <KPICard
              label="Info Ratio"
              value={fmtRatio(sim.metrics.information_ratio)}
              accent={sim.metrics.information_ratio >= 0 ? "positive" : "negative"}
            />
          )}
        </>
      )}
    </RightSidebar>
  );

  /* ── Bottom panel: portfolio table ─────────────────────────── */
  const bottom = selectedStockInfo.length > 0 && (
    <BottomPanel>
      <h3 className="text-[10px] uppercase tracking-wider text-zinc-500 mb-2 font-semibold">
        Portfolio Composition
      </h3>
      <table className="w-full text-xs text-left">
        <thead>
          <tr className="text-zinc-500 border-b border-zinc-800">
            <th className="pb-1 font-semibold">Symbol</th>
            <th className="pb-1 font-semibold">Sector</th>
            <th className="pb-1 font-semibold">Industry</th>
            <th className="pb-1 font-semibold text-right">Weight</th>
          </tr>
        </thead>
        <tbody>
          {selectedStockInfo.map((row) => (
            <tr key={row.symbol} className="border-b border-zinc-800/50 text-zinc-300">
              <td className="py-1 font-mono text-emerald-400">{row.symbol}</td>
              <td className="py-1">{row.sector}</td>
              <td className="py-1">{row.industry}</td>
              <td className="py-1 text-right font-mono">{(row.weight * 100).toFixed(1)}%</td>
            </tr>
          ))}
        </tbody>
      </table>
    </BottomPanel>
  );

  /* ── Center: charts ────────────────────────────────────────── */
  return (
    <AppLayout left={left} right={right} bottom={bottom}>
      <div className="flex gap-2 mb-3">
        {(["nav", "drawdown"] as const).map((t) => (
          <button
            key={t}
            onClick={() => setActiveTab(t)}
            className={cn(
              "px-3 py-1 rounded text-xs font-medium transition-colors",
              activeTab === t
                ? "bg-zinc-800 text-zinc-100"
                : "text-zinc-500 hover:text-zinc-300",
            )}
          >
            {t === "nav" ? "Cumulative NAV" : "Drawdown"}
          </button>
        ))}
      </div>

      {!sim && !loading && (
        <div className="flex items-center justify-center h-64 text-zinc-600 text-sm">
          Select stocks, configure weights, then click Run Simulation.
        </div>
      )}

      {activeTab === "nav" && sim && (
        <Plot
          data={[
            {
              x: sim.nav.map((p) => p.date),
              y: sim.nav.map((p) => p.value),
              type: "scatter",
              mode: "lines",
              name: "Portfolio",
              line: { color: "#3b82f6", width: 1.5 },
            },
            ...(bench
              ? [
                  {
                    x: bench.dates,
                    y: bench.cumulative_returns,
                    type: "scatter" as const,
                    mode: "lines" as const,
                    name: bench.benchmark_name,
                    line: { color: "#a1a1aa", width: 1, dash: "dot" as const },
                  },
                ]
              : []),
          ]}
          layout={{
            ...PLOTLY_LAYOUT,
            yaxis: {
              ...PLOTLY_LAYOUT.yaxis,
              title: { text: "NAV", font: { size: 10, color: "#71717a" } } as Partial<Plotly.LayoutAxis["title"]>,
            },
            showlegend: true,
          }}
          config={{ displayModeBar: false, responsive: true }}
          className="w-full h-[420px]"
        />
      )}

      {activeTab === "drawdown" && sim && (
        <Plot
          data={[
            {
              x: portfDD.map((p) => p.date),
              y: portfDD.map((p) => p.dd),
              type: "scatter",
              mode: "lines",
              fill: "tozeroy",
              name: "Portfolio DD",
              line: { color: "#ef4444", width: 1 },
              fillcolor: "rgba(239,68,68,0.15)",
            },
            ...(bench
              ? [
                  {
                    x: benchDD.map((p) => p.date),
                    y: benchDD.map((p) => p.dd),
                    type: "scatter" as const,
                    mode: "lines" as const,
                    name: `${bench.benchmark_name} DD`,
                    line: { color: "#a1a1aa", width: 1, dash: "dot" as const },
                  },
                ]
              : []),
          ]}
          layout={{
            ...PLOTLY_LAYOUT,
            yaxis: {
              ...PLOTLY_LAYOUT.yaxis,
              title: { text: "Drawdown", font: { size: 10, color: "#71717a" } } as Partial<Plotly.LayoutAxis["title"]>,
              tickformat: ".0%",
            },
            showlegend: true,
          }}
          config={{ displayModeBar: false, responsive: true }}
          className="w-full h-[420px]"
        />
      )}
    </AppLayout>
  );
}

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
