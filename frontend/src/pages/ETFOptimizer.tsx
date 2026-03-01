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
import type { OptimizeResponse, SimulateResponse } from "@/lib/types.ts";
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

const DEFAULT_ETFS = [
  "VOO", "SOXX", "ITA", "DTEC", "IXJ", "IYK",
  "AIRR", "UGA", "MLPX", "GREK", "ARGT", "GLD",
];

type Tab = "frontier" | "nav";

export default function ETFOptimizer() {
  const [selected, setSelected] = useState<string[]>([]);
  const [rfRate, setRfRate] = useState(8.15);
  const [borrowRate, setBorrowRate] = useState(11.15);
  const [rebalFreq, setRebalFreq] = useState("Annual");
  const [activeTab, setActiveTab] = useState<Tab>("frontier");
  const [filter, setFilter] = useState("");

  const assetsQuery = useQuery({ queryKey: ["assets"], queryFn: api.listAssets });
  const assets = assetsQuery.data?.assets ?? [];
  const etfAssets = assets.filter(
    (a) => a.type === "ETF" || DEFAULT_ETFS.includes(a.symbol),
  );
  const filtered = filter
    ? etfAssets.filter((a) => a.symbol.includes(filter.toUpperCase()))
    : etfAssets;

  const optimizeMut = useMutation({
    mutationFn: () =>
      api.optimizePortfolio({
        symbols: selected,
        risk_free_rate: rfRate / 100,
        borrowing_rate: borrowRate / 100,
      }),
  });

  const simMut = useMutation({
    mutationFn: () => {
      const w = optimizeMut.data?.tangency.weights ?? {};
      return api.simulatePortfolio({
        symbols: selected,
        weights: w,
        freq: rebalFreq,
      });
    },
  });

  const loading = optimizeMut.isPending || simMut.isPending;
  const opt: OptimizeResponse | undefined = optimizeMut.data;
  const sim: SimulateResponse | undefined = simMut.data;

  const handleRun = () => {
    if (selected.length < 2) return;
    optimizeMut.mutate(undefined, {
      onSuccess: () => simMut.mutate(),
    });
  };

  const toggleSymbol = (sym: string) => {
    setSelected((prev) =>
      prev.includes(sym) ? prev.filter((s) => s !== sym) : [...prev, sym],
    );
  };

  const fetchCetes = () => {
    api.getCetes28().then((d) => setRfRate(d.rate)).catch(() => {});
  };

  return (
    <AppLayout
      left={
        <LeftSidebar>
          <Field label="ETF Selection">
            <input
              type="text"
              placeholder="Filter..."
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5 text-xs text-zinc-200 font-mono mb-1"
            />
            <div className="max-h-40 overflow-y-auto border border-zinc-800 rounded">
              {filtered.map((a) => (
                <label
                  key={a.symbol}
                  className="flex items-center gap-1.5 px-2 py-0.5 text-[11px] font-mono text-zinc-300 hover:bg-zinc-800/50 cursor-pointer"
                >
                  <input
                    type="checkbox"
                    checked={selected.includes(a.symbol)}
                    onChange={() => toggleSymbol(a.symbol)}
                    className="accent-blue-500"
                  />
                  {a.symbol}
                </label>
              ))}
            </div>
            <div className="text-[10px] text-zinc-600 mt-1">
              {selected.length} selected
            </div>
          </Field>

          <Field label="Risk-Free Rate (%)">
            <div className="flex gap-1">
              <input
                type="number"
                step={0.1}
                value={rfRate}
                onChange={(e) => setRfRate(Number(e.target.value))}
                className="flex-1 bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5 text-xs text-zinc-200 font-mono tabular-nums"
              />
              <button
                onClick={fetchCetes}
                className="px-2 py-1.5 text-[10px] font-semibold bg-zinc-800 hover:bg-zinc-700 text-zinc-300 rounded cursor-pointer"
              >
                CETES
              </button>
            </div>
          </Field>

          <Field label="Borrowing Rate (%)">
            <input
              type="number"
              step={0.1}
              value={borrowRate}
              onChange={(e) => setBorrowRate(Number(e.target.value))}
              className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5 text-xs text-zinc-200 font-mono tabular-nums"
            />
          </Field>

          <Field label="Rebalance Frequency">
            <select
              value={rebalFreq}
              onChange={(e) => setRebalFreq(e.target.value)}
              className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5 text-xs text-zinc-200 font-mono"
            >
              <option value="Annual">Annual</option>
              <option value="Quarterly">Quarterly</option>
              <option value="Monthly">Monthly</option>
            </select>
          </Field>

          <RunButton
            onClick={handleRun}
            loading={loading}
            disabled={selected.length < 2}
            label="Optimize"
          />

          {optimizeMut.isError && (
            <div className="text-[10px] text-red-400 bg-red-950/50 border border-red-900 rounded px-2 py-1">
              {(optimizeMut.error as Error).message}
            </div>
          )}
        </LeftSidebar>
      }
      right={
        <RightSidebar>
          {opt ? <TangencyKPIs opt={opt} rfRate={rfRate} /> : (
            <div className="text-xs text-zinc-600">Select ETFs and optimize</div>
          )}
        </RightSidebar>
      }
      bottom={opt ? <WeightsTable opt={opt} sim={sim} /> : undefined}
    >
      {opt ? (
        <div className="flex flex-col h-full">
          <TabBar active={activeTab} onChange={setActiveTab} />
          <div className="flex-1 min-h-0">
            {activeTab === "frontier" && <FrontierChart opt={opt} />}
            {activeTab === "nav" && sim && <NAVChart sim={sim} />}
            {activeTab === "nav" && !sim && (
              <div className="flex items-center justify-center h-full text-zinc-600 text-xs">
                Running simulation...
              </div>
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
    { id: "frontier", label: "Efficient Frontier" },
    { id: "nav", label: "Portfolio NAV" },
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
        <div className="text-zinc-600 text-sm">Select at least 2 ETFs and click Optimize</div>
        <div className="text-zinc-700 text-xs mt-1">
          Efficient frontier and tangency portfolio will appear here
        </div>
      </div>
    </div>
  );
}

/* ── Charts ────────────────────────────────────────────────── */

function FrontierChart({ opt }: { opt: OptimizeResponse }) {
  const frontier = opt.frontier;
  const cal = opt.cal;
  const t = opt.tangency;
  const mv = opt.min_vol;
  const ind = opt.individual;

  return (
    <Plot
      data={[
        {
          type: "scatter",
          x: frontier.map((p) => p.volatility * 100),
          y: frontier.map((p) => p.ret * 100),
          mode: "lines",
          name: "Efficient Frontier",
          line: { color: "#3b82f6", width: 2.5 },
        },
        {
          type: "scatter",
          x: cal.map((p) => p.volatility * 100),
          y: cal.map((p) => p.ret * 100),
          mode: "lines",
          name: "CAL",
          line: { color: "#22c55e", width: 1.5, dash: "dash" },
        },
        {
          type: "scatter",
          x: [t.volatility * 100],
          y: [t.ret * 100],
          mode: "markers",
          name: "Tangency",
          marker: { size: 12, color: "#ef4444", symbol: "star" },
        },
        {
          type: "scatter",
          x: [mv.volatility * 100],
          y: [mv.ret * 100],
          mode: "markers",
          name: "Min Variance",
          marker: { size: 10, color: "#f59e0b", symbol: "diamond" },
        },
        {
          type: "scatter",
          x: ind.map((a) => a.volatility * 100),
          y: ind.map((a) => a.ret * 100),
          mode: "markers+text",
          name: "Assets",
          marker: { size: 7, color: "#a1a1aa" },
          text: ind.map((a) => a.symbol),
          textposition: "top center",
          textfont: { size: 9 },
        },
      ]}
      layout={{
        ...PLOTLY_LAYOUT,
        height: 440,
        xaxis: {
          ...PLOTLY_LAYOUT.xaxis,
          title: { text: "Volatility (%)" } as Partial<Plotly.LayoutAxis["title"]>,
        },
        yaxis: {
          ...PLOTLY_LAYOUT.yaxis,
          title: { text: "Expected Return (%)" } as Partial<Plotly.LayoutAxis["title"]>,
        },
        showlegend: true,
      }}
      config={{ displayModeBar: false, responsive: true }}
      className="w-full"
    />
  );
}

function NAVChart({ sim }: { sim: SimulateResponse }) {
  return (
    <Plot
      data={[
        {
          type: "scatter",
          x: sim.nav.map((p) => p.date),
          y: sim.nav.map((p) => p.value),
          mode: "lines",
          line: { color: "#3b82f6", width: 1.5 },
          fill: "tozeroy",
          fillcolor: "rgba(59,130,246,0.05)",
          name: "NAV",
        },
      ]}
      layout={{
        ...PLOTLY_LAYOUT,
        height: 440,
        yaxis: {
          ...PLOTLY_LAYOUT.yaxis,
          title: { text: "Portfolio Value" } as Partial<Plotly.LayoutAxis["title"]>,
        },
      }}
      config={{ displayModeBar: false, responsive: true }}
      className="w-full"
    />
  );
}

/* ── Right-rail KPIs ───────────────────────────────────────── */

function TangencyKPIs({ opt, rfRate }: { opt: OptimizeResponse; rfRate: number }) {
  const t = opt.tangency;
  const topWeights = Object.entries(t.weights)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 5);

  return (
    <div className="flex flex-col gap-2">
      <div className="text-[10px] uppercase tracking-wider text-zinc-500">
        Tangency Portfolio
      </div>
      <KPICard label="Return" value={fmtPct(t.ret)} accent="positive" />
      <KPICard label="Volatility" value={fmtPct(t.volatility)} />
      <KPICard label="Sharpe" value={fmtRatio(t.sharpe)} accent="positive" />
      <KPICard label="Risk-Free" value={`${rfRate.toFixed(2)}%`} />

      <div className="text-[10px] uppercase tracking-wider text-zinc-500 mt-3">
        Top Weights
      </div>
      {topWeights.map(([sym, w]) => (
        <div key={sym} className="flex justify-between text-[11px] font-mono">
          <span className="text-zinc-400">{sym}</span>
          <span className="text-zinc-200 tabular-nums">{(w * 100).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  );
}

/* ── Bottom panel: weights + metrics ───────────────────────── */

function WeightsTable({ opt, sim }: { opt: OptimizeResponse; sim?: SimulateResponse }) {
  const weights = Object.entries(opt.tangency.weights).sort(([, a], [, b]) => b - a);

  return (
    <BottomPanel>
      <div className="flex gap-8">
        <div className="flex-1">
          <div className="text-[10px] uppercase tracking-wider text-zinc-500 mb-2">
            Tangency Weights
          </div>
          <table className="w-full text-[11px] font-mono">
            <thead>
              <tr className="text-zinc-500 border-b border-zinc-800">
                <th className="text-left px-2 py-1">Asset</th>
                <th className="text-right px-2 py-1">Weight</th>
              </tr>
            </thead>
            <tbody>
              {weights.map(([sym, w]) => (
                <tr key={sym} className="border-b border-zinc-900">
                  <td className="px-2 py-1 text-zinc-300">{sym}</td>
                  <td className="px-2 py-1 text-right tabular-nums text-zinc-200">
                    {(w * 100).toFixed(2)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {sim && (
          <div className="flex-1">
            <div className="text-[10px] uppercase tracking-wider text-zinc-500 mb-2">
              Simulation Metrics
            </div>
            <table className="w-full text-[11px] font-mono">
              <thead>
                <tr className="text-zinc-500 border-b border-zinc-800">
                  <th className="text-left px-2 py-1">Metric</th>
                  <th className="text-right px-2 py-1">Value</th>
                </tr>
              </thead>
              <tbody>
                {[
                  ["Total Return", fmtPct(sim.metrics.total_return)],
                  ["Ann. Return", fmtPct(sim.metrics.annualized_return)],
                  ["Volatility", fmtPct(sim.metrics.annualized_volatility)],
                  ["Sharpe", fmtRatio(sim.metrics.sharpe_ratio)],
                  ["Max DD", fmtPct(sim.metrics.max_drawdown)],
                ].map(([label, value]) => (
                  <tr key={label} className="border-b border-zinc-900">
                    <td className="px-2 py-1 text-zinc-400">{label}</td>
                    <td className="px-2 py-1 text-right tabular-nums text-zinc-200">
                      {value}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </BottomPanel>
  );
}
