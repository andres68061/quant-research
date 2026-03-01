import { useMutation } from "@tanstack/react-query";
import { useState } from "react";
import Plot from "react-plotly.js";

import KPICard from "@/components/cards/KPICard.tsx";
import TeX from "@/components/TeX.tsx";
import RunButton from "@/components/controls/RunButton.tsx";
import AppLayout from "@/components/layout/AppLayout.tsx";
import BottomPanel from "@/components/layout/BottomPanel.tsx";
import LeftSidebar from "@/components/layout/LeftSidebar.tsx";
import RightSidebar from "@/components/layout/RightSidebar.tsx";
import { api } from "@/lib/api.ts";
import type { SharpeComparisonResponse } from "@/lib/types.ts";
import { cn } from "@/lib/utils.ts";

type Tab = "cumulative" | "drawdown" | "distributions";

const PL: Partial<Plotly.Layout> = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "transparent",
  font: { color: "#a1a1aa", family: "JetBrains Mono, monospace", size: 11 },
  margin: { t: 36, r: 16, b: 44, l: 52 },
};

export default function SharpeRatioLimitations() {
  const [targetSharpe, setTargetSharpe] = useState(1.5);
  const [nDays, setNDays] = useState(1260);
  const [seed, setSeed] = useState(42);
  const [activeTab, setActiveTab] = useState<Tab>("cumulative");

  const mut = useMutation({
    mutationFn: () => api.sharpeComparison(targetSharpe, nDays, seed),
  });

  const data = mut.data as SharpeComparisonResponse | undefined;
  const invs = data?.investments ?? [];

  return (
    <AppLayout
      left={
        <LeftSidebar>
          <Field label="Target Sharpe">
            <input
              type="range"
              min={0.5}
              max={3.0}
              step={0.1}
              value={targetSharpe}
              onChange={(e) => setTargetSharpe(+e.target.value)}
              className="w-full accent-blue-500"
            />
            <span className="text-xs font-mono text-zinc-400">{targetSharpe.toFixed(1)}</span>
          </Field>

          <Field label="Trading Days">
            <select
              value={nDays}
              onChange={(e) => setNDays(+e.target.value)}
              className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5 text-xs text-zinc-200 font-mono"
            >
              {[252, 504, 756, 1008, 1260, 1512, 2520].map((d) => (
                <option key={d} value={d}>
                  {d} ({(d / 252).toFixed(0)}y)
                </option>
              ))}
            </select>
          </Field>

          <Field label="Seed">
            <input
              type="number"
              min={1}
              max={99999}
              value={seed}
              onChange={(e) => setSeed(+e.target.value)}
              className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5 text-xs text-zinc-200 font-mono"
            />
          </Field>

          <RunButton onClick={() => mut.mutate()} loading={mut.isPending} label="Simulate" />

          <div className="border-t border-zinc-800 pt-3 mt-1">
            <div className="text-[10px] uppercase tracking-wider text-zinc-500 mb-2">Key Formulas</div>
            <div className="space-y-2 text-[11px] text-zinc-400">
              <TeX tex="S = \frac{R_p - R_f}{\sigma_p}\cdot\sqrt{252}" />
              <TeX tex="\text{Sortino} = \frac{R_p - R_f}{\sigma_d}\cdot\sqrt{252}" />
            </div>
          </div>
        </LeftSidebar>
      }
      right={
        <RightSidebar>
          {invs.map((inv) => (
            <KPICard
              key={inv.name}
              label={inv.name}
              value={inv.metrics.sharpe.toFixed(2)}
              accent="neutral"
            />
          ))}
          {invs.length === 0 && (
            <p className="text-xs text-zinc-600">Run simulation to see metrics</p>
          )}
        </RightSidebar>
      }
      bottom={
        invs.length > 0 ? (
          <BottomPanel>
            <MetricsTable investments={invs} />
          </BottomPanel>
        ) : undefined
      }
    >
      {/* Tabs */}
      <div className="flex gap-2 mb-3">
        {(["cumulative", "drawdown", "distributions"] as Tab[]).map((t) => (
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

      {invs.length === 0 ? (
        <EmptyState />
      ) : activeTab === "cumulative" ? (
        <CumulativeChart investments={invs} />
      ) : activeTab === "drawdown" ? (
        <DrawdownChart investments={invs} />
      ) : (
        <DistributionsChart investments={invs} />
      )}
    </AppLayout>
  );
}

/* ── Charts ─────────────────────────────────────────────────── */

type Inv = SharpeComparisonResponse["investments"][number];

function CumulativeChart({ investments }: { investments: Inv[] }) {
  return (
    <Plot
      data={investments.map((inv) => ({
        y: inv.prices,
        type: "scatter" as const,
        mode: "lines" as const,
        name: inv.name,
        line: { color: inv.color, width: 2 },
      }))}
      layout={{
        ...PL,
        title: { text: "Growth of $100", font: { size: 13, color: "#d4d4d8" } } as Partial<Plotly.Layout["title"]>,
        yaxis: { title: { text: "Value ($)" } as Partial<Plotly.LayoutAxis["title"]>, gridcolor: "#27272a" },
        xaxis: { title: { text: "Trading Day" } as Partial<Plotly.LayoutAxis["title"]>, gridcolor: "#27272a" },
        showlegend: true,
        legend: { font: { size: 10 } },
        shapes: [
          { type: "line", y0: 100, y1: 100, x0: 0, x1: investments[0].prices.length - 1, line: { dash: "dash", color: "#52525b", width: 1 } } as Partial<Plotly.Shape>,
        ],
      }}
      useResizeHandler
      className="w-full h-full min-h-[400px]"
      config={{ displayModeBar: false }}
    />
  );
}

function DrawdownChart({ investments }: { investments: Inv[] }) {
  const traces = investments.map((inv) => {
    const prices = inv.prices;
    const dd: number[] = [];
    let peak = prices[0];
    for (const p of prices) {
      peak = Math.max(peak, p);
      dd.push(((p - peak) / peak) * 100);
    }
    return {
      y: dd,
      type: "scatter" as const,
      mode: "lines" as const,
      fill: "tozeroy" as const,
      name: inv.name,
      line: { color: inv.color, width: 1.5 },
    };
  });

  return (
    <Plot
      data={traces}
      layout={{
        ...PL,
        title: { text: "Drawdown", font: { size: 13, color: "#d4d4d8" } } as Partial<Plotly.Layout["title"]>,
        yaxis: { title: { text: "Drawdown (%)" } as Partial<Plotly.LayoutAxis["title"]>, gridcolor: "#27272a" },
        xaxis: { title: { text: "Trading Day" } as Partial<Plotly.LayoutAxis["title"]>, gridcolor: "#27272a" },
        showlegend: true,
        legend: { font: { size: 10 } },
      }}
      useResizeHandler
      className="w-full h-full min-h-[400px]"
      config={{ displayModeBar: false }}
    />
  );
}

function DistributionsChart({ investments }: { investments: Inv[] }) {
  return (
    <Plot
      data={investments.map((inv) => ({
        x: inv.daily_returns,
        type: "histogram" as const,
        name: inv.name,
        marker: { color: inv.color },
        opacity: 0.6,
        nbinsx: 60,
      }))}
      layout={{
        ...PL,
        title: { text: "Daily Return Distributions", font: { size: 13, color: "#d4d4d8" } } as Partial<Plotly.Layout["title"]>,
        barmode: "overlay",
        xaxis: { title: { text: "Daily Return (%)" } as Partial<Plotly.LayoutAxis["title"]>, gridcolor: "#27272a" },
        yaxis: { title: { text: "Frequency" } as Partial<Plotly.LayoutAxis["title"]>, gridcolor: "#27272a" },
        showlegend: true,
        legend: { font: { size: 10 } },
      }}
      useResizeHandler
      className="w-full h-full min-h-[400px]"
      config={{ displayModeBar: false }}
    />
  );
}

/* ── Metrics Table ──────────────────────────────────────────── */

function MetricsTable({ investments }: { investments: Inv[] }) {
  const cols = [
    "Sharpe",
    "Sortino",
    "Max DD",
    "Calmar",
    "Total Ret",
    "Vol (Ann)",
    "Skew",
    "Kurt",
    "Win %",
    "Best Day",
    "Worst Day",
  ];

  const rows = investments.map((inv) => {
    const m = inv.metrics;
    return [
      inv.name,
      m.sharpe.toFixed(2),
      m.sortino.toFixed(2),
      `${m.max_drawdown.toFixed(1)}%`,
      m.calmar.toFixed(2),
      `${m.total_return.toFixed(1)}%`,
      `${m.annualized_vol.toFixed(1)}%`,
      m.skewness.toFixed(2),
      m.kurtosis.toFixed(2),
      `${m.win_rate.toFixed(1)}%`,
      `${m.best_day.toFixed(2)}%`,
      `${m.worst_day.toFixed(2)}%`,
    ];
  });

  return (
    <table className="text-xs font-mono w-full">
      <thead>
        <tr className="text-zinc-500 text-left">
          <th className="pr-3 pb-1">Name</th>
          {cols.map((c) => (
            <th key={c} className="pr-3 pb-1">
              {c}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((r, i) => (
          <tr key={i} className="text-zinc-300 border-t border-zinc-800/50">
            <td className="pr-3 py-1 font-semibold" style={{ color: investments[i].color }}>
              {r[0]}
            </td>
            {r.slice(1).map((v, j) => (
              <td key={j} className="pr-3 py-1">
                {v}
              </td>
            ))}
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
      <div className="text-lg font-semibold">Sharpe Ratio Limitations</div>
      <p className="text-xs max-w-md text-center leading-relaxed">
        Generate five investments with <strong>identical Sharpe ratios</strong> but
        very different risk profiles. See why the Sharpe ratio alone can be misleading.
      </p>
      <p className="text-xs text-zinc-700">Configure parameters on the left and click Simulate.</p>
    </div>
  );
}
