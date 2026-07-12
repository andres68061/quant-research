import { useMutation } from "@tanstack/react-query";
import { useState } from "react";
import Plot from "react-plotly.js";

import KPICard from "@/components/cards/KPICard.tsx";
import TeX from "@/components/TeX.tsx";
import RunButton from "@/components/controls/RunButton.tsx";
import AppLayout from "@/components/layout/AppLayout.tsx";
import LeftSidebar from "@/components/layout/LeftSidebar.tsx";
import RightSidebar from "@/components/layout/RightSidebar.tsx";
import { api } from "@/lib/api.ts";
import type { MultiRatioComparisonResponse, SimulatedInvestment } from "@/lib/types.ts";
import { cn } from "@/lib/utils.ts";

/** One chart ≈ visible main area height so the first plot fills the screen; scroll for the rest. */
const CHART_WRAP_CLASS =
  "w-full shrink-0 h-[calc(100dvh-7rem)] min-h-[420px]";

const PL: Partial<Plotly.Layout> = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "transparent",
  font: { color: "#a1a1aa", family: "JetBrains Mono, monospace", size: 11 },
  margin: { t: 36, r: 16, b: 44, l: 52 },
};

export default function SharpeRatioLimitations() {
  const [targetRatio, setTargetRatio] = useState(1.5);
  const [nDays, setNDays] = useState(1260);
  const [seed, setSeed] = useState(42);

  const mut = useMutation({
    mutationFn: () => api.sharpeComparison(targetRatio, nDays, seed),
  });

  const data = mut.data as MultiRatioComparisonResponse | undefined;
  const sharpeInvs = data?.by_sharpe ?? [];

  return (
    <AppLayout
      left={
        <LeftSidebar>
          <Field label="Target ratio (all three blocks)">
            <input
              type="range"
              min={0.5}
              max={3.0}
              step={0.1}
              value={targetRatio}
              onChange={(e) => setTargetRatio(+e.target.value)}
              className="w-full accent-blue-500"
            />
            <span className="text-xs font-mono text-zinc-400">{targetRatio.toFixed(1)}</span>
          </Field>
          <p className="text-[10px] text-zinc-600 leading-snug -mt-1">
            One value calibrates each demo: identical Sharpe, identical Sortino, or identical Calmar—same
            synthetic paths, different daily means.
          </p>

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
            <div className="text-[10px] uppercase tracking-wider text-zinc-500 mb-3">Key Formulas</div>
            <div className="flex flex-col gap-5 text-[11px] text-zinc-400">
              <TeX display math="S = \frac{R_p - R_f}{\sigma_p}\cdot\sqrt{252}" />
              <TeX display math="\text{Sortino} = \frac{R_p - R_f}{\sigma_d}\cdot\sqrt{252}" />
              <TeX display math="\text{Calmar} = \frac{R_{ann}}{|\text{max drawdown}|}" />
            </div>
          </div>
        </LeftSidebar>
      }
      right={
        <RightSidebar>
          {sharpeInvs.length > 0 && (
            <p className="text-[10px] text-zinc-600 mb-2 leading-snug">
              Cards: Sharpe-block KPIs (each path&apos;s Sortino/Calmar differ). Scroll main panel for Sortino and Calmar
              demos.
            </p>
          )}
          {sharpeInvs.map((inv) => (
            <KPICard
              key={inv.name}
              label={inv.name}
              value={inv.metrics.sharpe.toFixed(2)}
              accent="neutral"
            />
          ))}
          {sharpeInvs.length === 0 && (
            <p className="text-xs text-zinc-600">Run simulation to see metrics</p>
          )}
        </RightSidebar>
      }
      bottom={undefined}
    >
      {!data ? (
        <EmptyState />
      ) : (
        <div className="space-y-16 pb-8">
          <RatioDemoSection
            title="Sharpe ratio"
            subtitle={`Each series mean-tuned so annualised Sharpe ≈ ${data.target.toFixed(1)} (same five risk shapes).`}
            highlightMetric="sharpe"
            investments={data.by_sharpe}
          />
          <RatioDemoSection
            title="Sortino ratio"
            subtitle={`Same paths as above; daily mean chosen so annualised Sortino ≈ ${data.target.toFixed(1)}.`}
            highlightMetric="sortino"
            investments={data.by_sortino}
          />
          <RatioDemoSection
            title="Calmar ratio"
            subtitle={`Same paths; daily mean chosen so Calmar (ann. return / |max DD|) ≈ ${data.target.toFixed(1)}.`}
            highlightMetric="calmar"
            investments={data.by_calmar}
          />
        </div>
      )}
    </AppLayout>
  );
}

/* ── Per-ratio section ──────────────────────────────────────── */

type HighlightMetric = "sharpe" | "sortino" | "calmar";

function RatioDemoSection({
  title,
  subtitle,
  highlightMetric,
  investments,
}: {
  title: string;
  subtitle: string;
  highlightMetric: HighlightMetric;
  investments: SimulatedInvestment[];
}) {
  return (
    <section className="border-t border-zinc-800 pt-6 first:border-t-0 first:pt-0">
      <h2 className="text-sm font-semibold text-zinc-200 mb-1">{title}</h2>
      <p className="text-[11px] text-zinc-500 mb-4 leading-relaxed max-w-3xl">{subtitle}</p>

      <div className="flex flex-col gap-6">
        <div className={CHART_WRAP_CLASS}>
          <CumulativeChart investments={investments} />
        </div>
        <div className={CHART_WRAP_CLASS}>
          <DrawdownChart investments={investments} />
        </div>
        <div className={CHART_WRAP_CLASS}>
          <DistributionsChart investments={investments} />
        </div>
      </div>

      <div className="mt-6">
        <MetricsTable investments={investments} highlightMetric={highlightMetric} />
      </div>
    </section>
  );
}

/* ── Charts ─────────────────────────────────────────────────── */

function CumulativeChart({ investments }: { investments: SimulatedInvestment[] }) {
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
        autosize: true,
        title: { text: "Growth of $100", font: { size: 13, color: "#d4d4d8" } } as Partial<
          Plotly.Layout["title"]
        >,
        yaxis: { title: { text: "Value ($)" } as Partial<Plotly.LayoutAxis["title"]>, gridcolor: "#27272a" },
        xaxis: { title: { text: "Trading Day" } as Partial<Plotly.LayoutAxis["title"]>, gridcolor: "#27272a" },
        showlegend: true,
        legend: { font: { size: 10 } },
        shapes: [
          {
            type: "line",
            y0: 100,
            y1: 100,
            x0: 0,
            x1: investments[0].prices.length - 1,
            line: { dash: "dash", color: "#52525b", width: 1 },
          } as Partial<Plotly.Shape>,
        ],
      }}
      useResizeHandler
      className="w-full h-full"
      style={{ width: "100%", height: "100%" }}
      config={{ displayModeBar: false }}
    />
  );
}

function DrawdownChart({ investments }: { investments: SimulatedInvestment[] }) {
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
        autosize: true,
        title: { text: "Drawdown", font: { size: 13, color: "#d4d4d8" } } as Partial<Plotly.Layout["title"]>,
        yaxis: { title: { text: "Drawdown (%)" } as Partial<Plotly.LayoutAxis["title"]>, gridcolor: "#27272a" },
        xaxis: { title: { text: "Trading Day" } as Partial<Plotly.LayoutAxis["title"]>, gridcolor: "#27272a" },
        showlegend: true,
        legend: { font: { size: 10 } },
      }}
      useResizeHandler
      className="w-full h-full"
      style={{ width: "100%", height: "100%" }}
      config={{ displayModeBar: false }}
    />
  );
}

function DistributionsChart({ investments }: { investments: SimulatedInvestment[] }) {
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
        autosize: true,
        title: { text: "Daily Return Distributions", font: { size: 13, color: "#d4d4d8" } } as Partial<
          Plotly.Layout["title"]
        >,
        barmode: "overlay",
        xaxis: { title: { text: "Daily Return (%)" } as Partial<Plotly.LayoutAxis["title"]>, gridcolor: "#27272a" },
        yaxis: { title: { text: "Frequency" } as Partial<Plotly.LayoutAxis["title"]>, gridcolor: "#27272a" },
        showlegend: true,
        legend: { font: { size: 10 } },
      }}
      useResizeHandler
      className="w-full h-full"
      style={{ width: "100%", height: "100%" }}
      config={{ displayModeBar: false }}
    />
  );
}

/* ── Metrics Table ──────────────────────────────────────────── */

function MetricsTable({
  investments,
  highlightMetric,
}: {
  investments: SimulatedInvestment[];
  highlightMetric: HighlightMetric;
}) {
  const cols = [
    { key: "sharpe" as const, label: "Sharpe" },
    { key: "sortino" as const, label: "Sortino" },
    { key: "max_drawdown" as const, label: "Max DD" },
    { key: "calmar" as const, label: "Calmar" },
    { key: "total_return" as const, label: "Total Ret" },
    { key: "annualized_vol" as const, label: "Vol (Ann)" },
    { key: "skewness" as const, label: "Skew" },
    { key: "kurtosis" as const, label: "Kurt" },
    { key: "win_rate" as const, label: "Win %" },
    { key: "best_day" as const, label: "Best Day" },
    { key: "worst_day" as const, label: "Worst Day" },
  ];

  const fmtCell = (key: (typeof cols)[number]["key"], m: SimulatedInvestment["metrics"]): string => {
    const v = m[key];
    if (key === "max_drawdown" || key === "total_return" || key === "annualized_vol" || key === "win_rate") {
      return `${Number(v).toFixed(1)}%`;
    }
    if (key === "best_day" || key === "worst_day") {
      return `${Number(v).toFixed(2)}%`;
    }
    return Number(v).toFixed(2);
  };

  return (
    <table className="text-xs font-mono w-full">
      <thead>
        <tr className="text-zinc-500 text-left">
          <th className="pr-3 pb-1">Name</th>
          {cols.map((c) => (
            <th
              key={c.key}
              className={cn(
                "pr-3 pb-1",
                c.key === highlightMetric && "text-blue-400",
              )}
            >
              {c.label}
              {c.key === highlightMetric ? " *" : ""}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {investments.map((inv) => (
          <tr key={inv.name} className="text-zinc-300 border-t border-zinc-800/50">
            <td className="pr-3 py-1 font-semibold" style={{ color: inv.color }}>
              {inv.name}
            </td>
            {cols.map((c) => (
              <td
                key={c.key}
                className={cn("pr-3 py-1", c.key === highlightMetric && "text-blue-400/90")}
              >
                {fmtCell(c.key, inv.metrics)}
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
      <div className="text-lg font-semibold">Ratio limitations</div>
      <p className="text-xs max-w-md text-center leading-relaxed">
        Generate five investments with the same <strong>Sharpe</strong>, <strong>Sortino</strong>, and{" "}
        <strong>Calmar</strong> (each in its own block) but very different risk profiles. Scroll down for full-height
        charts—growth, drawdown, then distributions—then the next ratio block.
      </p>
      <p className="text-xs text-zinc-700">Configure parameters on the left and click Simulate.</p>
    </div>
  );
}
