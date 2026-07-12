import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import Plot from "react-plotly.js";

import AppLayout from "@/components/layout/AppLayout.tsx";
import LeftSidebar from "@/components/layout/LeftSidebar.tsx";
import { api } from "@/lib/api.ts";
import { cn } from "@/lib/utils.ts";

const FACTOR_META: Record<string, { name: string; long: string; short: string; color: string }> = {
  mkt_rf: {
    name: "Market (Mkt−RF)",
    long: "the whole stock market, cap-weighted",
    short: "1-month T-bills (the risk-free rate)",
    color: "#60a5fa",
  },
  smb: {
    name: "Size (SMB)",
    long: "small-cap stocks",
    short: "big-cap stocks",
    color: "#34d399",
  },
  hml: {
    name: "Value (HML)",
    long: "high book-to-market (cheap) stocks",
    short: "low book-to-market (expensive) stocks",
    color: "#f87171",
  },
  rmw: {
    name: "Profitability (RMW)",
    long: "robust-profitability firms",
    short: "weak-profitability firms",
    color: "#fbbf24",
  },
  cma: {
    name: "Investment (CMA)",
    long: "conservative firms (low asset growth)",
    short: "aggressive firms (high asset growth)",
    color: "#a78bfa",
  },
};

const START_OPTIONS = [
  { label: "1963 (full)", value: "" },
  { label: "1990", value: "1990-01-01" },
  { label: "2000", value: "2000-01-01" },
  { label: "2010", value: "2010-01-01" },
];

const PLOT_LAYOUT: Partial<Plotly.Layout> = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "transparent",
  font: { color: "#a1a1aa", family: "JetBrains Mono, monospace", size: 11 },
  margin: { t: 24, r: 16, b: 44, l: 56 },
  xaxis: { gridcolor: "#27272a" },
  yaxis: { gridcolor: "#27272a", type: "log", title: { text: "growth of $1 (log)" } },
  legend: { orientation: "h", y: -0.15 },
};

export default function FamaFrench() {
  const [start, setStart] = useState("");
  const { data, isLoading, error } = useQuery({
    queryKey: ["ff5-series", start],
    queryFn: () => api.getFF5Series(start || undefined),
    staleTime: 10 * 60 * 1000,
  });

  return (
    <AppLayout
      left={
        <LeftSidebar>
          <div className="space-y-1">
            <div className="text-[10px] uppercase tracking-wider text-zinc-500 mb-2">
              Chart Start
            </div>
            {START_OPTIONS.map((opt) => (
              <button
                key={opt.value}
                onClick={() => setStart(opt.value)}
                className={cn(
                  "w-full text-left px-2.5 py-1.5 text-xs rounded transition-colors",
                  start === opt.value
                    ? "bg-zinc-800 text-zinc-100"
                    : "text-zinc-500 hover:text-zinc-300 hover:bg-zinc-900",
                )}
              >
                {opt.label}
              </button>
            ))}
          </div>
          <div className="mt-6 text-[11px] text-zinc-500 leading-relaxed space-y-2">
            <p>
              Source: Kenneth French data library (daily, 1963→). Published with a ~2-month lag —
              reference data, never a live trading signal.
            </p>
            <p>
              Stored at <span className="font-mono text-zinc-400">fama_french_5.parquet</span>,
              refreshed by the daily update.
            </p>
          </div>
        </LeftSidebar>
      }
    >
      <div className="p-4 space-y-4 overflow-y-auto h-full">
        <h1 className="text-sm font-semibold text-zinc-200">Fama-French 5 Factors</h1>

        <Section title="What they are">
          <p>
            Five long/short portfolios that explain most of the cross-section of stock returns
            (Fama &amp; French 1993, 2015). Each factor is the return of buying one group of
            stocks and shorting the opposite group — so a factor&apos;s return is the{" "}
            <em>premium</em> the market paid for that characteristic, not the return of any single
            stock. Your alpha only counts if it survives after regressing against these five.
          </p>
        </Section>

        <div className="grid grid-cols-1 gap-2">
          {Object.entries(FACTOR_META).map(([key, meta]) => (
            <div
              key={key}
              className="bg-zinc-900 border border-zinc-800 rounded px-3 py-2 flex items-baseline gap-3"
            >
              <span className="w-3 h-3 rounded-sm shrink-0 self-center" style={{ background: meta.color }} />
              <span className="font-mono text-xs text-zinc-200 w-44 shrink-0">{meta.name}</span>
              <span className="text-xs text-zinc-400">
                LONG <span className="text-emerald-400">{meta.long}</span> · SHORT{" "}
                <span className="text-red-400">{meta.short}</span>
              </span>
            </div>
          ))}
        </div>

        <Section title="How they're built: the 2×3 double sort">
          <p className="mb-3">
            Every June, all NYSE/AMEX/NASDAQ stocks are placed into a grid: 2 size buckets (split
            at the NYSE median market cap) × 3 characteristic buckets (30th / 70th percentiles).
            Portfolios are cap-weighted and held for a year. HML, for example, is the average
            return of the two Value cells minus the average of the two Growth cells — the size
            split ensures the factor isn&apos;t secretly a small-cap bet:
          </p>
          <SortGrid />
          <p className="mt-3">
            RMW does the same with operating profitability instead of book-to-market; CMA with
            asset growth (inverted: conservative minus aggressive); SMB averages the small-minus-big
            spread across all characteristic columns.
          </p>
        </Section>

        <Section title={`Cumulative growth of $1 ${data ? `(${data.first_date} → ${data.last_date})` : ""}`}>
          {isLoading && <p className="text-xs text-zinc-500">Loading factor history…</p>}
          {error && <p className="text-xs text-red-400">Failed: {(error as Error).message}</p>}
          {data && (
            <Plot
              data={Object.entries(data.growth).map(([key, values]) => ({
                x: data.dates,
                y: values,
                type: "scatter" as const,
                mode: "lines" as const,
                name: FACTOR_META[key]?.name ?? key,
                line: { color: FACTOR_META[key]?.color, width: 1.5 },
              }))}
              layout={PLOT_LAYOUT}
              config={{ displayModeBar: false }}
              style={{ width: "100%", height: 420 }}
              useResizeHandler
            />
          )}
        </Section>

        {data && (
          <Section title="Annualized statistics (daily returns, full selected window)">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-zinc-500 border-b border-zinc-800">
                  <th className="px-3 py-2 text-left font-medium">Factor</th>
                  <th className="px-3 py-2 text-right font-medium">Ann. Return</th>
                  <th className="px-3 py-2 text-right font-medium">Ann. Vol</th>
                  <th className="px-3 py-2 text-right font-medium">Sharpe</th>
                </tr>
              </thead>
              <tbody>
                {data.stats.map((s) => (
                  <tr key={s.factor} className="border-b border-zinc-800/50">
                    <td className="px-3 py-1.5 font-mono text-zinc-200">
                      {FACTOR_META[s.factor]?.name ?? s.factor}
                    </td>
                    <td
                      className={cn(
                        "px-3 py-1.5 text-right font-mono tabular-nums",
                        s.annualized_return >= 0 ? "text-emerald-400" : "text-red-400",
                      )}
                    >
                      {(s.annualized_return * 100).toFixed(2)}%
                    </td>
                    <td className="px-3 py-1.5 text-right font-mono tabular-nums text-zinc-300">
                      {(s.annualized_volatility * 100).toFixed(2)}%
                    </td>
                    <td className="px-3 py-1.5 text-right font-mono tabular-nums text-zinc-300">
                      {s.sharpe_ratio.toFixed(2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Section>
        )}

        <Section title="Why we care">
          <ul className="list-disc pl-5 space-y-1">
            <li>
              <span className="text-zinc-300">Benchmarking:</span> a strategy&apos;s returns are
              regressed on these factors — the intercept is true alpha, the loadings show what
              risk you&apos;re actually being paid for.
            </li>
            <li>
              <span className="text-zinc-300">Sanity checks:</span> if a &quot;new signal&quot;
              loads 0.9 on HML, it is value in disguise, not a discovery.
            </li>
            <li>
              <span className="text-zinc-300">Regime context:</span> factor premia move in long
              cycles (value&apos;s lost decade 2010–2020 is visible in the chart above).
            </li>
          </ul>
        </Section>
      </div>
    </AppLayout>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded p-4">
      <h2 className="text-xs font-semibold text-zinc-300 mb-2 uppercase tracking-wider">{title}</h2>
      <div className="text-xs text-zinc-400 leading-relaxed">{children}</div>
    </div>
  );
}

function SortGrid() {
  const cells = [
    { label: "Small Growth", tone: "short" },
    { label: "Small Neutral", tone: "none" },
    { label: "Small Value", tone: "long" },
    { label: "Big Growth", tone: "short" },
    { label: "Big Neutral", tone: "none" },
    { label: "Big Value", tone: "long" },
  ];
  return (
    <div>
      <div className="grid grid-cols-3 gap-1 max-w-md">
        {cells.map((c) => (
          <div
            key={c.label}
            className={cn(
              "px-2 py-3 text-center text-[11px] font-mono rounded border",
              c.tone === "long" && "border-emerald-800 bg-emerald-950/40 text-emerald-400",
              c.tone === "short" && "border-red-900 bg-red-950/30 text-red-400",
              c.tone === "none" && "border-zinc-800 bg-zinc-950 text-zinc-500",
            )}
          >
            {c.label}
          </div>
        ))}
      </div>
      <p className="mt-2 text-[11px] text-zinc-500">
        HML = avg(<span className="text-emerald-400">green</span>) − avg(
        <span className="text-red-400">red</span>): long both value cells, short both growth
        cells, size-neutral by construction.
      </p>
    </div>
  );
}
