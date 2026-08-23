import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import Plot from "react-plotly.js";

import KPICard from "@/components/cards/KPICard.tsx";
import AppLayout from "@/components/layout/AppLayout.tsx";
import LeftSidebar from "@/components/layout/LeftSidebar.tsx";
import RightSidebar from "@/components/layout/RightSidebar.tsx";
import { api } from "@/lib/api.ts";
import type { Cid1StatSummary } from "@/lib/types.ts";
import { cn, fmtPct, fmtRatio } from "@/lib/utils.ts";

const dark = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "transparent",
  font: { color: "#a1a1aa", size: 11, family: "JetBrains Mono, monospace" },
  xaxis: { gridcolor: "#27272a", linecolor: "#27272a", zeroline: false },
  yaxis: { gridcolor: "#27272a", linecolor: "#27272a", zeroline: true, zerolinecolor: "#3f3f46" },
  margin: { l: 56, r: 16, t: 32, b: 32 },
  showlegend: false,
};

function fmtStat(s: Cid1StatSummary | undefined): string {
  if (!s || s.mean === null || s.tstat === null) return "—";
  return `${s.mean >= 0 ? "+" : ""}${s.mean.toFixed(3)} (t ${s.tstat.toFixed(2)})`;
}

export default function IndexTop500() {
  const perf = useQuery({
    queryKey: ["top500-performance"],
    queryFn: () => api.getTop500Performance(),
    staleTime: Infinity,
  });
  const study = useQuery({
    queryKey: ["top500-cid1-study"],
    queryFn: () => api.getTop500Cid1Study(),
    staleTime: Infinity,
  });

  const rebalances = perf.data?.rebalances ?? [];
  const [holdingsDate, setHoldingsDate] = useState<string | null>(null);
  const selectedDate = holdingsDate ?? rebalances[rebalances.length - 1]?.date ?? null;

  const holdings = useQuery({
    queryKey: ["top500-holdings", selectedDate],
    queryFn: () => api.getTop500Holdings(selectedDate as string),
    enabled: selectedDate !== null,
    staleTime: Infinity,
  });

  const m = perf.data?.metrics;
  const s = study.data;
  const fmbCid1 = s?.fama_macbeth_multivariate?.["cid1"];

  return (
    <AppLayout
      left={
        <LeftSidebar>
          <h2 className="text-xs font-semibold text-zinc-300 mb-2">Simple Top-500 Index</h2>
          <p className="text-[11px] text-zinc-500 leading-relaxed mb-3">
            Same theory as the S&amp;P 500 — hold the largest 500 companies, cap-weighted — with
            the committee removed: membership is recomputed mechanically at each quarter&apos;s
            last trading day, executed at the next close. Between rebalances share counts are
            fixed, so weights drift with prices.
          </p>
          <div className="text-[11px] text-zinc-500 leading-relaxed mb-3 border border-zinc-800 rounded px-2 py-1.5">
            <div className="text-[10px] uppercase tracking-wider text-zinc-500 mb-1">
              Deliberate simplifications
            </div>
            <ul className="list-disc list-inside space-y-1">
              <li>Full-cap weights, not float-adjusted</li>
              <li>Universe = tickers with cap history here (collected from current + former
                S&amp;P members), so this is closer to a reconstruction than an independent index</li>
              <li>Mid-quarter delistings frozen at last close until next rebalance</li>
              <li>Gross returns; ^GSPC benchmark is a price index (no dividends) while our
                prices are dividend-adjusted — expect a structural excess</li>
            </ul>
          </div>
          <h2 className="text-xs font-semibold text-zinc-300 mb-2">Cid-1 relevance study</h2>
          <p className="text-[11px] text-zinc-500 leading-relaxed mb-3">
            At each rebalance, every constituent gets a trailing 252-day Cid-1 (total return ÷
            cost-basis pain; ADR-0009 boundary convention). Relevance is tested against the{" "}
            <span className="text-zinc-300">forward</span> quarter: Spearman IC, quintile
            spread, and Fama-MacBeth with momentum / vol / size controls — all with
            Newey-West t-stats. Persistence (rank autocorrelation between consecutive
            rebalances) is the prerequisite check.
          </p>
          {s && (
            <div className="text-[11px] text-zinc-500 leading-relaxed border border-zinc-800 rounded px-2 py-1.5">
              <div className="text-[10px] uppercase tracking-wider text-zinc-500 mb-1">Config</div>
              <div className="font-mono text-zinc-400">
                {String(s.config["start"])} → {String(s.config["end"])}
                <br />
                {String(s.config["n_rebalances"])} rebalances · window{" "}
                {String(s.config["window_days"])}d · lag {String(s.config["execution_lag_days"])}d
              </div>
            </div>
          )}
        </LeftSidebar>
      }
      right={
        <RightSidebar>
          <div className="flex items-center justify-between mb-2">
            <h2 className="text-xs font-semibold text-zinc-300">Constituents</h2>
            <select
              value={selectedDate ?? ""}
              onChange={(e) => setHoldingsDate(e.target.value)}
              className="bg-zinc-900 border border-zinc-800 rounded px-1.5 py-1 text-[11px] font-mono text-zinc-300"
            >
              {[...rebalances].reverse().map((r) => (
                <option key={r.date} value={r.date}>
                  {r.date}
                </option>
              ))}
            </select>
          </div>
          {holdings.data && (
            <>
              <p className="text-[10px] text-zinc-500 mb-2">
                {holdings.data.holdings.length} names · Cid-1 over trailing{" "}
                {holdings.data.window_days}d · ∞ = never below window start
              </p>
              <div className="overflow-y-auto max-h-[calc(100vh-160px)] border border-zinc-800 rounded">
                <table className="w-full text-[10px] font-mono tabular-nums">
                  <thead className="sticky top-0 bg-zinc-900 text-zinc-500">
                    <tr>
                      <th className="text-left px-1.5 py-1">Sym</th>
                      <th className="text-right px-1.5 py-1">Wgt</th>
                      <th className="text-right px-1.5 py-1">252d Ret</th>
                      <th className="text-right px-1.5 py-1">Cid-1</th>
                    </tr>
                  </thead>
                  <tbody>
                    {holdings.data.holdings.map((h) => (
                      <tr key={h.symbol} className="border-t border-zinc-800/50">
                        <td className="px-1.5 py-0.5 text-zinc-300">{h.symbol}</td>
                        <td className="px-1.5 py-0.5 text-right text-zinc-400">
                          {fmtPct(h.weight, 2)}
                        </td>
                        <td
                          className={cn(
                            "px-1.5 py-0.5 text-right",
                            (h.total_return ?? 0) >= 0 ? "text-emerald-400" : "text-red-400",
                          )}
                        >
                          {h.total_return === null ? "—" : fmtPct(h.total_return, 1)}
                        </td>
                        <td className="px-1.5 py-0.5 text-right text-zinc-400">
                          {h.cid1_ratio !== null
                            ? fmtRatio(h.cid1_ratio, 2)
                            : (h.cid1_angle ?? 0) > 1.5
                              ? "∞"
                              : "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </RightSidebar>
      }
    >
      {(perf.isLoading || study.isLoading) && (
        <p className="text-xs text-zinc-500">Building index and study (first load computes 21 years)…</p>
      )}
      {(perf.error || study.error) && (
        <p className="text-xs text-red-400">
          {String((perf.error ?? study.error) as Error)}
        </p>
      )}

      {perf.data && m && (
        <>
          <div className="grid grid-cols-6 gap-2 mb-4">
            <KPICard
              label="Ann Return"
              value={fmtPct(m["annualized_return"] ?? 0, 1)}
              accent={(m["annualized_return"] ?? 0) >= 0 ? "positive" : "negative"}
            />
            <KPICard label="Sharpe" value={fmtRatio(m["sharpe_ratio"] ?? 0)} />
            <KPICard
              label="Max Drawdown"
              value={fmtPct(m["max_drawdown"] ?? 0, 1)}
              accent="negative"
            />
            <KPICard label="Index Cid-1" value={fmtRatio(m["cid1_ratio"] ?? 0, 3)} />
            <KPICard
              label="Corr vs ^GSPC"
              value={fmtRatio(perf.data.correlation_vs_benchmark ?? 0, 3)}
              accent="neutral"
            />
            <KPICard
              label="Tracking Error"
              value={fmtPct(perf.data.tracking_error_ann ?? 0, 1)}
              accent="neutral"
            />
          </div>

          <div className="bg-zinc-900 border border-zinc-800 rounded p-2 mb-4">
            <Plot
              data={[
                {
                  x: perf.data.dates,
                  y: perf.data.index_cumulative,
                  type: "scatter",
                  mode: "lines",
                  name: "Top-500 (ours)",
                  line: { color: "#3b82f6", width: 1.5 },
                  hovertemplate: "%{x}<br>%{y:.2f}x<extra>Top-500</extra>",
                },
                ...(perf.data.benchmark_cumulative
                  ? [
                      {
                        x: perf.data.dates,
                        y: perf.data.benchmark_cumulative,
                        type: "scatter" as const,
                        mode: "lines" as const,
                        name: "^GSPC",
                        line: { color: "#71717a", width: 1 },
                        hovertemplate: "%{x}<br>%{y:.2f}x<extra>^GSPC</extra>",
                      },
                    ]
                  : []),
              ]}
              layout={{
                ...dark,
                title: {
                  text: "Growth of $1 — quarterly top-500 cap-weighted vs ^GSPC (price index)",
                  font: { size: 12, color: "#71717a" },
                  x: 0,
                } as Partial<Plotly.Layout["title"]>,
                height: 340,
                showlegend: true,
                legend: { orientation: "h", y: 1.12, font: { size: 10 } },
                yaxis: { ...dark.yaxis, type: "log", tickformat: ".1f" },
                autosize: true,
              }}
              config={{ displayModeBar: false, responsive: true }}
              className="w-full"
              useResizeHandler
              style={{ width: "100%" }}
            />
          </div>
        </>
      )}

      {s && (
        <>
          <h2 className="text-xs font-semibold text-zinc-300 mb-2">
            Is Cid-1 relevant? Evidence at each rebalance ({s.ic_summary.n_obs} quarters)
          </h2>
          <div className="grid grid-cols-4 gap-2 mb-4">
            <KPICard label="Persistence (rank AC)" value={fmtStat(s.persistence_summary)} accent="neutral" />
            <KPICard
              label="Mean IC (fwd qtr)"
              value={fmtStat(s.ic_summary)}
              accent={Math.abs(s.ic_summary.tstat ?? 0) >= 2 ? "positive" : undefined}
            />
            <KPICard label="Q5−Q1 spread / qtr" value={fmtStat(s.quantile_spread_summary)} />
            <KPICard
              label="FMB γ(Cid-1) w/ controls"
              value={fmtStat(fmbCid1)}
              accent={(fmbCid1?.tstat ?? 0) <= -2 ? "negative" : undefined}
            />
          </div>

          <div className="grid grid-cols-2 gap-4 mb-4">
            <div className="bg-zinc-900 border border-zinc-800 rounded p-2">
              <Plot
                data={[
                  {
                    x: s.per_date.map((r) => r.date),
                    y: s.per_date.map((r) => r.ic),
                    type: "bar",
                    marker: {
                      color: s.per_date.map((r) => ((r.ic ?? 0) >= 0 ? "#34d399" : "#f87171")),
                    },
                    hovertemplate: "%{x}<br>IC %{y:.3f}<extra></extra>",
                  },
                ]}
                layout={{
                  ...dark,
                  title: {
                    text: "Spearman IC: Cid-1 at rebalance vs forward-quarter return",
                    font: { size: 11, color: "#71717a" },
                    x: 0,
                  } as Partial<Plotly.Layout["title"]>,
                  height: 240,
                  autosize: true,
                }}
                config={{ displayModeBar: false, responsive: true }}
                className="w-full"
                useResizeHandler
                style={{ width: "100%" }}
              />
            </div>
            <div className="bg-zinc-900 border border-zinc-800 rounded p-2">
              <Plot
                data={[
                  {
                    x: ["Q1 (low)", "Q2", "Q3", "Q4", "Q5 (high)"],
                    y: s.quantile_avg_forward_returns,
                    type: "bar",
                    marker: { color: "#3b82f6" },
                    hovertemplate: "%{x}<br>%{y:.2%}<extra></extra>",
                  },
                ]}
                layout={{
                  ...dark,
                  title: {
                    text: "Avg forward-quarter return by Cid-1 quintile",
                    font: { size: 11, color: "#71717a" },
                    x: 0,
                  } as Partial<Plotly.Layout["title"]>,
                  height: 240,
                  yaxis: { ...dark.yaxis, tickformat: ".1%" },
                  autosize: true,
                }}
                config={{ displayModeBar: false, responsive: true }}
                className="w-full"
                useResizeHandler
                style={{ width: "100%" }}
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4 mb-4">
            <div className="bg-zinc-900 border border-zinc-800 rounded p-2">
              <Plot
                data={[
                  {
                    x: s.per_date.map((r) => r.date),
                    y: s.per_date.map((r) => r.persistence_vs_prev),
                    type: "scatter",
                    mode: "lines",
                    line: { color: "#a78bfa", width: 1.5 },
                    hovertemplate: "%{x}<br>rank AC %{y:.2f}<extra></extra>",
                  },
                ]}
                layout={{
                  ...dark,
                  title: {
                    text: "Persistence: constituent Cid-1 rank autocorrelation vs prior rebalance",
                    font: { size: 11, color: "#71717a" },
                    x: 0,
                  } as Partial<Plotly.Layout["title"]>,
                  height: 240,
                  yaxis: { ...dark.yaxis, range: [-0.2, 1] },
                  autosize: true,
                }}
                config={{ displayModeBar: false, responsive: true }}
                className="w-full"
                useResizeHandler
                style={{ width: "100%" }}
              />
            </div>
            <div className="bg-zinc-900 border border-zinc-800 rounded p-2 overflow-x-auto">
              <div className="text-[11px] text-zinc-500 mb-1">
                Start-year sensitivity (walk-forward re-evaluation, annual cadence)
              </div>
              <table className="w-full text-[10px] font-mono tabular-nums">
                <thead className="text-zinc-500">
                  <tr>
                    <th className="text-left px-1.5 py-1">From</th>
                    <th className="text-right px-1.5 py-1">Qtrs</th>
                    <th className="text-right px-1.5 py-1">Mean IC</th>
                    <th className="text-right px-1.5 py-1">t</th>
                    <th className="text-right px-1.5 py-1">Spread</th>
                    <th className="text-right px-1.5 py-1">t</th>
                  </tr>
                </thead>
                <tbody>
                  {s.start_year_sensitivity.map((r) => (
                    <tr key={r.start_year} className="border-t border-zinc-800/50 text-zinc-400">
                      <td className="px-1.5 py-0.5">{r.start_year}</td>
                      <td className="px-1.5 py-0.5 text-right">{r.n_quarters}</td>
                      <td className="px-1.5 py-0.5 text-right">{r.mean_ic?.toFixed(3) ?? "—"}</td>
                      <td
                        className={cn(
                          "px-1.5 py-0.5 text-right",
                          Math.abs(r.ic_tstat ?? 0) >= 2 ? "text-amber-400" : "",
                        )}
                      >
                        {r.ic_tstat?.toFixed(2) ?? "—"}
                      </td>
                      <td className="px-1.5 py-0.5 text-right">
                        {r.mean_spread !== null ? fmtPct(r.mean_spread, 2) : "—"}
                      </td>
                      <td
                        className={cn(
                          "px-1.5 py-0.5 text-right",
                          Math.abs(r.spread_tstat ?? 0) >= 2 ? "text-amber-400" : "",
                        )}
                      >
                        {r.spread_tstat?.toFixed(2) ?? "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="text-[11px] text-amber-400/80 leading-relaxed mb-2 border border-amber-900/40 bg-amber-950/20 rounded px-3 py-2">
            <span className="font-semibold">Verdict:</span> Cid-1 is a highly persistent
            characteristic (rank autocorrelation ≈ {fmtRatio(s.persistence_summary.mean ?? 0, 2)},
            t ≈ {fmtRatio(s.persistence_summary.tstat ?? 0, 1)}) but shows no univariate
            predictive power for the next quarter (mean IC {fmtRatio(s.ic_summary.mean ?? 0, 3)},
            t {fmtRatio(s.ic_summary.tstat ?? 0, 2)}; flat quintiles). The Fama-MacBeth
            coefficient turns significantly <span className="font-semibold">negative</span> once
            momentum / vol / size controls enter — but Cid-1&apos;s numerator is trailing
            return itself, so that partial effect is heavily collinear with momentum and
            flips sign across start years. Not usable as a selection criterion.
          </div>
          <div className="grid grid-cols-4 gap-2 mb-4">
            {Object.entries(s.fama_macbeth_multivariate).map(([name, stat]) => (
              <div key={name} className="bg-zinc-900 border border-zinc-800 rounded px-3 py-2">
                <div className="text-[10px] uppercase tracking-wider text-zinc-500 mb-1">
                  FMB γ · {name}
                </div>
                <div
                  className={cn(
                    "text-sm font-mono tabular-nums",
                    Math.abs(stat.tstat ?? 0) >= 2 ? "text-zinc-100" : "text-zinc-500",
                  )}
                >
                  {fmtStat(stat)}
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </AppLayout>
  );
}
