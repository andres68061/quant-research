import { useMutation, useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import Plot from "react-plotly.js";

import KPICard from "@/components/cards/KPICard.tsx";
import KPIPanel from "@/components/cards/KPIPanel.tsx";
import SignalBadge from "@/components/cards/SignalBadge.tsx";
import EquityCurve from "@/components/charts/EquityCurve.tsx";
import ReplayControls from "@/components/controls/ReplayControls.tsx";
import RunButton from "@/components/controls/RunButton.tsx";
import AppLayout from "@/components/layout/AppLayout.tsx";
import BottomPanel from "@/components/layout/BottomPanel.tsx";
import LeftSidebar from "@/components/layout/LeftSidebar.tsx";
import RightSidebar from "@/components/layout/RightSidebar.tsx";
import { api } from "@/lib/api.ts";
import type { AllVarResult, BacktestResponse, ReplayFrame } from "@/lib/types.ts";
import { cn, fmtInt, fmtPct, fmtRatio } from "@/lib/utils.ts";
import { useReplayStore } from "@/stores/replayStore.ts";

type ViewMode = "summary" | "replay";

const PLOTLY_LAYOUT: Partial<Plotly.Layout> = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "transparent",
  font: { color: "#a1a1aa", family: "JetBrains Mono, monospace", size: 11 },
  margin: { t: 28, r: 16, b: 36, l: 52 },
  xaxis: { gridcolor: "#27272a", zerolinecolor: "#3f3f46" },
  yaxis: { gridcolor: "#27272a", zerolinecolor: "#3f3f46" },
  legend: { x: 0.02, y: 0.98, bgcolor: "transparent", font: { size: 10 } },
};

export default function PortfolioSimulator() {
  const [factor, setFactor] = useState("");
  const [rebalFreq, setRebalFreq] = useState("ME");
  const [tcost, setTcost] = useState(10);
  const [topPct, setTopPct] = useState(20);
  const [longOnly, setLongOnly] = useState(false);

  const fiveYearsAgo = new Date(Date.now() - 5 * 365.25 * 86_400_000).toISOString().slice(0, 10);
  const todayStr = new Date().toISOString().slice(0, 10);
  const [startDate, setStartDate] = useState(fiveYearsAgo);
  const [endDate, setEndDate] = useState(todayStr);

  const [viewMode, setViewMode] = useState<ViewMode>("summary");

  const factorsQuery = useQuery({
    queryKey: ["factors"],
    queryFn: api.listFactors,
  });

  const backtest = useMutation({
    mutationFn: api.runBacktest,
  });

  const assetsQuery = useQuery({ queryKey: ["assets"], queryFn: api.listAssets });
  const assets = assetsQuery.data?.assets ?? [];
  const [varSymbol, setVarSymbol] = useState("");
  const [varData, setVarData] = useState<AllVarResult | null>(null);

  const factors = factorsQuery.data?.factors ?? [];
  const selectedFactor = factor || factors[0] || "";
  const result: BacktestResponse | undefined = backtest.data;

  /* ── Replay state ───────────────────────────────────────── */
  const { frameIndex, totalFrames, setTotalFrames, setFrame, pause } = useReplayStore();

  const replayMut = useMutation({
    mutationFn: () =>
      api.getReplayFrames({
        factor: selectedFactor,
        rebalance_freq: rebalFreq,
        tail: 1200,
      }),
    onSuccess: (data) => {
      setTotalFrames(data.frames.length);
      setFrame(0);
      pause();
    },
  });

  const frames: ReplayFrame[] = replayMut.data?.frames ?? [];
  const currentFrame = frames[frameIndex] ?? null;
  const visibleFrames = useMemo(() => frames.slice(0, frameIndex + 1), [frames, frameIndex]);

  useEffect(() => {
    return () => {
      pause();
      setTotalFrames(0);
    };
  }, [pause, setTotalFrames]);

  /* ── Handlers ───────────────────────────────────────────── */
  const handleRun = () => {
    if (!selectedFactor) return;
    backtest.mutate({
      factor_col: selectedFactor,
      rebalance_freq: rebalFreq,
      transaction_cost_bps: tcost,
      top_pct: topPct / 100,
      bottom_pct: topPct / 100,
      long_only: longOnly,
      start_date: startDate,
      end_date: endDate,
    });
  };

  const handleLoadReplay = () => {
    if (!selectedFactor) return;
    replayMut.mutate();
  };

  const handleFetchVar = () => {
    if (!varSymbol) return;
    api.getVar(varSymbol).then(setVarData).catch(() => setVarData(null));
  };

  const loading = backtest.isPending || replayMut.isPending;
  const hasResult = !!result;
  const hasReplay = frames.length > 0;

  return (
    <AppLayout
      left={
        <LeftSidebar>
          <Field label="Factor">
            <select
              value={selectedFactor}
              onChange={(e) => setFactor(e.target.value)}
              disabled={loading}
              className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5 text-xs text-zinc-200 font-mono disabled:opacity-50"
            >
              {factors.map((f) => (
                <option key={f} value={f}>{f}</option>
              ))}
            </select>
          </Field>

          <Field label="Rebalance Frequency">
            <select
              value={rebalFreq}
              onChange={(e) => setRebalFreq(e.target.value)}
              disabled={loading}
              className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5 text-xs text-zinc-200 font-mono disabled:opacity-50"
            >
              <option value="ME">Monthly</option>
              <option value="QE">Quarterly</option>
            </select>
          </Field>

          <Field label="Transaction Cost (bps)">
            <input
              type="number"
              value={tcost}
              onChange={(e) => setTcost(Number(e.target.value))}
              min={0}
              step={1}
              disabled={loading}
              className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5 text-xs text-zinc-200 font-mono tabular-nums disabled:opacity-50"
            />
          </Field>

          <Field label={`Top/Bottom ${topPct}%`}>
            <input
              type="range"
              min={5}
              max={50}
              value={topPct}
              onChange={(e) => setTopPct(Number(e.target.value))}
              disabled={loading}
              className="w-full accent-blue-500"
            />
          </Field>

          <label className="flex items-center gap-2 text-xs text-zinc-400 cursor-pointer">
            <input
              type="checkbox"
              checked={longOnly}
              onChange={(e) => setLongOnly(e.target.checked)}
              className="accent-blue-500"
            />
            Long only
          </label>

          <Field label="Start Date">
            <input
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              disabled={loading}
              className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5 text-xs text-zinc-200 font-mono disabled:opacity-50"
            />
          </Field>

          <Field label="End Date">
            <input
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              disabled={loading}
              className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5 text-xs text-zinc-200 font-mono disabled:opacity-50"
            />
          </Field>

          <div className="flex gap-2">
            <RunButton onClick={handleRun} loading={backtest.isPending} label="Run Backtest" />
            <RunButton onClick={handleLoadReplay} loading={replayMut.isPending} label="Replay" />
          </div>

          {(backtest.isError || replayMut.isError) && (
            <div className="text-[10px] text-red-400 bg-red-950/50 border border-red-900 rounded px-2 py-1">
              {((backtest.error ?? replayMut.error) as Error).message}
            </div>
          )}

          {/* View mode toggle (visible once any data is loaded) */}
          {(hasResult || hasReplay) && (
            <div className="flex rounded border border-zinc-800 overflow-hidden mt-1">
              <button
                onClick={() => setViewMode("summary")}
                className={cn(
                  "flex-1 text-[10px] py-1.5 font-semibold uppercase tracking-wider transition-colors cursor-pointer",
                  viewMode === "summary"
                    ? "bg-zinc-800 text-zinc-200"
                    : "bg-zinc-900 text-zinc-600 hover:text-zinc-400",
                )}
              >
                Summary
              </button>
              <button
                onClick={() => setViewMode("replay")}
                className={cn(
                  "flex-1 text-[10px] py-1.5 font-semibold uppercase tracking-wider transition-colors cursor-pointer",
                  viewMode === "replay"
                    ? "bg-zinc-800 text-zinc-200"
                    : "bg-zinc-900 text-zinc-600 hover:text-zinc-400",
                )}
              >
                Replay
              </button>
            </div>
          )}

          {viewMode === "replay" && totalFrames > 0 && (
            <div className="mt-1">
              <div className="text-[10px] uppercase tracking-wider text-zinc-500 mb-1">Timeline</div>
              <ReplayControls />
            </div>
          )}

          <div className="border-t border-zinc-800 pt-3 mt-2">
            <div className="text-[10px] uppercase tracking-wider text-zinc-500 mb-2">VaR Analysis</div>
            <Field label="Symbol">
              <select
                value={varSymbol}
                onChange={(e) => setVarSymbol(e.target.value)}
                className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5 text-xs text-zinc-200 font-mono"
              >
                <option value="">Select symbol</option>
                {assets.map((a) => (
                  <option key={a.symbol} value={a.symbol}>{a.symbol}</option>
                ))}
              </select>
            </Field>
            <button
              onClick={handleFetchVar}
              disabled={!varSymbol}
              className={cn(
                "w-full mt-1 py-1.5 rounded text-[10px] font-semibold uppercase tracking-wider transition-all",
                varSymbol
                  ? "bg-zinc-800 hover:bg-zinc-700 text-zinc-300 cursor-pointer"
                  : "bg-zinc-900 text-zinc-600 cursor-not-allowed",
              )}
            >
              Compute VaR
            </button>
          </div>
        </LeftSidebar>
      }
      right={
        viewMode === "replay" && currentFrame ? (
          <RightSidebar>
            <FrameKPIs frame={currentFrame} totalFrames={totalFrames} frameIndex={frameIndex} />
          </RightSidebar>
        ) : result ? (
          <RightSidebar>
            <KPIPanel metrics={result.metrics} />
            {result.metrics.information_ratio != null && (
              <div className="mt-2 flex flex-col gap-2">
                <div className="text-[10px] uppercase tracking-wider text-zinc-500">Benchmark</div>
                <KPICard label="Info Ratio" value={fmtRatio(result.metrics.information_ratio)} />
                {result.metrics.beta != null && (
                  <KPICard label="Beta" value={fmtRatio(result.metrics.beta)} />
                )}
                {result.metrics.alpha != null && (
                  <KPICard label="Alpha" value={fmtPct(result.metrics.alpha)} />
                )}
              </div>
            )}
            <div className="mt-2 text-[10px] text-zinc-600 font-mono">
              {fmtInt(result.total_days)} trading days
            </div>
          </RightSidebar>
        ) : (
          <RightSidebar>
            <div className="text-xs text-zinc-600">Run a backtest to see metrics</div>
          </RightSidebar>
        )
      }
      bottom={
        viewMode === "replay" && hasReplay ? (
          <BottomPanel>
            <FrameTable frames={frames} currentIndex={frameIndex} />
          </BottomPanel>
        ) : result ? (
          <BottomPanel>
            <MetricsTable metrics={result.metrics} varData={varData} />
          </BottomPanel>
        ) : undefined
      }
    >
      {viewMode === "replay" && hasReplay ? (
        <div className="flex flex-col gap-4 h-full">
          <PnLChart frames={visibleFrames} allFrames={frames} currentIndex={frameIndex} />
          <DrawdownChart frames={visibleFrames} />
        </div>
      ) : result ? (
        <EquityCurve data={result.equity_curve} title="Cumulative Returns" height={420} />
      ) : (
        <EmptyState />
      )}
    </AppLayout>
  );
}

/* ── Shared sub-components ────────────────────────────────── */

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

function EmptyState() {
  return (
    <div className="flex items-center justify-center h-full">
      <div className="text-center">
        <div className="text-zinc-600 text-sm">Select a factor and run a backtest</div>
        <div className="text-zinc-700 text-xs mt-1">
          Results will appear here — use Replay for frame-by-frame animation
        </div>
      </div>
    </div>
  );
}

/* ── Summary metrics table ────────────────────────────────── */

function MetricsTable({
  metrics,
  varData,
}: {
  metrics: BacktestResponse["metrics"];
  varData: AllVarResult | null;
}) {
  const rows: [string, string][] = [
    ["Total Return", fmtPct(metrics.total_return)],
    ["Annualized Return", fmtPct(metrics.annualized_return)],
    ["Annualized Volatility", fmtPct(metrics.annualized_volatility)],
    ["Sharpe Ratio", fmtRatio(metrics.sharpe_ratio)],
    ["Sortino Ratio", fmtRatio(metrics.sortino_ratio)],
    ["Max Drawdown", fmtPct(metrics.max_drawdown)],
    ["Calmar Ratio", fmtRatio(metrics.calmar_ratio)],
    ["Periods", fmtInt(metrics.n_periods)],
  ];

  if (metrics.information_ratio != null)
    rows.push(["Information Ratio", fmtRatio(metrics.information_ratio)]);
  if (metrics.beta != null) rows.push(["Beta", fmtRatio(metrics.beta)]);
  if (metrics.alpha != null) rows.push(["Alpha", fmtPct(metrics.alpha)]);

  return (
    <div className="flex gap-8">
      <table className="text-xs font-mono flex-1">
        <thead>
          <tr className="text-[10px] uppercase tracking-wider text-zinc-500 border-b border-zinc-800">
            <th className="text-left py-1.5 px-2">Metric</th>
            <th className="text-right py-1.5 px-2">Value</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([label, value]) => (
            <tr key={label} className="border-b border-zinc-900">
              <td className="py-1 px-2 text-zinc-400">{label}</td>
              <td className="py-1 px-2 text-right tabular-nums text-zinc-200">{value}</td>
            </tr>
          ))}
        </tbody>
      </table>

      {varData && (
        <table className="text-xs font-mono flex-1">
          <thead>
            <tr className="text-[10px] uppercase tracking-wider text-zinc-500 border-b border-zinc-800">
              <th className="text-left py-1.5 px-2">VaR ({varData.confidence}%)</th>
              <th className="text-right py-1.5 px-2">VaR</th>
              <th className="text-right py-1.5 px-2">CVaR</th>
            </tr>
          </thead>
          <tbody>
            {(["historical", "parametric", "monte_carlo"] as const).map((method) => (
              <tr key={method} className="border-b border-zinc-900">
                <td className="py-1 px-2 text-zinc-400 capitalize">
                  {method.replace("_", " ")}
                </td>
                <td className="py-1 px-2 text-right tabular-nums text-red-400">
                  {varData[method].var.toFixed(2)}%
                </td>
                <td className="py-1 px-2 text-right tabular-nums text-red-400">
                  {varData[method].cvar.toFixed(2)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

/* ── Replay charts ────────────────────────────────────────── */

function PnLChart({
  frames,
  allFrames,
  currentIndex,
}: {
  frames: ReplayFrame[];
  allFrames: ReplayFrame[];
  currentIndex: number;
}) {
  const dates = frames.map((f) => f.date);
  const cumPnl = frames.map((f) => f.cumulative_pnl);
  const currentDate = allFrames[currentIndex]?.date;

  return (
    <Plot
      data={[
        {
          type: "scatter",
          x: dates,
          y: cumPnl,
          mode: "lines",
          line: { color: "#3b82f6", width: 1.5 },
          name: "Cumulative PnL",
          fill: "tozeroy",
          fillcolor: "rgba(59,130,246,0.05)",
        },
      ]}
      layout={{
        ...PLOTLY_LAYOUT,
        height: 260,
        yaxis: {
          ...PLOTLY_LAYOUT.yaxis,
          title: { text: "Cum. PnL" } as Partial<Plotly.LayoutAxis["title"]>,
          tickformat: ".2%",
        },
        shapes: currentDate
          ? ([
              {
                type: "line",
                x0: currentDate,
                x1: currentDate,
                y0: 0,
                y1: 1,
                yref: "paper",
                line: { color: "#a1a1aa", width: 1, dash: "dot" },
              },
            ] as Partial<Plotly.Shape>[])
          : [],
      }}
      config={{ displayModeBar: false, responsive: true }}
      className="w-full"
    />
  );
}

function DrawdownChart({ frames }: { frames: ReplayFrame[] }) {
  return (
    <Plot
      data={[
        {
          type: "scatter",
          x: frames.map((f) => f.date),
          y: frames.map((f) => f.drawdown),
          mode: "lines",
          line: { color: "#ef4444", width: 1 },
          fill: "tozeroy",
          fillcolor: "rgba(239,68,68,0.08)",
          name: "Drawdown",
        },
      ]}
      layout={{
        ...PLOTLY_LAYOUT,
        height: 160,
        yaxis: {
          ...PLOTLY_LAYOUT.yaxis,
          title: { text: "Drawdown" } as Partial<Plotly.LayoutAxis["title"]>,
          tickformat: ".1%",
        },
      }}
      config={{ displayModeBar: false, responsive: true }}
      className="w-full"
    />
  );
}

/* ── Replay right-rail KPIs ───────────────────────────────── */

function FrameKPIs({
  frame,
  totalFrames,
  frameIndex,
}: {
  frame: ReplayFrame;
  totalFrames: number;
  frameIndex: number;
}) {
  return (
    <div className="flex flex-col gap-2">
      <div className="text-[10px] font-mono text-zinc-500 bg-zinc-900 border border-zinc-800 rounded px-2 py-1">
        {frame.date} | Frame {frameIndex + 1}/{totalFrames}
      </div>

      <div className="text-[10px] uppercase tracking-wider text-zinc-500">Position</div>
      <SignalBadge signal={frame.position} />

      <KPICard
        label="PnL Today"
        value={fmtPct(frame.pnl_today)}
        accent={frame.pnl_today >= 0 ? "positive" : "negative"}
      />
      <KPICard
        label="Cumulative PnL"
        value={fmtPct(frame.cumulative_pnl)}
        accent={frame.cumulative_pnl >= 0 ? "positive" : "negative"}
      />
      <KPICard
        label="Drawdown"
        value={fmtPct(frame.drawdown)}
        accent="negative"
      />
      <KPICard
        label="Rolling Sharpe"
        value={frame.rolling_sharpe != null ? fmtRatio(frame.rolling_sharpe) : "---"}
        accent={
          frame.rolling_sharpe != null
            ? frame.rolling_sharpe >= 0
              ? "positive"
              : "negative"
            : undefined
        }
      />
    </div>
  );
}

/* ── Replay bottom panel: frame table ─────────────────────── */

function FrameTable({ frames, currentIndex }: { frames: ReplayFrame[]; currentIndex: number }) {
  const start = Math.max(0, currentIndex - 5);
  const end = Math.min(frames.length, currentIndex + 6);
  const visible = frames.slice(start, end);

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-[11px] font-mono">
        <thead>
          <tr className="text-zinc-500 border-b border-zinc-800">
            {["#", "Date", "Position", "PnL Today", "Cum PnL", "Drawdown", "Sharpe"].map((h) => (
              <th key={h} className="text-left px-2 py-1 font-medium">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {visible.map((f, i) => {
            const globalIdx = start + i;
            const isCurrent = globalIdx === currentIndex;
            return (
              <tr
                key={f.date}
                className={
                  isCurrent
                    ? "bg-zinc-800/60 border-b border-zinc-700"
                    : "border-b border-zinc-900 hover:bg-zinc-900/50"
                }
              >
                <td className="px-2 py-1 tabular-nums text-zinc-600">{globalIdx + 1}</td>
                <td className="px-2 py-1">{f.date}</td>
                <td className="px-2 py-1">
                  <SignalBadge signal={f.position} />
                </td>
                <td className={`px-2 py-1 tabular-nums ${f.pnl_today >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                  {fmtPct(f.pnl_today)}
                </td>
                <td className={`px-2 py-1 tabular-nums ${f.cumulative_pnl >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                  {fmtPct(f.cumulative_pnl)}
                </td>
                <td className="px-2 py-1 tabular-nums text-red-400">{fmtPct(f.drawdown)}</td>
                <td className="px-2 py-1 tabular-nums">
                  {f.rolling_sharpe != null ? fmtRatio(f.rolling_sharpe) : "---"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
