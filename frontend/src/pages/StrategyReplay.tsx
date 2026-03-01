import { useMutation, useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import Plot from "react-plotly.js";

import KPICard from "@/components/cards/KPICard.tsx";
import SignalBadge from "@/components/cards/SignalBadge.tsx";
import ReplayControls from "@/components/controls/ReplayControls.tsx";
import RunButton from "@/components/controls/RunButton.tsx";
import AppLayout from "@/components/layout/AppLayout.tsx";
import BottomPanel from "@/components/layout/BottomPanel.tsx";
import LeftSidebar from "@/components/layout/LeftSidebar.tsx";
import RightSidebar from "@/components/layout/RightSidebar.tsx";
import { api } from "@/lib/api.ts";
import type { ReplayFrame } from "@/lib/types.ts";
import { fmtPct, fmtRatio } from "@/lib/utils.ts";
import { useReplayStore } from "@/stores/replayStore.ts";

const PLOTLY_LAYOUT: Partial<Plotly.Layout> = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "transparent",
  font: { color: "#a1a1aa", family: "JetBrains Mono, monospace", size: 11 },
  margin: { t: 28, r: 16, b: 36, l: 52 },
  xaxis: { gridcolor: "#27272a", zerolinecolor: "#3f3f46" },
  yaxis: { gridcolor: "#27272a", zerolinecolor: "#3f3f46" },
  legend: { x: 0.02, y: 0.98, bgcolor: "transparent", font: { size: 10 } },
};

export default function StrategyReplay() {
  const [factor, setFactor] = useState("");
  const [rebalFreq, setRebalFreq] = useState("ME");
  const [tcost, setTcost] = useState(10);
  const [topPct, setTopPct] = useState(20);

  const { frameIndex, totalFrames, setTotalFrames, setFrame, pause } = useReplayStore();

  const factorsQuery = useQuery({
    queryKey: ["factors"],
    queryFn: api.listFactors,
  });

  const factors = factorsQuery.data?.factors ?? [];
  const selectedFactor = factor || factors[0] || "";

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

  useEffect(() => {
    return () => {
      pause();
      setTotalFrames(0);
    };
  }, [pause, setTotalFrames]);

  const visibleFrames = useMemo(() => frames.slice(0, frameIndex + 1), [frames, frameIndex]);

  const handleRun = () => {
    if (!selectedFactor) return;
    replayMut.mutate();
  };

  return (
    <AppLayout
      left={
        <LeftSidebar>
          <Field label="Factor">
            <select
              value={selectedFactor}
              onChange={(e) => setFactor(e.target.value)}
              disabled={replayMut.isPending}
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
              disabled={replayMut.isPending}
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
              disabled={replayMut.isPending}
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
              disabled={replayMut.isPending}
              className="w-full accent-blue-500"
            />
          </Field>

          <RunButton onClick={handleRun} loading={replayMut.isPending} label="Load Replay" />

          {replayMut.isError && (
            <div className="text-[10px] text-red-400 bg-red-950/50 border border-red-900 rounded px-2 py-1">
              {(replayMut.error as Error).message}
            </div>
          )}

          {totalFrames > 0 && (
            <div className="mt-2">
              <div className="text-[10px] uppercase tracking-wider text-zinc-500 mb-1">Timeline</div>
              <ReplayControls />
            </div>
          )}
        </LeftSidebar>
      }
      right={
        <RightSidebar>
          {currentFrame ? (
            <FrameKPIs frame={currentFrame} totalFrames={totalFrames} frameIndex={frameIndex} />
          ) : (
            <div className="text-xs text-zinc-600">Load replay to see live KPIs</div>
          )}
        </RightSidebar>
      }
      bottom={
        frames.length > 0 ? (
          <BottomPanel>
            <FrameTable frames={frames} currentIndex={frameIndex} />
          </BottomPanel>
        ) : undefined
      }
    >
      {frames.length > 0 ? (
        <div className="flex flex-col gap-4 h-full">
          <PnLChart frames={visibleFrames} allFrames={frames} currentIndex={frameIndex} />
          <DrawdownChart frames={visibleFrames} />
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

function EmptyState() {
  return (
    <div className="flex items-center justify-center h-full">
      <div className="text-center">
        <div className="text-zinc-600 text-sm">Select a factor and load the replay</div>
        <div className="text-zinc-700 text-xs mt-1">
          The strategy will play through time with live KPI updates
        </div>
      </div>
    </div>
  );
}

/* ── Charts ────────────────────────────────────────────────── */

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

/* ── Right-rail KPIs ───────────────────────────────────────── */

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

/* ── Bottom panel: frame table ─────────────────────────────── */

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
