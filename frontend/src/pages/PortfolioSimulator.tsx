import { useMutation, useQuery } from "@tanstack/react-query";
import { useState } from "react";

import KPICard from "@/components/cards/KPICard.tsx";
import KPIPanel from "@/components/cards/KPIPanel.tsx";
import EquityCurve from "@/components/charts/EquityCurve.tsx";
import RunButton from "@/components/controls/RunButton.tsx";
import AppLayout from "@/components/layout/AppLayout.tsx";
import BottomPanel from "@/components/layout/BottomPanel.tsx";
import LeftSidebar from "@/components/layout/LeftSidebar.tsx";
import RightSidebar from "@/components/layout/RightSidebar.tsx";
import { api } from "@/lib/api.ts";
import type { AllVarResult, BacktestResponse } from "@/lib/types.ts";
import { cn, fmtInt, fmtPct, fmtRatio } from "@/lib/utils.ts";

export default function PortfolioSimulator() {
  const [factor, setFactor] = useState("");
  const [rebalFreq, setRebalFreq] = useState("ME");
  const [tcost, setTcost] = useState(10);
  const [topPct, setTopPct] = useState(20);
  const [longOnly, setLongOnly] = useState(false);

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

  const handleRun = () => {
    if (!selectedFactor) return;
    backtest.mutate({
      factor_col: selectedFactor,
      rebalance_freq: rebalFreq,
      transaction_cost_bps: tcost,
      top_pct: topPct / 100,
      bottom_pct: topPct / 100,
      long_only: longOnly,
    });
  };

  const handleFetchVar = () => {
    if (!varSymbol) return;
    api.getVar(varSymbol).then(setVarData).catch(() => setVarData(null));
  };

  return (
    <AppLayout
      left={
        <LeftSidebar>
          <Field label="Factor">
            <select
              value={selectedFactor}
              onChange={(e) => setFactor(e.target.value)}
              className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5 text-xs text-zinc-200 font-mono"
            >
              {factors.map((f) => (
                <option key={f} value={f}>
                  {f}
                </option>
              ))}
            </select>
          </Field>

          <Field label="Rebalance Frequency">
            <select
              value={rebalFreq}
              onChange={(e) => setRebalFreq(e.target.value)}
              className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5 text-xs text-zinc-200 font-mono"
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
              className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5 text-xs text-zinc-200 font-mono tabular-nums"
            />
          </Field>

          <Field label={`Top/Bottom ${topPct}%`}>
            <input
              type="range"
              min={5}
              max={50}
              value={topPct}
              onChange={(e) => setTopPct(Number(e.target.value))}
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

          <RunButton onClick={handleRun} loading={backtest.isPending} label="Run Backtest" />

          {backtest.isError && (
            <div className="text-[10px] text-red-400 bg-red-950/50 border border-red-900 rounded px-2 py-1">
              {(backtest.error as Error).message}
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
        result ? (
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
        result ? (
          <BottomPanel>
            <MetricsTable metrics={result.metrics} varData={varData} />
          </BottomPanel>
        ) : undefined
      }
    >
      {result ? (
        <EquityCurve data={result.equity_curve} title="Cumulative Returns" height={420} />
      ) : (
        <EmptyState />
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

function EmptyState() {
  return (
    <div className="flex items-center justify-center h-full">
      <div className="text-center">
        <div className="text-zinc-600 text-sm">Select a factor and run a backtest</div>
        <div className="text-zinc-700 text-xs mt-1">
          Results will appear here
        </div>
      </div>
    </div>
  );
}

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
