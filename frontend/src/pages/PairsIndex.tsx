import { useMutation } from "@tanstack/react-query";
import { useState } from "react";

import KPICard from "@/components/cards/KPICard.tsx";
import EquityCurve from "@/components/charts/EquityCurve.tsx";
import RunButton from "@/components/controls/RunButton.tsx";
import AppLayout from "@/components/layout/AppLayout.tsx";
import LeftSidebar from "@/components/layout/LeftSidebar.tsx";
import RightSidebar from "@/components/layout/RightSidebar.tsx";
import { api } from "@/lib/api.ts";
import type { PairsIndexBacktestResponse } from "@/lib/types.ts";
import { fmtPct, fmtRatio } from "@/lib/utils.ts";

const SECTOR_OPTIONS = [
  "Consumer Defensive",
  "Financial Services",
  "Utilities",
  "Energy",
  "Real Estate",
  "Basic Materials",
  "Technology",
  "Industrials",
  "Communication Services",
  "Healthcare",
];

export default function PairsIndex() {
  const [sectorNames, setSectorNames] = useState<string[]>([
    "Energy",
    "Utilities",
    "Financial Services",
  ]);
  const [formationMonths, setFormationMonths] = useState(12);
  const [tradingMonths, setTradingMonths] = useState(6);
  const [topNPairs, setTopNPairs] = useState(10);
  const [maxSymbolsPerSector, setMaxSymbolsPerSector] = useState(12);
  const [hedgeWindow, setHedgeWindow] = useState(252);
  const [zWindow, setZWindow] = useState(60);
  const [entryZ, setEntryZ] = useState(2.0);
  const [exitZ, setExitZ] = useState(0.5);
  const [tcost, setTcost] = useState(10);

  const [startDate, setStartDate] = useState(() =>
    new Date(Date.now() - 14 * 365.25 * 86_400_000).toISOString().slice(0, 10),
  );
  const [endDate, setEndDate] = useState(() => new Date().toISOString().slice(0, 10));

  const toggleSector = (s: string) => {
    setSectorNames((prev) => (prev.includes(s) ? prev.filter((x) => x !== s) : [...prev, s]));
  };

  const mut = useMutation({
    mutationFn: () =>
      api.runPairsIndexBacktest({
        sector_names: sectorNames,
        start_date: startDate,
        end_date: endDate,
        formation_months: formationMonths,
        trading_months: tradingMonths,
        top_n_pairs: topNPairs,
        max_symbols_per_sector: maxSymbolsPerSector,
        use_adv: true,
        hedge_window: hedgeWindow,
        zscore_window: zWindow,
        entry_z: entryZ,
        exit_z: exitZ,
        transaction_cost_bps: tcost,
        signal_lag_days: 1,
      }),
  });

  const data = mut.data as PairsIndexBacktestResponse | undefined;
  const m = data?.metrics;
  const avgPairsHeld = data
    ? data.periods.reduce((s, p) => s + p.avg_active_pairs, 0) / Math.max(data.periods.length, 1)
    : 0;

  return (
    <AppLayout
      left={
        <LeftSidebar>
          <p className="text-[11px] text-zinc-500 leading-relaxed mb-2">
            Rolling multi-pair basket (Gatev SSD formation): re-forms every{" "}
            <span className="text-zinc-300">trading_months</span> from the trailing{" "}
            <span className="text-zinc-300">formation_months</span> window, equal-weights the top-N
            pairs' rolling-hedge/z-score returns into one index.
          </p>
          <p className="text-[11px] text-amber-400/80 leading-relaxed mb-3 border border-amber-900/40 bg-amber-950/20 rounded px-2 py-1.5">
            On this platform's real data (2012-2026), the systematic basket lost money net of
            costs under every formation criterion tested (notebook 18) and underperformed the
            single hand-vetted XOM/CVX pair. Treat this as a research tool, not a strategy with
            demonstrated live edge.
          </p>

          <div className="mb-3">
            <label className="block text-[10px] uppercase tracking-wider text-zinc-500 mb-1">
              Sectors ({sectorNames.length})
            </label>
            <div className="grid grid-cols-1 gap-1 max-h-40 overflow-y-auto border border-zinc-800 rounded p-2">
              {SECTOR_OPTIONS.map((s) => (
                <label key={s} className="flex items-center gap-2 text-[11px] text-zinc-400">
                  <input
                    type="checkbox"
                    checked={sectorNames.includes(s)}
                    onChange={() => toggleSector(s)}
                    className="accent-blue-500"
                  />
                  {s}
                </label>
              ))}
            </div>
          </div>

          <Field label="Formation (months)">
            <input
              type="number"
              min={3}
              max={36}
              value={formationMonths}
              onChange={(e) => setFormationMonths(+e.target.value)}
              className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5 text-xs font-mono text-zinc-200"
            />
          </Field>
          <Field label="Trading (months)">
            <input
              type="number"
              min={1}
              max={12}
              value={tradingMonths}
              onChange={(e) => setTradingMonths(+e.target.value)}
              className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5 text-xs font-mono text-zinc-200"
            />
          </Field>
          <Field label={`Top N pairs (${topNPairs})`}>
            <input
              type="range"
              min={1}
              max={20}
              step={1}
              value={topNPairs}
              onChange={(e) => setTopNPairs(+e.target.value)}
              className="w-full accent-blue-500"
            />
          </Field>
          <Field label={`Max symbols / sector (${maxSymbolsPerSector})`}>
            <input
              type="range"
              min={4}
              max={15}
              step={1}
              value={maxSymbolsPerSector}
              onChange={(e) => setMaxSymbolsPerSector(+e.target.value)}
              className="w-full accent-blue-500"
            />
          </Field>
          <Field label="Hedge window (days)">
            <input
              type="number"
              min={30}
              max={1000}
              value={hedgeWindow}
              onChange={(e) => setHedgeWindow(+e.target.value)}
              className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5 text-xs font-mono text-zinc-200"
            />
          </Field>
          <Field label="Z-score window">
            <input
              type="number"
              min={10}
              max={500}
              value={zWindow}
              onChange={(e) => setZWindow(+e.target.value)}
              className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5 text-xs font-mono text-zinc-200"
            />
          </Field>
          <Field label={`Entry |z| (${entryZ.toFixed(1)})`}>
            <input
              type="range"
              min={1}
              max={3.5}
              step={0.1}
              value={entryZ}
              onChange={(e) => setEntryZ(+e.target.value)}
              className="w-full accent-blue-500"
            />
          </Field>
          <Field label={`Exit |z| (${exitZ.toFixed(1)})`}>
            <input
              type="range"
              min={0}
              max={1.5}
              step={0.1}
              value={exitZ}
              onChange={(e) => setExitZ(+e.target.value)}
              className="w-full accent-blue-500"
            />
          </Field>
          <Field label="Cost (bps)">
            <input
              type="number"
              min={0}
              max={200}
              value={tcost}
              onChange={(e) => setTcost(+e.target.value)}
              className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5 text-xs font-mono text-zinc-200"
            />
          </Field>
          <Field label="Start">
            <input
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5 text-xs font-mono text-zinc-200"
            />
          </Field>
          <Field label="End">
            <input
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5 text-xs font-mono text-zinc-200"
            />
          </Field>
          <RunButton
            onClick={() => mut.mutate()}
            loading={mut.isPending}
            label="Run index backtest"
            disabled={sectorNames.length === 0}
          />
          {mut.isError && (
            <p className="text-xs text-red-400 mt-2">{(mut.error as Error).message}</p>
          )}
        </LeftSidebar>
      }
      right={
        <RightSidebar>
          {m ? (
            <>
              <KPICard label="Index Sharpe" value={fmtRatio(m.sharpe_ratio)} accent="neutral" />
              <KPICard
                label="Ann. return"
                value={fmtPct(m.annualized_return)}
                accent={m.annualized_return >= 0 ? "positive" : "negative"}
              />
              <KPICard label="Max DD" value={fmtPct(m.max_drawdown)} accent="negative" />
              <KPICard label="Periods" value={String(data?.periods.length ?? 0)} accent="neutral" />
              <KPICard label="Avg pairs held" value={avgPairsHeld.toFixed(1)} accent="neutral" />
              <KPICard label="Universe size" value={String(data?.universe.length ?? 0)} accent="neutral" />
            </>
          ) : (
            <p className="text-xs text-zinc-600">Run to see index metrics + per-period basket</p>
          )}
        </RightSidebar>
      }
    >
      <div className="flex flex-col gap-4 h-full overflow-y-auto p-1">
        {!data ? (
          <div className="flex items-center justify-center min-h-[200px] text-xs text-zinc-600">
            Pick sectors and run the rolling pairs-index backtest
          </div>
        ) : (
          <>
            <EquityCurve data={data.equity_curve} title="Pairs stat-arb index — growth of $1" height={320} />
            <div className="bg-zinc-900 border border-zinc-800 rounded overflow-x-auto">
              <div className="px-3 py-2 border-b border-zinc-800 text-[11px] text-zinc-500">
                {data.periods.length} rolling periods · universe {data.universe.length} symbols
              </div>
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-[10px] uppercase tracking-wider text-zinc-500 border-b border-zinc-800">
                    <th className="text-left px-3 py-2">Trading window</th>
                    <th className="text-right px-3 py-2">Candidates</th>
                    <th className="text-right px-3 py-2">Selected</th>
                    <th className="text-right px-3 py-2">Period Sharpe</th>
                    <th className="text-left px-3 py-2">Pairs</th>
                  </tr>
                </thead>
                <tbody>
                  {data.periods.map((p) => (
                    <tr
                      key={`${p.trading_start}-${p.trading_end}`}
                      className="border-b border-zinc-800/50 hover:bg-zinc-800/30"
                    >
                      <td className="px-3 py-2 font-mono text-zinc-300 whitespace-nowrap">
                        {p.trading_start} .. {p.trading_end}
                      </td>
                      <td className="px-3 py-2 text-right font-mono tabular-nums text-zinc-300">
                        {p.n_candidates_formed}
                      </td>
                      <td className="px-3 py-2 text-right font-mono tabular-nums text-zinc-300">
                        {p.n_pairs_selected}
                      </td>
                      <td
                        className={`px-3 py-2 text-right font-mono tabular-nums ${
                          (p.blended_sharpe ?? 0) >= 0 ? "text-emerald-400" : "text-red-400"
                        }`}
                      >
                        {p.blended_sharpe != null ? fmtRatio(p.blended_sharpe) : "—"}
                      </td>
                      <td className="px-3 py-2 font-mono text-zinc-400">
                        {p.selected_pairs.map((r) => `${r.symbol_y}/${r.symbol_x}`).join(", ")}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </div>
    </AppLayout>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="mb-2">
      <label className="block text-[10px] uppercase tracking-wider text-zinc-500 mb-1">{label}</label>
      {children}
    </div>
  );
}
