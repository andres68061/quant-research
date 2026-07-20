import { useMutation } from "@tanstack/react-query";
import { useState } from "react";

import KPICard from "@/components/cards/KPICard.tsx";
import EquityCurve from "@/components/charts/EquityCurve.tsx";
import RunButton from "@/components/controls/RunButton.tsx";
import AppLayout from "@/components/layout/AppLayout.tsx";
import LeftSidebar from "@/components/layout/LeftSidebar.tsx";
import RightSidebar from "@/components/layout/RightSidebar.tsx";
import { api } from "@/lib/api.ts";
import type { PairsPersistentBacktestResponse } from "@/lib/types.ts";
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

export default function PairsPersistent() {
  const [sectorNames, setSectorNames] = useState<string[]>([
    "Energy",
    "Utilities",
    "Financial Services",
    "Technology",
    "Real Estate",
  ]);
  const [formationMonths, setFormationMonths] = useState(60);
  const [rescreenMonths, setRescreenMonths] = useState(12);
  const [topNPairs, setTopNPairs] = useState(10);
  const [minCrossings, setMinCrossings] = useState(3);
  const [persistenceChecks, setPersistenceChecks] = useState(4);
  const [tcost, setTcost] = useState(10);
  const [freezeHedge, setFreezeHedge] = useState(false);

  const [startDate, setStartDate] = useState("2011-01-01");
  const [endDate, setEndDate] = useState(() => new Date().toISOString().slice(0, 10));

  const toggleSector = (s: string) => {
    setSectorNames((prev) => (prev.includes(s) ? prev.filter((x) => x !== s) : [...prev, s]));
  };

  const mut = useMutation({
    mutationFn: () =>
      api.runPairsPersistentBacktest({
        sector_names: sectorNames,
        start_date: startDate,
        end_date: endDate,
        formation_months: formationMonths,
        rescreen_months: rescreenMonths,
        top_n_pairs: topNPairs,
        min_crossings: minCrossings,
        persistence_checks: persistenceChecks,
        transaction_cost_bps: tcost,
        freeze_hedge_in_trade: freezeHedge,
        use_adv: true,
      }),
  });

  const data = mut.data as PairsPersistentBacktestResponse | undefined;
  const m = data?.metrics;
  const stoppedEarly = data ? data.pair_history.filter((h) => h.stopped_early).length : 0;

  return (
    <AppLayout
      left={
        <LeftSidebar>
          <p className="text-[11px] text-zinc-500 leading-relaxed mb-2">
            Event-driven pairs basket: candidates must be cointegrated{" "}
            <span className="text-zinc-300">and</span> have price paths that cross with real
            amplitude over the trailing <span className="text-zinc-300">formation</span> window.
            Each pair trades until a rolling cointegration monitor fails{" "}
            <span className="text-zinc-300">{persistenceChecks}</span> consecutive checks; free
            slots re-fill every <span className="text-zinc-300">rescreen</span> months.
          </p>
          <p className="text-[11px] text-amber-400/80 leading-relaxed mb-3 border border-amber-900/40 bg-amber-950/20 rounded px-2 py-1.5">
            Not a validated edge. Re-screening annually with a long lookback is sign-stable across
            shifted start dates (Sharpe ≈ 0.1–0.5, market beta ≈ 0) but remains below the
            deflated-Sharpe bar (~0.6) set by the number of configurations this repo has tried.
            Move the start date and see for yourself — fragility is the finding.
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

          <Field label="Formation lookback (months)">
            <input
              type="number"
              min={6}
              max={72}
              value={formationMonths}
              onChange={(e) => setFormationMonths(+e.target.value)}
              className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5 text-xs font-mono text-zinc-200"
            />
          </Field>
          <Field label="Re-screen every (months)">
            <input
              type="number"
              min={3}
              max={72}
              value={rescreenMonths}
              onChange={(e) => setRescreenMonths(+e.target.value)}
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
          <Field label={`Min path crossings (${minCrossings})`}>
            <input
              type="range"
              min={1}
              max={15}
              step={1}
              value={minCrossings}
              onChange={(e) => setMinCrossings(+e.target.value)}
              className="w-full accent-blue-500"
            />
          </Field>
          <Field label={`Stop after N failed checks (${persistenceChecks})`}>
            <input
              type="range"
              min={1}
              max={8}
              step={1}
              value={persistenceChecks}
              onChange={(e) => setPersistenceChecks(+e.target.value)}
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
          <label className="flex items-center gap-2 text-[11px] text-zinc-400 mb-2">
            <input
              type="checkbox"
              checked={freezeHedge}
              onChange={(e) => setFreezeHedge(e.target.checked)}
              className="accent-blue-500"
            />
            Freeze hedge at entry (no daily re-hedge cost)
          </label>
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
            label="Run persistence backtest"
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
              <KPICard
                label="Cid-1 Ratio"
                value={fmtRatio(m.cid1_ratio)}
                accent={m.cid1_ratio >= 0 ? "positive" : "negative"}
              />
              <KPICard
                label="Cid-2 Ratio"
                value={fmtRatio(m.cid2_ratio)}
                accent={m.cid2_ratio >= 0 ? "positive" : "negative"}
              />
              <KPICard label="Sharpe" value={fmtRatio(m.sharpe_ratio)} accent="neutral" />
              <KPICard
                label="Total return"
                value={fmtPct(m.total_return)}
                accent={m.total_return >= 0 ? "positive" : "negative"}
              />
              <KPICard label="Max DD" value={fmtPct(m.max_drawdown)} accent="negative" />
              <KPICard
                label="Pairs traded"
                value={String(data?.pair_history.length ?? 0)}
                accent="neutral"
              />
              <KPICard
                label="Stopped by monitor"
                value={`${stoppedEarly}/${data?.pair_history.length ?? 0}`}
                accent="neutral"
              />
            </>
          ) : (
            <p className="text-xs text-zinc-600">Run to see index metrics + pair lifecycle</p>
          )}
        </RightSidebar>
      }
    >
      <div className="flex flex-col gap-4 h-full overflow-y-auto p-1">
        {!data ? (
          <div className="flex items-center justify-center min-h-[200px] text-xs text-zinc-600">
            Pick sectors and run the cointegration-persistence backtest
          </div>
        ) : (
          <>
            <EquityCurve
              data={data.equity_curve}
              title="Persistence pairs index — growth of $1"
              height={320}
            />
            <div className="bg-zinc-900 border border-zinc-800 rounded overflow-x-auto">
              <div className="px-3 py-2 border-b border-zinc-800 text-[11px] text-zinc-500">
                {data.pair_history.length} pairs ever traded · {data.screens.length} screening
                rounds
              </div>
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-[10px] uppercase tracking-wider text-zinc-500 border-b border-zinc-800">
                    <th className="text-left px-3 py-2">Pair</th>
                    <th className="text-left px-3 py-2">Sector</th>
                    <th className="text-right px-3 py-2">Form. ADF p</th>
                    <th className="text-right px-3 py-2">Crossings</th>
                    <th className="text-left px-3 py-2">Traded</th>
                    <th className="text-right px-3 py-2">Days</th>
                    <th className="text-left px-3 py-2">Stop</th>
                  </tr>
                </thead>
                <tbody>
                  {data.pair_history.map((h) => (
                    <tr
                      key={`${h.symbol_y}/${h.symbol_x}@${h.trading_start}`}
                      className="border-b border-zinc-800/50 hover:bg-zinc-800/30"
                    >
                      <td className="px-3 py-2 font-mono text-zinc-300 whitespace-nowrap">
                        {h.symbol_y}/{h.symbol_x}
                      </td>
                      <td className="px-3 py-2 text-zinc-400 whitespace-nowrap">{h.sector}</td>
                      <td className="px-3 py-2 text-right font-mono tabular-nums text-zinc-300">
                        {h.formation_adf_pvalue.toExponential(1)}
                      </td>
                      <td className="px-3 py-2 text-right font-mono tabular-nums text-zinc-300">
                        {h.formation_crossings}
                      </td>
                      <td className="px-3 py-2 font-mono text-zinc-400 whitespace-nowrap">
                        {h.trading_start} → {h.stop_date ?? "end"}
                      </td>
                      <td className="px-3 py-2 text-right font-mono tabular-nums text-zinc-300">
                        {h.n_days}
                      </td>
                      <td className="px-3 py-2 whitespace-nowrap">
                        {h.stopped_early ? (
                          <span className="text-red-400">monitor stop</span>
                        ) : (
                          <span className="text-zinc-500">ran to end</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="bg-zinc-900 border border-zinc-800 rounded overflow-x-auto">
              <div className="px-3 py-2 border-b border-zinc-800 text-[11px] text-zinc-500">
                Screening rounds (lookback {formationMonths}mo, cadence {rescreenMonths}mo)
              </div>
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-[10px] uppercase tracking-wider text-zinc-500 border-b border-zinc-800">
                    <th className="text-left px-3 py-2">Formation window</th>
                    <th className="text-right px-3 py-2">Active before</th>
                    <th className="text-right px-3 py-2">Free slots</th>
                    <th className="text-right px-3 py-2">Candidates</th>
                    <th className="text-right px-3 py-2">Selected</th>
                  </tr>
                </thead>
                <tbody>
                  {data.screens.map((s) => (
                    <tr
                      key={s.formation_end}
                      className="border-b border-zinc-800/50 hover:bg-zinc-800/30"
                    >
                      <td className="px-3 py-2 font-mono text-zinc-300 whitespace-nowrap">
                        {s.formation_start} .. {s.formation_end}
                      </td>
                      <td className="px-3 py-2 text-right font-mono tabular-nums text-zinc-300">
                        {s.active_before}
                      </td>
                      <td className="px-3 py-2 text-right font-mono tabular-nums text-zinc-300">
                        {s.free_slots}
                      </td>
                      <td className="px-3 py-2 text-right font-mono tabular-nums text-zinc-300">
                        {s.n_candidates_found}
                      </td>
                      <td className="px-3 py-2 text-right font-mono tabular-nums text-zinc-300">
                        {s.n_selected}
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
