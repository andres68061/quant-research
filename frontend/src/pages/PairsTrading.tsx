import { useMutation } from "@tanstack/react-query";
import { useState } from "react";
import Plot from "react-plotly.js";

import KPICard from "@/components/cards/KPICard.tsx";
import EquityCurve from "@/components/charts/EquityCurve.tsx";
import RunButton from "@/components/controls/RunButton.tsx";
import AppLayout from "@/components/layout/AppLayout.tsx";
import LeftSidebar from "@/components/layout/LeftSidebar.tsx";
import RightSidebar from "@/components/layout/RightSidebar.tsx";
import { api } from "@/lib/api.ts";
import type { PairsBacktestResponse, PairsScreenResponse } from "@/lib/types.ts";
import { fmtPct, fmtRatio } from "@/lib/utils.ts";

const PLOTLY_LAYOUT: Partial<Plotly.Layout> = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "transparent",
  font: { color: "#a1a1aa", size: 10, family: "JetBrains Mono, monospace" },
  margin: { l: 48, r: 16, t: 32, b: 32 },
  xaxis: { gridcolor: "#27272a", linecolor: "#27272a" },
  yaxis: { gridcolor: "#27272a", linecolor: "#27272a" },
};

const SECTOR_PRESETS = [
  "Consumer Defensive",
  "Energy",
  "Technology",
  "Financial Services",
  "Healthcare",
  "Utilities",
];

export default function PairsTrading() {
  const [symbolY, setSymbolY] = useState("XOM");
  const [symbolX, setSymbolX] = useState("CVX");
  const [entryZ, setEntryZ] = useState(2.0);
  const [exitZ, setExitZ] = useState(0.5);
  const [hedgeWindow, setHedgeWindow] = useState(252);
  const [zWindow, setZWindow] = useState(60);
  const [tcost, setTcost] = useState(10);
  const [sector, setSector] = useState("Energy");
  const [maxSymbols, setMaxSymbols] = useState(10);
  const [screenMethod, setScreenMethod] = useState<"gatev" | "engle_granger">("gatev");
  const [useAdv, setUseAdv] = useState(true);
  const [validateOos, setValidateOos] = useState(true);
  const [trainFrac, setTrainFrac] = useState(0.6);

  const fiveYearsAgo = new Date(Date.now() - 5 * 365.25 * 86_400_000).toISOString().slice(0, 10);
  const todayStr = new Date().toISOString().slice(0, 10);
  const [startDate, setStartDate] = useState(fiveYearsAgo);
  const [endDate, setEndDate] = useState(todayStr);

  const mut = useMutation({
    mutationFn: () =>
      api.runPairsBacktest({
        symbol_y: symbolY.trim().toUpperCase(),
        symbol_x: symbolX.trim().toUpperCase(),
        start_date: startDate,
        end_date: endDate,
        hedge_window: hedgeWindow,
        zscore_window: zWindow,
        entry_z: entryZ,
        exit_z: exitZ,
        transaction_cost_bps: tcost,
        signal_lag_days: 1,
        train_frac: validateOos ? trainFrac : undefined,
      }),
  });

  const screenMut = useMutation({
    mutationFn: () =>
      api.screenPairs({
        sector: sector.trim(),
        method: screenMethod,
        use_adv: useAdv,
        max_symbols: maxSymbols,
        start_date: startDate,
        end_date: endDate,
        train_frac: screenMethod === "gatev" ? 0.67 : 0.6,
        min_train_corr: 0.5,
        max_train_adf_pvalue: 0.05,
        max_oos_backtests: 15,
        hedge_window: hedgeWindow,
        zscore_window: zWindow,
        entry_z: entryZ,
        exit_z: exitZ,
        transaction_cost_bps: tcost,
      }),
  });

  const data = mut.data as PairsBacktestResponse | undefined;
  const screen = screenMut.data as PairsScreenResponse | undefined;
  const eg = data?.diagnostics.engle_granger;
  const m = data?.metrics;

  const loadPair = (y: string, x: string) => {
    setSymbolY(y);
    setSymbolX(x);
  };

  return (
    <AppLayout
      left={
        <LeftSidebar>
          <p className="text-[11px] text-zinc-500 leading-relaxed mb-2">
            Engle–Granger pairs + Gatev SSD formation on an ADV-ranked same-sector universe.
            Formation window selects closest pairs; trading window scores OOS PnL only.
          </p>
          <Field label="Symbol Y (dependent)">
            <input
              value={symbolY}
              onChange={(e) => setSymbolY(e.target.value)}
              className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5 text-xs font-mono text-zinc-200"
            />
          </Field>
          <Field label="Symbol X (hedge)">
            <input
              value={symbolX}
              onChange={(e) => setSymbolX(e.target.value)}
              className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5 text-xs font-mono text-zinc-200"
            />
          </Field>
          <label className="flex items-center gap-2 mb-2 text-[11px] text-zinc-400">
            <input
              type="checkbox"
              checked={validateOos}
              onChange={(e) => setValidateOos(e.target.checked)}
              className="accent-blue-500"
            />
            Validate out-of-sample
          </label>
          {validateOos && (
            <>
              <p className="text-[10px] text-zinc-500 leading-relaxed mb-2">
                Splits Start–End below at train_frac. Diagnostics come from the
                train slice only; every metric/curve you see is computed on
                the held-out slice only — this pair/range can&apos;t
                self-mislead you.
              </p>
              <Field label={`Train fraction (${trainFrac.toFixed(2)})`}>
                <input
                  type="range"
                  min={0.2}
                  max={0.8}
                  step={0.05}
                  value={trainFrac}
                  onChange={(e) => setTrainFrac(+e.target.value)}
                  className="w-full accent-blue-500"
                />
              </Field>
            </>
          )}
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
          <Field label="Hedge window (days)">
            <input
              type="number"
              min={60}
              max={504}
              value={hedgeWindow}
              onChange={(e) => setHedgeWindow(+e.target.value)}
              className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5 text-xs font-mono text-zinc-200"
            />
          </Field>
          <Field label="Z-score window">
            <input
              type="number"
              min={20}
              max={252}
              value={zWindow}
              onChange={(e) => setZWindow(+e.target.value)}
              className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5 text-xs font-mono text-zinc-200"
            />
          </Field>
          <Field label="Cost (bps)">
            <input
              type="number"
              min={0}
              max={50}
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
          <RunButton onClick={() => mut.mutate()} loading={mut.isPending} label="Run pairs" />
          {mut.isError && (
            <p className="text-xs text-red-400 mt-2">{(mut.error as Error).message}</p>
          )}

          <div className="mt-4 pt-3 border-t border-zinc-800">
            <p className="text-[10px] uppercase tracking-wider text-zinc-500 mb-2">
              Universe screen
            </p>
            <Field label="Method">
              <select
                value={screenMethod}
                onChange={(e) =>
                  setScreenMethod(e.target.value as "gatev" | "engle_granger")
                }
                className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5 text-xs font-mono text-zinc-200"
              >
                <option value="gatev">Gatev SSD formation</option>
                <option value="engle_granger">Engle–Granger filter</option>
              </select>
            </Field>
            <Field label="Sector">
              <select
                value={sector}
                onChange={(e) => setSector(e.target.value)}
                className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5 text-xs font-mono text-zinc-200"
              >
                {SECTOR_PRESETS.map((s) => (
                  <option key={s} value={s}>
                    {s}
                  </option>
                ))}
              </select>
            </Field>
            <label className="flex items-center gap-2 mb-2 text-[11px] text-zinc-400">
              <input
                type="checkbox"
                checked={useAdv}
                onChange={(e) => setUseAdv(e.target.checked)}
                className="accent-blue-500"
              />
              Rank universe by dollar ADV
            </label>
            <Field label={`Max symbols (${maxSymbols})`}>
              <input
                type="range"
                min={4}
                max={12}
                step={1}
                value={maxSymbols}
                onChange={(e) => setMaxSymbols(+e.target.value)}
                className="w-full accent-blue-500"
              />
            </Field>
            <RunButton
              onClick={() => screenMut.mutate()}
              loading={screenMut.isPending}
              label="Screen sector"
            />
            {screenMut.isError && (
              <p className="text-xs text-red-400 mt-2">{(screenMut.error as Error).message}</p>
            )}
          </div>
        </LeftSidebar>
      }
      right={
        <RightSidebar>
          {m ? (
            <>
              {data.is_held_out && (
                <p className="text-[10px] uppercase tracking-wider text-emerald-400 mb-1">
                  Held-out mode
                </p>
              )}
              <KPICard label="Sharpe" value={fmtRatio(m.sharpe_ratio)} accent="neutral" />
              <KPICard
                label="Ann. return"
                value={fmtPct(m.annualized_return)}
                accent={m.annualized_return >= 0 ? "positive" : "negative"}
              />
              <KPICard label="Max DD" value={fmtPct(m.max_drawdown)} accent="negative" />
              <KPICard label="Pain ratio" value={fmtRatio(m.pain_ratio)} accent="neutral" />
              {eg && (
                <>
                  <KPICard
                    label={data.is_held_out ? "Held-out ADF p" : "ADF p-value"}
                    value={eg.adf_pvalue.toFixed(4)}
                    accent="neutral"
                  />
                  {data.is_held_out && data.train_diagnostics && (
                    <KPICard
                      label="Train ADF p"
                      value={data.train_diagnostics.adf_pvalue.toFixed(4)}
                      accent="neutral"
                    />
                  )}
                  <KPICard label="Hedge β" value={eg.hedge_ratio.toFixed(3)} accent="neutral" />
                  <KPICard
                    label="% days in trade"
                    value={fmtPct(data.diagnostics.pct_days_in_trade)}
                    accent="neutral"
                  />
                </>
              )}
            </>
          ) : (
            <p className="text-xs text-zinc-600">Run to see metrics + Engle–Granger diagnostics</p>
          )}
        </RightSidebar>
      }
    >
      <div className="flex flex-col gap-4 h-full overflow-y-auto p-1">
        {screen && (
          <div className="bg-zinc-900 border border-zinc-800 rounded overflow-x-auto">
            <div className="px-3 py-2 border-b border-zinc-800 text-[11px] text-zinc-500">
              {screen.method ?? "screen"} · {screen.symbols.join(", ")} · split {screen.split_date} ·
              tested {screen.n_pairs_tested} · formed {screen.n_pairs_passed_train} · OOS{" "}
              {screen.results.length}
            </div>
            {screen.results.length === 0 ? (
              <p className="text-xs text-zinc-600 p-3">
                No pairs formed with OOS history. Try another sector or loosen filters.
              </p>
            ) : (
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-[10px] uppercase tracking-wider text-zinc-500 border-b border-zinc-800">
                    <th className="text-left px-3 py-2">Pair</th>
                    <th className="text-right px-3 py-2">SSD</th>
                    <th className="text-right px-3 py-2">ADF p</th>
                    <th className="text-right px-3 py-2">OOS Sharpe</th>
                    <th className="text-right px-3 py-2">OOS ret</th>
                    <th className="text-right px-3 py-2">Max DD</th>
                    <th className="text-left px-3 py-2" />
                  </tr>
                </thead>
                <tbody>
                  {screen.results.map((r) => (
                    <tr
                      key={`${r.symbol_y}-${r.symbol_x}`}
                      className="border-b border-zinc-800/50 hover:bg-zinc-800/30"
                    >
                      <td className="px-3 py-2 font-mono text-zinc-200">
                        {r.symbol_y}/{r.symbol_x}
                      </td>
                      <td className="px-3 py-2 text-right font-mono tabular-nums text-zinc-300">
                        {r.formation_ssd != null ? r.formation_ssd.toFixed(2) : "—"}
                      </td>
                      <td className="px-3 py-2 text-right font-mono tabular-nums text-zinc-300">
                        {r.train_adf_pvalue != null ? r.train_adf_pvalue.toFixed(3) : "—"}
                      </td>
                      <td className="px-3 py-2 text-right font-mono tabular-nums text-zinc-200">
                        {fmtRatio(r.oos_sharpe)}
                      </td>
                      <td className="px-3 py-2 text-right font-mono tabular-nums text-zinc-300">
                        {fmtPct(r.oos_annualized_return)}
                      </td>
                      <td className="px-3 py-2 text-right font-mono tabular-nums text-red-400">
                        {fmtPct(r.oos_max_drawdown)}
                      </td>
                      <td className="px-3 py-2">
                        <button
                          type="button"
                          className="text-[10px] uppercase tracking-wider text-blue-400 hover:text-blue-300"
                          onClick={() => loadPair(r.symbol_y, r.symbol_x)}
                        >
                          Load
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        )}

        {!data ? (
          <div className="flex items-center justify-center min-h-[200px] text-xs text-zinc-600">
            Screen a sector or pick a pair and run the cointegration mean-reversion backtest
          </div>
        ) : (
          <>
            {data.is_held_out && (
              <div className="bg-emerald-950/20 border border-emerald-900/40 rounded px-3 py-2 text-[11px] text-emerald-400 leading-relaxed">
                Train {data.train_start_date}..{data.train_end_date} (diagnostic
                only, never traded) → Held-out {data.held_out_start_date}..
                {endDate} (every number below is computed on this slice only).
              </div>
            )}
            <EquityCurve data={data.equity_curve} title="Pairs cumulative return" height={320} />
            <Plot
              data={[
                {
                  x: data.spread_series.map((p) => p.date),
                  y: data.spread_series.map((p) => p.zscore),
                  type: "scatter",
                  mode: "lines",
                  name: "Spread z",
                  line: { color: "#3b82f6", width: 1.2 },
                },
                {
                  x: data.spread_series.map((p) => p.date),
                  y: data.spread_series.map((p) => p.position),
                  type: "scatter",
                  mode: "lines",
                  name: "Position",
                  yaxis: "y2",
                  line: { color: "#34d399", width: 1.2 },
                },
              ]}
              layout={{
                ...PLOTLY_LAYOUT,
                title: {
                  text: "Spread z-score and position",
                  font: { size: 12, color: "#71717a" },
                  x: 0,
                } as Partial<Plotly.Layout["title"]>,
                yaxis: { ...PLOTLY_LAYOUT.yaxis, title: "z" },
                yaxis2: {
                  overlaying: "y",
                  side: "right",
                  gridcolor: "#27272a",
                  title: "pos",
                  range: [-1.5, 1.5],
                },
                height: 280,
                showlegend: true,
              }}
              config={{ displayModeBar: false, responsive: true }}
              className="w-full"
              useResizeHandler
              style={{ width: "100%" }}
            />
            <p className="text-[11px] text-zinc-500 leading-relaxed">
              {data.is_held_out
                ? `Engle–Granger ADF p=${eg?.adf_pvalue.toFixed(4)} above is re-tested on the held-out slice only (the train slice's ADF p=${data.train_diagnostics?.adf_pvalue.toFixed(4)} was never used to pick this pair's live PnL). `
                : `Full-sample Engle–Granger ADF p=${eg?.adf_pvalue.toFixed(4)} is diagnostic only, computed over the same range as the PnL below — turn on "Validate out-of-sample" to stop that from self-misleading you. `}
              Trading uses a rolling {data.diagnostics.hedge_window}d hedge and{" "}
              {data.diagnostics.zscore_window}d z-score with entry |z|≥{data.diagnostics.entry_z} and
              exit |z|≤{data.diagnostics.exit_z}.
            </p>
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
