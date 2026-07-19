/**
 * Unified portfolio workbench.
 *
 * Replaces the former ``ETFOptimizer`` and ``ManualPortfolioBuilder`` pages
 * with a single workflow driven by two orthogonal switches:
 *
 *   1. Universe — where the candidate symbols come from:
 *        - "etf"     : the ETF picker (filterable list of ETF assets, one
 *                      checkbox each, optionally restricted to the default
 *                      basket and to symbols with enough solo history).
 *        - "stocks"  : the sector/industry-aware stock picker
 *                      (sector filter + free-text search + add-buttons).
 *
 *   2. Weighting — how those symbols turn into a weight vector:
 *        - "optimizer" : run mean–variance, then simulate at the tangency
 *                        portfolio. Surfaces the efficient frontier chart
 *                        and a tangency KPI block.
 *        - "equal"     : 1/N over the selection.
 *        - "manual"    : user-entered weights, normalised to sum to 1.
 *
 * Everything else — eligibility checks, benchmark overlay, NAV/drawdown
 * charts, KPI rails, composition table — is shared infrastructure rendered
 * the same way regardless of mode.
 */
import { useMutation, useQuery } from "@tanstack/react-query";
import { useCallback, useEffect, useMemo, useState } from "react";
import Plot from "react-plotly.js";

import KPICard from "@/components/cards/KPICard.tsx";
import RunButton from "@/components/controls/RunButton.tsx";
import AppLayout from "@/components/layout/AppLayout.tsx";
import BottomPanel from "@/components/layout/BottomPanel.tsx";
import LeftSidebar from "@/components/layout/LeftSidebar.tsx";
import RightSidebar from "@/components/layout/RightSidebar.tsx";
import BenchmarkKPIs from "@/components/portfolio/BenchmarkKPIs.tsx";
import DrawdownChart from "@/components/portfolio/DrawdownChart.tsx";
import EligibilityBanner from "@/components/portfolio/EligibilityBanner.tsx";
import NAVChart from "@/components/portfolio/NAVChart.tsx";
import SimMetricsKPIs from "@/components/portfolio/SimMetricsKPIs.tsx";
import { useEligibility } from "@/hooks/useEligibility.ts";
import { api } from "@/lib/api.ts";
import { DARK_PLOTLY_LAYOUT } from "@/lib/plotlyTheme.ts";
import type {
  BenchmarkResponse,
  OptimizeResponse,
  PortfolioCoverageEntry,
  SectorBreakdownResponse,
  SectorSummaryResponse,
  SimulateResponse,
} from "@/lib/types.ts";
import { cn, fmtPct, fmtRatio } from "@/lib/utils.ts";

type BrowseTab = "etf" | "stocks";
type Weighting = "optimizer" | "equal" | "manual";
type ChartTab = "frontier" | "nav" | "drawdown";

const DEFAULT_ETFS = [
  "VOO", "SOXX", "ITA", "DTEC", "IXJ", "IYK",
  "AIRR", "UGA", "MLPX", "GREK", "ARGT", "GLD",
];

const BENCHMARK_TYPES = [
  "None",
  "S&P 500 (^GSPC)",
  "S&P 500 Reconstructed (2020+)",
  "Equal Weight Universe",
  "Synthetic (Custom Mix)",
] as const;

const REBALANCE_OPTIONS = ["Annual", "Quarterly", "Monthly"] as const;

function defaultStartDateIso(): string {
  const d = new Date();
  d.setFullYear(d.getFullYear() - 5);
  return d.toISOString().slice(0, 10);
}

export default function Portfolio() {
  // ── Mode switches ────────────────────────────────────────────
  const [browseTab, setBrowseTab] = useState<BrowseTab>("etf");
  const [weighting, setWeighting] = useState<Weighting>("optimizer");

  // ── Shared run params ────────────────────────────────────────
  const [startDate, setStartDate] = useState<string>(defaultStartDateIso);
  const [endDate, setEndDate] = useState<string>("");
  const [selected, setSelected] = useState<string[]>([]);
  const [manualWeights, setManualWeights] = useState<Record<string, number>>({});
  const [rebalFreq, setRebalFreq] = useState<string>("Annual");
  const [benchmarkType, setBenchmarkType] = useState<string>("None");

  // ── Optimizer-specific ───────────────────────────────────────
  const [rfRate, setRfRate] = useState(8.15);
  const [borrowRate, setBorrowRate] = useState(11.15);

  // ── Picker UI state ──────────────────────────────────────────
  const [etfFilter, setEtfFilter] = useState("");
  const [sectorFilter, setSectorFilter] = useState<string[]>([]);
  const [stockSearch, setStockSearch] = useState("");
  const [activeTab, setActiveTab] = useState<ChartTab>("nav");

  const elig = useEligibility({ startDate, selected, setSelected });
  const {
    assetsQuery,
    rowCountsQuery,
    jointQuery,
    priceSymbols,
    minReq,
    lastPanelDate,
    coverage,
    jointReady,
    jointOk,
  } = elig;

  // Threshold for "addable but visually addable" depends on the weighting mode:
  // optimizer needs the joint-history minimum (≈60 rows), while equal/manual
  // can run on as little as a couple of dozen days.
  const addableThreshold = weighting === "optimizer" ? minReq : 5;

  // A symbol counts as "still trading" if its last observation is within
  // ~30 trading days of the panel tail. Anything older we mark as delisted.
  const STALE_DAYS = 45;
  const isLive = useCallback(
    (sym: string) => {
      const cov = coverage(sym);
      if (!cov || !lastPanelDate) return true;
      const lastTrade = new Date(cov.last).getTime();
      const tail = new Date(lastPanelDate).getTime();
      return (tail - lastTrade) / 86_400_000 <= STALE_DAYS;
    },
    [coverage, lastPanelDate],
  );

  // ── ETF universe ─────────────────────────────────────────────
  const etfAssets = useMemo(
    () =>
      (assetsQuery.data?.assets ?? []).filter(
        (a) => a.type === "ETF" || DEFAULT_ETFS.includes(a.symbol),
      ),
    [assetsQuery.data],
  );
  // We no longer drop ineligible ETFs from the list — they're shown dimmed so
  // the user can see what's there but understands why they can't pick it.
  const etfFiltered = etfFilter
    ? etfAssets.filter((a) => a.symbol.includes(etfFilter.toUpperCase()))
    : etfAssets;

  // ── Stock universe ───────────────────────────────────────────
  const sectorsQuery = useQuery({
    queryKey: ["sectorSummary"],
    queryFn: api.getSectorSummary,
  });
  const breakdownQuery = useQuery({
    queryKey: ["sectorBreakdown"],
    queryFn: () => api.getSectorBreakdown(),
  });
  const sectors: SectorSummaryResponse | undefined = sectorsQuery.data;
  const breakdown: SectorBreakdownResponse | undefined = breakdownQuery.data;

  const stockOptions = useMemo(() => {
    if (!breakdown?.symbols || !assetsQuery.isSuccess) return [];
    let stocks = breakdown.symbols.filter((s) => priceSymbols.has(s.symbol));
    if (sectorFilter.length > 0) {
      stocks = stocks.filter((s) => sectorFilter.includes(s.sector));
    }
    if (stockSearch) {
      const q = stockSearch.toUpperCase();
      stocks = stocks.filter(
        (s) =>
          s.symbol.includes(q) ||
          s.sector.toUpperCase().includes(q) ||
          s.industry.toUpperCase().includes(q),
      );
    }
    return stocks;
  }, [breakdown, sectorFilter, stockSearch, assetsQuery.isSuccess, priceSymbols]);

  // ── Weights ──────────────────────────────────────────────────
  const computedWeights = useMemo(() => {
    if (selected.length === 0) return {};
    if (weighting === "equal") {
      const w = 1 / selected.length;
      return Object.fromEntries(selected.map((s) => [s, w]));
    }
    if (weighting === "manual") {
      const raw: Record<string, number> = {};
      let total = 0;
      for (const sym of selected) {
        const v = manualWeights[sym] ?? 1;
        raw[sym] = v;
        total += v;
      }
      if (total === 0) return Object.fromEntries(selected.map((s) => [s, 0]));
      return Object.fromEntries(Object.entries(raw).map(([k, v]) => [k, v / total]));
    }
    // optimizer mode: weights filled in after optimize succeeds.
    return {};
  }, [selected, weighting, manualWeights]);

  // ── Mutations ────────────────────────────────────────────────
  const optimizeMut = useMutation({
    mutationFn: () =>
      api.optimizePortfolio({
        symbols: selected,
        start_date: startDate,
        risk_free_rate: rfRate / 100,
        borrowing_rate: borrowRate / 100,
      }),
  });

  const simMut = useMutation({
    mutationFn: (weights: Record<string, number>) =>
      api.simulatePortfolio({
        symbols: selected,
        weights,
        freq: rebalFreq,
        start_date: startDate,
        end_date: endDate || undefined,
      }),
  });

  const benchMut = useMutation({
    mutationFn: () =>
      api.getBenchmarkReturns({
        benchmark_type: benchmarkType,
        start_date: startDate,
        end_date: endDate || new Date().toISOString().slice(0, 10),
      }),
  });

  const opt: OptimizeResponse | undefined = optimizeMut.data;
  const sim: SimulateResponse | undefined = simMut.data;
  const bench: BenchmarkResponse | undefined = benchMut.data;
  const wantsBench = benchmarkType !== "None";

  // ── Run-button gating ────────────────────────────────────────
  const minSelected = weighting === "optimizer" ? 2 : 1;
  const canRun =
    selected.length >= minSelected &&
    (selected.length < 2 || jointReady) &&
    (selected.length < 2 || jointOk);

  const loading = optimizeMut.isPending || simMut.isPending || benchMut.isPending;

  const handleRun = useCallback(() => {
    if (!canRun) return;
    if (wantsBench) benchMut.mutate();
    if (weighting === "optimizer") {
      optimizeMut.mutate(undefined, {
        onSuccess: (res) => simMut.mutate(res.tangency.weights),
      });
    } else {
      simMut.mutate(computedWeights);
    }
  }, [canRun, wantsBench, weighting, optimizeMut, simMut, benchMut, computedWeights]);

  // ── Selection helpers ────────────────────────────────────────
  const toggleSym = (sym: string) =>
    setSelected((prev) => (prev.includes(sym) ? prev.filter((s) => s !== sym) : [...prev, sym]));
  const addSym = (sym: string) =>
    setSelected((prev) => (prev.includes(sym) ? prev : [...prev, sym]));
  const removeSym = (sym: string) => {
    setSelected((prev) => prev.filter((s) => s !== sym));
    setManualWeights((prev) => {
      const next = { ...prev };
      delete next[sym];
      return next;
    });
  };
  const toggleSector = (sector: string) =>
    setSectorFilter((prev) =>
      prev.includes(sector) ? prev.filter((s) => s !== sector) : [...prev, sector],
    );

  const fetchCetes = () => {
    api.getCetes28().then((d) => setRfRate(d.rate)).catch(() => {});
  };

  // Auto-pick a sensible chart tab when results arrive.
  useEffect(() => {
    if (opt && weighting === "optimizer" && activeTab !== "frontier" && !sim) {
      setActiveTab("frontier");
    }
  }, [opt, weighting, sim, activeTab]);

  // Set of ETF tickers for fast row classification.
  const etfSymbolSet = useMemo(
    () => new Set(etfAssets.map((a) => a.symbol)),
    [etfAssets],
  );

  // ── Composition rows for bottom panel ────────────────────────
  const compositionRows = useMemo(() => {
    if (!sim || selected.length === 0) return [];
    const weights = optimizerWeights(opt) ?? computedWeights;
    return selected.map((sym) => {
      const info = breakdown?.symbols?.find((s) => s.symbol === sym);
      const isEtf = etfSymbolSet.has(sym);
      const cov = coverage(sym);
      return {
        symbol: sym,
        kind: isEtf ? "ETF" : "Stock",
        sector: info?.sector ?? "—",
        industry: info?.industry ?? (isEtf ? "ETF" : "—"),
        weight: weights[sym] ?? 0,
        first: cov?.first,
        last: cov?.last,
        live: isLive(sym),
      };
    });
  }, [sim, selected, opt, computedWeights, breakdown, etfSymbolSet, coverage, isLive]);

  // ── Layout ───────────────────────────────────────────────────
  return (
    <AppLayout
      left={
        <LeftSidebar>
          <ModeSwitch
            label="Weighting"
            value={weighting}
            options={[
              { id: "optimizer", label: "Optimizer" },
              { id: "equal", label: "Equal" },
              { id: "manual", label: "Manual" },
            ]}
            onChange={setWeighting}
          />

          <Field label="Date Range">
            <div className="flex gap-2">
              <input
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                className="flex-1 bg-zinc-900 border border-zinc-800 rounded px-2 py-1 text-xs text-zinc-200 font-mono"
              />
              <input
                type="date"
                value={endDate}
                placeholder="end (optional)"
                onChange={(e) => setEndDate(e.target.value)}
                className="flex-1 bg-zinc-900 border border-zinc-800 rounded px-2 py-1 text-xs text-zinc-200 font-mono"
              />
            </div>
            <p className="text-[10px] text-zinc-600 mt-1 leading-snug">
              Eligibility filters use the start date. Only symbols with ≥ {minReq} solo trading
              days in range qualify.
            </p>
          </Field>

          {rowCountsQuery.isError && (
            <Notice tone="error">Could not load price coverage</Notice>
          )}

          <div>
            <label className="text-[10px] uppercase tracking-wider text-zinc-500 mb-1 block">
              Browse
            </label>
            <div className="flex gap-1 bg-zinc-900 border border-zinc-800 rounded p-0.5 mb-2">
              {(
                [
                  { id: "etf", label: "ETFs" },
                  { id: "stocks", label: "Stocks" },
                ] as const
              ).map((o) => (
                <button
                  key={o.id}
                  onClick={() => setBrowseTab(o.id)}
                  className={cn(
                    "flex-1 px-2 py-1 text-[11px] font-medium rounded transition-colors cursor-pointer",
                    browseTab === o.id
                      ? "bg-zinc-800 text-zinc-100"
                      : "text-zinc-500 hover:text-zinc-300",
                  )}
                >
                  {o.label}
                </button>
              ))}
            </div>
            <p className="text-[10px] text-zinc-600 mb-2 leading-snug">
              Browse one pool at a time — your selection list below combines symbols from both.
            </p>
          </div>

          {browseTab === "etf" ? (
            <ETFPicker
              filter={etfFilter}
              onFilterChange={setEtfFilter}
              candidates={etfFiltered}
              loading={rowCountsQuery.isLoading}
              addableThreshold={addableThreshold}
              coverage={coverage}
              isLive={isLive}
              selected={selected}
              onToggle={toggleSym}
            />
          ) : (
            <StockPicker
              sectors={sectors}
              sectorFilter={sectorFilter}
              onToggleSector={toggleSector}
              search={stockSearch}
              onSearchChange={setStockSearch}
              candidates={stockOptions}
              loadingUniverse={assetsQuery.isLoading}
              universeError={assetsQuery.isError}
              addableThreshold={addableThreshold}
              coverage={coverage}
              isLive={isLive}
              selected={selected}
              onAdd={addSym}
            />
          )}

          <SelectedList
            selected={selected}
            etfSymbols={etfSymbolSet}
            weighting={weighting}
            manualWeights={manualWeights}
            coverage={coverage}
            isLive={isLive}
            onChangeWeight={(sym, v) =>
              setManualWeights((prev) => ({ ...prev, [sym]: Math.max(0, v) }))
            }
            onRemove={removeSym}
          />

          {weighting === "optimizer" && (
            <>
              <Field label="Risk-Free Rate (%)">
                <div className="flex gap-1">
                  <input
                    type="number"
                    step={0.1}
                    value={rfRate}
                    onChange={(e) => setRfRate(Number(e.target.value))}
                    className="flex-1 bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5 text-xs text-zinc-200 font-mono tabular-nums"
                  />
                  <button
                    onClick={fetchCetes}
                    className="px-2 py-1.5 text-[10px] font-semibold bg-zinc-800 hover:bg-zinc-700 text-zinc-300 rounded cursor-pointer"
                  >
                    CETES
                  </button>
                </div>
              </Field>
              <Field label="Borrowing Rate (%)">
                <input
                  type="number"
                  step={0.1}
                  value={borrowRate}
                  onChange={(e) => setBorrowRate(Number(e.target.value))}
                  className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5 text-xs text-zinc-200 font-mono tabular-nums"
                />
              </Field>
            </>
          )}

          <Field label="Rebalance">
            <select
              value={rebalFreq}
              onChange={(e) => setRebalFreq(e.target.value)}
              className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5 text-xs text-zinc-200 font-mono"
            >
              {REBALANCE_OPTIONS.map((o) => (
                <option key={o} value={o}>
                  {o}
                </option>
              ))}
            </select>
          </Field>

          <Field label="Benchmark">
            <select
              value={benchmarkType}
              onChange={(e) => setBenchmarkType(e.target.value)}
              className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5 text-xs text-zinc-200 font-mono"
            >
              {BENCHMARK_TYPES.map((b) => (
                <option key={b} value={b}>
                  {b}
                </option>
              ))}
            </select>
          </Field>

          <RunButton
            onClick={handleRun}
            loading={loading}
            disabled={!canRun}
            label={weighting === "optimizer" ? "Optimize" : "Simulate"}
          />

          <EligibilityBanner
            selectedCount={selected.length}
            joint={jointQuery.data}
            isError={jointQuery.isError}
          />

          {(optimizeMut.error || simMut.error || benchMut.error) && (
            <Notice tone="error">
              {(optimizeMut.error as Error)?.message ??
                (simMut.error as Error)?.message ??
                (benchMut.error as Error)?.message}
            </Notice>
          )}
        </LeftSidebar>
      }
      right={
        <RightSidebar>
          {weighting === "optimizer" && opt && <TangencyMini opt={opt} rfRate={rfRate} />}
          {sim && <SimMetricsKPIs metrics={sim.metrics} />}
          {bench && <BenchmarkKPIs benchmark={bench} />}
          {sim && bench && <RelativeKPIs sim={sim} bench={bench} />}
          {!sim && !opt && (
            <div className="text-xs text-zinc-600">
              Configure the run on the left, then{" "}
              {weighting === "optimizer" ? "optimize" : "simulate"}.
            </div>
          )}
        </RightSidebar>
      }
      bottom={
        compositionRows.length > 0 ? (
          <BottomPanel>
            <CompositionTable rows={compositionRows} />
          </BottomPanel>
        ) : undefined
      }
    >
      <ChartArea
        opt={opt}
        sim={sim}
        bench={bench}
        weighting={weighting}
        activeTab={activeTab}
        onTab={setActiveTab}
        loading={loading}
      />
      <WalkForwardPanel selected={selected} startDate={startDate} endDate={endDate} rfRate={rfRate} />
    </AppLayout>
  );
}

/**
 * Self-contained walk-forward validation for the tangency/min-variance
 * optimizer above. `/portfolio/optimize` + `/simulate` fit weights on
 * [startDate, endDate] and then evaluate those same weights over the
 * identical window — in-sample look-ahead (the optimizer has seen the
 * returns it's graded on). This panel calls
 * `POST /portfolio/walk-forward-optimize` instead: it re-fits weights on a
 * trailing lookback window every rebalance_months and only ever reports
 * *realized* returns from after each fit, so the Sharpe shown here is a
 * genuine out-of-sample number — deliberately kept separate from the
 * `weighting` state machine above so it can't destabilize it.
 */
function WalkForwardPanel({
  selected,
  startDate,
  endDate,
  rfRate,
}: {
  selected: string[];
  startDate: string;
  endDate: string;
  rfRate: number;
}) {
  const [lookbackMonths, setLookbackMonths] = useState(24);
  const [rebalanceMonths, setRebalanceMonths] = useState(6);
  const [portfolioKind, setPortfolioKind] = useState<"tangency" | "min_variance">("tangency");
  const [open, setOpen] = useState(false);

  const wfMut = useMutation({
    mutationFn: () =>
      api.walkForwardOptimizePortfolio({
        symbols: selected,
        start_date: startDate,
        end_date: endDate || undefined,
        lookback_months: lookbackMonths,
        rebalance_months: rebalanceMonths,
        risk_free_rate: rfRate / 100,
        portfolio_kind: portfolioKind,
      }),
  });
  const wf = wfMut.data;

  return (
    <div className="mt-4 border-t border-zinc-800 pt-4">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="text-[11px] uppercase tracking-wider text-zinc-400 hover:text-zinc-200 flex items-center gap-1.5"
      >
        <span>{open ? "▾" : "▸"}</span>
        Walk-forward validation (no look-ahead)
      </button>
      {open && (
        <div className="mt-3 flex flex-col gap-3">
          <p className="text-[11px] text-zinc-500 leading-relaxed max-w-2xl">
            Re-fits {portfolioKind === "tangency" ? "tangency" : "min-variance"} weights on a
            trailing lookback window every rebalance period, using the same {selected.length}{" "}
            selected symbols and date range as above, then reports only realized (never-fit-on)
            returns. This is the honest comparison to the optimizer&apos;s Sharpe shown above,
            which is fit and evaluated on the same window.
          </p>
          <div className="flex flex-wrap gap-3 items-end">
            <Field label="Lookback (months)">
              <input
                type="number"
                min={3}
                max={60}
                value={lookbackMonths}
                onChange={(e) => setLookbackMonths(+e.target.value)}
                className="w-24 bg-zinc-900 border border-zinc-800 rounded px-2 py-1 text-xs font-mono text-zinc-200"
              />
            </Field>
            <Field label="Rebalance (months)">
              <input
                type="number"
                min={1}
                max={12}
                value={rebalanceMonths}
                onChange={(e) => setRebalanceMonths(+e.target.value)}
                className="w-24 bg-zinc-900 border border-zinc-800 rounded px-2 py-1 text-xs font-mono text-zinc-200"
              />
            </Field>
            <Field label="Objective">
              <select
                value={portfolioKind}
                onChange={(e) => setPortfolioKind(e.target.value as "tangency" | "min_variance")}
                className="bg-zinc-900 border border-zinc-800 rounded px-2 py-1 text-xs font-mono text-zinc-200"
              >
                <option value="tangency">Tangency (max Sharpe)</option>
                <option value="min_variance">Min variance</option>
              </select>
            </Field>
            <RunButton
              onClick={() => wfMut.mutate()}
              loading={wfMut.isPending}
              label="Run walk-forward"
              disabled={selected.length < 2 || !startDate}
            />
          </div>
          {wfMut.isError && (
            <Notice tone="error">{(wfMut.error as Error).message}</Notice>
          )}
          {wf && (
            <>
              <div className="flex flex-wrap gap-2">
                <KPICard label="OOS Sharpe" value={fmtRatio(wf.metrics.sharpe_ratio)} accent="neutral" />
                <KPICard
                  label="Ann. return"
                  value={fmtPct(wf.metrics.annualized_return)}
                  accent={wf.metrics.annualized_return >= 0 ? "positive" : "negative"}
                />
                <KPICard label="Max DD" value={fmtPct(wf.metrics.max_drawdown)} accent="negative" />
                <KPICard label="Pain ratio" value={fmtRatio(wf.metrics.pain_ratio)} accent="neutral" />
                <KPICard label="Periods" value={String(wf.periods.length)} accent="neutral" />
              </div>
              <Plot
                data={[
                  {
                    x: wf.equity_curve.map((p) => p.date),
                    y: wf.equity_curve.map((p) => p.cumulative_return),
                    type: "scatter",
                    mode: "lines",
                    name: "Walk-forward OOS",
                    line: { color: "#f59e0b", width: 1.5 },
                  },
                ]}
                layout={{
                  ...DARK_PLOTLY_LAYOUT,
                  height: 260,
                  margin: { l: 48, r: 16, t: 16, b: 32 },
                  yaxis: { ...DARK_PLOTLY_LAYOUT.yaxis, tickformat: ".1%" },
                }}
                config={{ displayModeBar: false, responsive: true }}
                className="w-full"
                useResizeHandler
                style={{ width: "100%" }}
              />
            </>
          )}
        </div>
      )}
    </div>
  );
}

/* ──────────────────────────────────────────────────────────────
 * Helpers
 * ────────────────────────────────────────────────────────────── */

function optimizerWeights(opt: OptimizeResponse | undefined): Record<string, number> | undefined {
  return opt?.tangency.weights;
}

/* ──────────────────────────────────────────────────────────────
 * Sub-components (page-local — small, single-purpose)
 * ────────────────────────────────────────────────────────────── */

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

function Notice({ tone, children }: { tone: "error"; children: React.ReactNode }) {
  return (
    <div
      className={cn(
        "text-[10px] rounded px-2 py-1 border",
        tone === "error" && "text-red-400 bg-red-950/50 border-red-900",
      )}
    >
      {children}
    </div>
  );
}

function ModeSwitch<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: T;
  options: { id: T; label: string }[];
  onChange: (v: T) => void;
}) {
  return (
    <div>
      <label className="text-[10px] uppercase tracking-wider text-zinc-500 mb-1 block">
        {label}
      </label>
      <div className="flex gap-1 bg-zinc-900 border border-zinc-800 rounded p-0.5">
        {options.map((o) => (
          <button
            key={o.id}
            onClick={() => onChange(o.id)}
            className={cn(
              "flex-1 px-2 py-1 text-[11px] font-medium rounded transition-colors cursor-pointer",
              value === o.id
                ? "bg-zinc-800 text-zinc-100"
                : "text-zinc-500 hover:text-zinc-300",
            )}
          >
            {o.label}
          </button>
        ))}
      </div>
    </div>
  );
}

interface AssetLite {
  symbol: string;
}

type CoverageLookup = (sym: string) => PortfolioCoverageEntry | undefined;
type LiveLookup = (sym: string) => boolean;

/** Small inline marker: green dot = still trading, red dot = delisted. */
function LiveDot({ live }: { live: boolean }) {
  return (
    <span
      title={live ? "Still trading" : "Delisted (last trade older than panel tail)"}
      className={cn(
        "inline-block w-1.5 h-1.5 rounded-full shrink-0",
        live ? "bg-emerald-500" : "bg-red-500",
      )}
    />
  );
}

/** "2014→" for live, "2016 – 20" for delisted. ISO range goes in the title. */
function CoverageWindow({ cov, live }: { cov: PortfolioCoverageEntry; live: boolean }) {
  const firstYear = cov.first.slice(0, 4);
  const lastYear = cov.last.slice(0, 4);
  const text = live
    ? `${firstYear}→`
    : firstYear === lastYear
      ? firstYear
      : `${firstYear} – ${lastYear.slice(2)}`;
  return (
    <span
      className="text-[9px] tabular-nums text-zinc-600 shrink-0"
      title={`${cov.first} → ${cov.last} · ${cov.count} rows in window`}
    >
      {text}
    </span>
  );
}

function ETFPicker({
  filter,
  onFilterChange,
  candidates,
  loading,
  addableThreshold,
  coverage,
  isLive,
  selected,
  onToggle,
}: {
  filter: string;
  onFilterChange: (v: string) => void;
  candidates: AssetLite[];
  loading: boolean;
  addableThreshold: number;
  coverage: CoverageLookup;
  isLive: LiveLookup;
  selected: string[];
  onToggle: (sym: string) => void;
}) {
  let addable = 0;
  for (const a of candidates) {
    if ((coverage(a.symbol)?.count ?? 0) >= addableThreshold) addable += 1;
  }
  return (
    <Field label="ETF Selection">
      <input
        type="text"
        placeholder="Filter..."
        value={filter}
        onChange={(e) => onFilterChange(e.target.value)}
        className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5 text-xs text-zinc-200 font-mono mb-1"
      />
      <div className="max-h-40 overflow-y-auto border border-zinc-800 rounded">
        {candidates.map((a) => {
          const cov = coverage(a.symbol);
          const enoughRows = (cov?.count ?? 0) >= addableThreshold;
          const live = isLive(a.symbol);
          const checked = selected.includes(a.symbol);
          return (
            <label
              key={a.symbol}
              title={
                cov && !enoughRows
                  ? `Only ${cov.count} rows in window — needs ≥ ${addableThreshold}`
                  : undefined
              }
              className={cn(
                "flex items-center gap-1.5 px-2 py-0.5 text-[11px] font-mono",
                enoughRows
                  ? "text-zinc-300 hover:bg-zinc-800/50 cursor-pointer"
                  : "text-zinc-600 cursor-not-allowed",
              )}
            >
              <input
                type="checkbox"
                checked={checked}
                onChange={() => enoughRows && onToggle(a.symbol)}
                disabled={!enoughRows && !checked}
                className="accent-blue-500"
              />
              {cov && <LiveDot live={live} />}
              <span className="flex-1 truncate">{a.symbol}</span>
              {cov && <CoverageWindow cov={cov} live={live} />}
            </label>
          );
        })}
      </div>
      <div className="text-[10px] text-zinc-600 mt-1">
        {loading ? (
          <span>Loading coverage…</span>
        ) : (
          <span>
            {candidates.length} shown · {addable} addable (≥{addableThreshold} rows) ·{" "}
            {selected.length} selected
          </span>
        )}
      </div>
    </Field>
  );
}

function StockPicker({
  sectors,
  sectorFilter,
  onToggleSector,
  search,
  onSearchChange,
  candidates,
  loadingUniverse,
  universeError,
  addableThreshold,
  coverage,
  isLive,
  selected,
  onAdd,
}: {
  sectors: SectorSummaryResponse | undefined;
  sectorFilter: string[];
  onToggleSector: (s: string) => void;
  search: string;
  onSearchChange: (v: string) => void;
  candidates: { symbol: string; industry: string }[];
  loadingUniverse: boolean;
  universeError: boolean;
  addableThreshold: number;
  coverage: CoverageLookup;
  isLive: LiveLookup;
  selected: string[];
  onAdd: (sym: string) => void;
}) {
  return (
    <>
      <Field label="Filter by Sector">
        <div className="max-h-32 overflow-y-auto space-y-1">
          {sectors?.sectors.map((s) => (
            <label
              key={s.sector}
              className="flex items-center gap-2 text-xs text-zinc-400 cursor-pointer hover:text-zinc-200"
            >
              <input
                type="checkbox"
                checked={sectorFilter.includes(s.sector)}
                onChange={() => onToggleSector(s.sector)}
                className="accent-blue-500"
              />
              <span className="truncate">{s.sector}</span>
              <span className="ml-auto text-zinc-600">{s.count}</span>
            </label>
          ))}
        </div>
      </Field>

      <Field label="Search Stocks">
        <input
          placeholder="Symbol, sector, or industry…"
          value={search}
          onChange={(e) => onSearchChange(e.target.value)}
          className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1 text-xs text-zinc-100 placeholder-zinc-600"
        />
        <p className="text-[10px] text-zinc-600 mt-1 leading-snug">
          Pick list is restricted to symbols present in the loaded price panel.
        </p>
      </Field>

      {universeError && <Notice tone="error">Could not load price universe.</Notice>}

      <Field
        label={loadingUniverse ? "Available (loading…)" : `Available (${candidates.length})`}
      >
        <div className="max-h-52 overflow-y-auto space-y-0.5">
          {candidates.map((s) => {
            const isSelected = selected.includes(s.symbol);
            const cov = coverage(s.symbol);
            const enoughRows = (cov?.count ?? 0) >= addableThreshold;
            const live = isLive(s.symbol);
            const dim = !enoughRows && !isSelected;
            return (
              <button
                key={s.symbol}
                onClick={() => enoughRows && !isSelected && onAdd(s.symbol)}
                disabled={isSelected || !enoughRows}
                title={
                  cov && !enoughRows
                    ? `Only ${cov.count} rows in window — needs ≥ ${addableThreshold}`
                    : undefined
                }
                className={cn(
                  "w-full text-left px-2 py-0.5 rounded text-xs truncate transition-colors flex items-center gap-1.5",
                  isSelected
                    ? "text-zinc-600 cursor-default"
                    : dim
                      ? "text-zinc-600 cursor-not-allowed"
                      : "text-zinc-300 hover:bg-zinc-800 cursor-pointer",
                )}
              >
                {cov && <LiveDot live={live} />}
                <span className="font-mono">{s.symbol}</span>
                <span className="text-zinc-600 flex-1 truncate">{s.industry}</span>
                {cov && <CoverageWindow cov={cov} live={live} />}
              </button>
            );
          })}
          {candidates.length === 0 && !loadingUniverse && (
            <div className="text-[10px] text-zinc-600 px-2 py-1">
              No matches — adjust sector filters or search.
            </div>
          )}
        </div>
      </Field>
    </>
  );
}

function SelectedList({
  selected,
  etfSymbols,
  weighting,
  manualWeights,
  coverage,
  isLive,
  onChangeWeight,
  onRemove,
}: {
  selected: string[];
  etfSymbols: Set<string>;
  weighting: Weighting;
  manualWeights: Record<string, number>;
  coverage: CoverageLookup;
  isLive: LiveLookup;
  onChangeWeight: (sym: string, v: number) => void;
  onRemove: (sym: string) => void;
}) {
  const etfCount = selected.filter((s) => etfSymbols.has(s)).length;
  const stockCount = selected.length - etfCount;
  return (
    <Field
      label={
        selected.length === 0
          ? "Selected (0)"
          : `Selected (${selected.length}) — ${etfCount} ETF · ${stockCount} stock`
      }
    >
      <div className="max-h-32 overflow-y-auto space-y-0.5">
        {selected.length === 0 && (
          <div className="text-[10px] text-zinc-600">Nothing selected yet.</div>
        )}
        {selected.map((sym) => {
          const isEtf = etfSymbols.has(sym);
          const cov = coverage(sym);
          const live = isLive(sym);
          return (
            <div key={sym} className="flex items-center justify-between text-xs gap-1">
              <div className="flex items-center gap-1.5 min-w-0">
                {cov && <LiveDot live={live} />}
                <span className="font-mono text-emerald-400">{sym}</span>
                <span
                  className={cn(
                    "text-[9px] uppercase tracking-wider px-1 rounded",
                    isEtf ? "bg-blue-950/60 text-blue-400" : "bg-zinc-800 text-zinc-500",
                  )}
                >
                  {isEtf ? "ETF" : "STK"}
                </span>
                {cov && <CoverageWindow cov={cov} live={live} />}
              </div>
              {weighting === "manual" && (
                <input
                  type="number"
                  min={0}
                  step={0.1}
                  value={manualWeights[sym] ?? 1}
                  onChange={(e) => onChangeWeight(sym, parseFloat(e.target.value) || 0)}
                  className="w-14 bg-zinc-900 border border-zinc-700 rounded px-1 py-0.5 text-[10px] font-mono text-zinc-300 text-right"
                />
              )}
              <button
                onClick={() => onRemove(sym)}
                className="text-zinc-600 hover:text-red-400 text-xs ml-1"
              >
                ×
              </button>
            </div>
          );
        })}
      </div>
    </Field>
  );
}

function TangencyMini({ opt, rfRate }: { opt: OptimizeResponse; rfRate: number }) {
  const t = opt.tangency;
  const top = Object.entries(t.weights)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 5);
  return (
    <div className="flex flex-col gap-2 mb-2">
      <div className="text-[10px] uppercase tracking-wider text-zinc-500 font-semibold">
        Tangency
      </div>
      <KPICard label="Return" value={fmtPct(t.ret)} accent="positive" />
      <KPICard label="Volatility" value={fmtPct(t.volatility)} />
      <KPICard label="Sharpe" value={fmtRatio(t.sharpe)} accent="positive" />
      <KPICard label="Risk-Free" value={`${rfRate.toFixed(2)}%`} />
      <div className="text-[10px] uppercase tracking-wider text-zinc-500 mt-2">
        Top Weights
      </div>
      {top.map(([sym, w]) => (
        <div key={sym} className="flex justify-between text-[11px] font-mono">
          <span className="text-zinc-400">{sym}</span>
          <span className="text-zinc-200 tabular-nums">{(w * 100).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  );
}

function RelativeKPIs({ sim, bench }: { sim: SimulateResponse; bench: BenchmarkResponse }) {
  return (
    <>
      <div className="mt-3 text-[10px] uppercase tracking-wider text-zinc-500 font-semibold">
        Relative
      </div>
      <KPICard
        label="Excess Return"
        value={fmtPct(sim.metrics.total_return - bench.total_return)}
        accent={sim.metrics.total_return - bench.total_return >= 0 ? "positive" : "negative"}
      />
      {sim.metrics.alpha !== undefined && (
        <KPICard
          label="Alpha"
          value={fmtPct(sim.metrics.alpha)}
          accent={sim.metrics.alpha >= 0 ? "positive" : "negative"}
        />
      )}
      {sim.metrics.beta !== undefined && (
        <KPICard label="Beta" value={sim.metrics.beta.toFixed(2)} accent="neutral" />
      )}
      {sim.metrics.information_ratio !== undefined && (
        <KPICard
          label="Info Ratio"
          value={sim.metrics.information_ratio.toFixed(2)}
          accent={sim.metrics.information_ratio >= 0 ? "positive" : "negative"}
        />
      )}
    </>
  );
}

function CompositionTable({
  rows,
}: {
  rows: {
    symbol: string;
    kind: string;
    sector: string;
    industry: string;
    weight: number;
    first?: string;
    last?: string;
    live: boolean;
  }[];
}) {
  const delisted = rows.filter((r) => !r.live);
  return (
    <>
      <h3 className="text-[10px] uppercase tracking-wider text-zinc-500 mb-2 font-semibold">
        Portfolio Composition
      </h3>
      {delisted.length > 0 && (
        <div className="text-[10px] text-amber-400 bg-amber-950/30 border border-amber-900/60 rounded px-2 py-1 mb-2">
          {delisted.length} holding{delisted.length > 1 ? "s" : ""} delisted before the
          panel tail: {delisted.map((r) => `${r.symbol} (${r.last})`).join(", ")}. NAV
          drops these symbols on their last trade date.
        </div>
      )}
      <table className="w-full text-xs text-left">
        <thead>
          <tr className="text-zinc-500 border-b border-zinc-800">
            <th className="pb-1 font-semibold">Symbol</th>
            <th className="pb-1 font-semibold">Kind</th>
            <th className="pb-1 font-semibold">Sector</th>
            <th className="pb-1 font-semibold">Industry</th>
            <th className="pb-1 font-semibold">Window</th>
            <th className="pb-1 font-semibold text-right">Weight</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.symbol} className="border-b border-zinc-800/50 text-zinc-300">
              <td className="py-1 font-mono text-emerald-400 flex items-center gap-1.5">
                <span
                  className={cn(
                    "inline-block w-1.5 h-1.5 rounded-full shrink-0",
                    row.live ? "bg-emerald-500" : "bg-red-500",
                  )}
                  title={row.live ? "Still trading" : "Delisted"}
                />
                {row.symbol}
              </td>
              <td className="py-1">
                <span
                  className={cn(
                    "text-[9px] uppercase tracking-wider px-1 rounded",
                    row.kind === "ETF"
                      ? "bg-blue-950/60 text-blue-400"
                      : "bg-zinc-800 text-zinc-500",
                  )}
                >
                  {row.kind}
                </span>
              </td>
              <td className="py-1">{row.sector}</td>
              <td className="py-1">{row.industry}</td>
              <td className="py-1 font-mono text-[11px] text-zinc-500 tabular-nums">
                {row.first && row.last
                  ? `${row.first} → ${row.live ? "live" : row.last}`
                  : "—"}
              </td>
              <td className="py-1 text-right font-mono">
                {(row.weight * 100).toFixed(1)}%
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}

/* ── Center chart area ─────────────────────────────────────── */

function ChartArea({
  opt,
  sim,
  bench,
  weighting,
  activeTab,
  onTab,
  loading,
}: {
  opt: OptimizeResponse | undefined;
  sim: SimulateResponse | undefined;
  bench: BenchmarkResponse | undefined;
  weighting: Weighting;
  activeTab: ChartTab;
  onTab: (t: ChartTab) => void;
  loading: boolean;
}) {
  const tabs: { id: ChartTab; label: string; show: boolean }[] = [
    { id: "frontier", label: "Efficient Frontier", show: weighting === "optimizer" && !!opt },
    { id: "nav", label: "NAV", show: !!sim },
    { id: "drawdown", label: "Drawdown", show: !!sim },
  ];

  // Snap to a visible tab if the current one disappeared.
  useEffect(() => {
    const visible = tabs.filter((t) => t.show);
    if (visible.length && !visible.find((t) => t.id === activeTab)) {
      onTab(visible[0].id);
    }
  }, [tabs, activeTab, onTab]);

  if (!opt && !sim) {
    return (
      <div className="flex items-center justify-center h-64 text-zinc-600 text-sm">
        {loading
          ? "Running…"
          : "Pick a universe + weighting, then run."}
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      <div className="flex gap-1 mb-3 border-b border-zinc-800 pb-2">
        {tabs
          .filter((t) => t.show)
          .map((t) => (
            <button
              key={t.id}
              onClick={() => onTab(t.id)}
              className={cn(
                "px-3 py-1 text-xs font-medium rounded-t transition-colors cursor-pointer",
                activeTab === t.id
                  ? "bg-zinc-800 text-zinc-200 border-b-2 border-blue-500"
                  : "text-zinc-500 hover:text-zinc-300",
              )}
            >
              {t.label}
            </button>
          ))}
      </div>

      <div className="flex-1 min-h-0">
        {activeTab === "frontier" && opt && <FrontierChart opt={opt} />}
        {activeTab === "nav" && sim && <NAVChart sim={sim} benchmark={bench} />}
        {activeTab === "drawdown" && sim && <DrawdownChart sim={sim} benchmark={bench} />}
      </div>
    </div>
  );
}

function FrontierChart({ opt }: { opt: OptimizeResponse }) {
  const { frontier, cal, tangency: t, min_vol: mv, individual: ind } = opt;
  return (
    <Plot
      data={[
        {
          type: "scatter",
          x: frontier.map((p) => p.volatility * 100),
          y: frontier.map((p) => p.ret * 100),
          mode: "lines",
          name: "Efficient Frontier",
          line: { color: "#3b82f6", width: 2.5 },
        },
        {
          type: "scatter",
          x: cal.map((p) => p.volatility * 100),
          y: cal.map((p) => p.ret * 100),
          mode: "lines",
          name: "CAL",
          line: { color: "#22c55e", width: 1.5, dash: "dash" },
        },
        {
          type: "scatter",
          x: [t.volatility * 100],
          y: [t.ret * 100],
          mode: "markers",
          name: "Tangency",
          marker: { size: 12, color: "#ef4444", symbol: "star" },
        },
        {
          type: "scatter",
          x: [mv.volatility * 100],
          y: [mv.ret * 100],
          mode: "markers",
          name: "Min Variance",
          marker: { size: 10, color: "#f59e0b", symbol: "diamond" },
        },
        {
          type: "scatter",
          x: ind.map((a) => a.volatility * 100),
          y: ind.map((a) => a.ret * 100),
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          mode: "markers+text" as any,
          name: "Assets",
          marker: { size: 7, color: "#a1a1aa" },
          text: ind.map((a) => a.symbol),
          textposition: "top center",
          textfont: { size: 9 },
        },
      ]}
      layout={{
        ...DARK_PLOTLY_LAYOUT,
        height: 440,
        xaxis: {
          ...DARK_PLOTLY_LAYOUT.xaxis,
          title: { text: "Volatility (%)" } as Partial<Plotly.LayoutAxis["title"]>,
        },
        yaxis: {
          ...DARK_PLOTLY_LAYOUT.yaxis,
          title: { text: "Expected Return (%)" } as Partial<Plotly.LayoutAxis["title"]>,
        },
        showlegend: true,
      }}
      config={{ displayModeBar: false, responsive: true }}
      className="w-full"
    />
  );
}
