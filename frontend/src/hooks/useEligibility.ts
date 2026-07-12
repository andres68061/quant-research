/**
 * Shared eligibility hook for portfolio pages.
 *
 * Combines three concerns that the ETF optimizer and the custom portfolio
 * builder both need:
 *   - the loaded price universe (``GET /data/assets``),
 *   - per-symbol row counts (``GET /portfolio/price-row-counts``) so the
 *     pick-list can be filtered to symbols with enough solo history, and
 *   - joint-history validation for the current selection
 *     (``POST /portfolio/joint-history``) so we block runs that would 400.
 *
 * Returns the raw query objects plus convenience flags (`canRun`, `jointReady`)
 * so callers don't reimplement the same boolean dance.
 */
import { useEffect, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";

import { api } from "@/lib/api.ts";
import type {
  AssetsResponse,
  PortfolioCoverageEntry,
  PortfolioJointHistoryResponse,
  PortfolioPriceRowCountsResponse,
} from "@/lib/types.ts";

export interface UseEligibilityArgs {
  startDate: string | undefined;
  selected: string[];
  /** Auto-prune ineligible symbols from selection. Pass setter to opt in. */
  setSelected?: (updater: (prev: string[]) => string[]) => void;
  /** Minimum joint rows the run requires (defaults to API-reported value). */
  minRequired?: number;
}

export interface UseEligibilityResult {
  assetsQuery: ReturnType<typeof useQuery<AssetsResponse>>;
  rowCountsQuery: ReturnType<typeof useQuery<PortfolioPriceRowCountsResponse>>;
  jointQuery: ReturnType<typeof useQuery<PortfolioJointHistoryResponse>>;
  /** Set of symbols actually present in the loaded price panel. */
  priceSymbols: Set<string>;
  /** Reported minimum (from row-counts response) or fallback. */
  minReq: number;
  /** Last trading day represented in the price panel (used to flag delisted symbols). */
  lastPanelDate: string | undefined;
  /** Look up a symbol's coverage record (count + first/last trade dates). */
  coverage: (symbol: string) => PortfolioCoverageEntry | undefined;
  /** True once joint history is loaded for the current selection. */
  jointReady: boolean;
  /** True iff selection has enough joint rows (or selection too small to check). */
  jointOk: boolean;
  /** Convenience: at least 2 selected, joint loaded, and eligible. */
  canRun: boolean;
}

const FALLBACK_MIN_REQ = 60;

export function useEligibility({
  startDate,
  selected,
  setSelected,
  minRequired,
}: UseEligibilityArgs): UseEligibilityResult {
  const assetsQuery = useQuery({ queryKey: ["assets"], queryFn: api.listAssets });

  const rowCountsQuery = useQuery({
    queryKey: ["portfolio-price-row-counts", startDate],
    queryFn: () => api.getPortfolioPriceRowCounts(startDate),
  });

  const priceSymbols = useMemo(
    () => new Set((assetsQuery.data?.assets ?? []).map((a) => a.symbol)),
    [assetsQuery.data],
  );

  const rowCounts = rowCountsQuery.data;
  const minReq = minRequired ?? rowCounts?.min_required ?? FALLBACK_MIN_REQ;
  const lastPanelDate = rowCounts?.last_panel_date ?? undefined;

  const coverage = useMemo(
    () => (symbol: string) => rowCounts?.symbols[symbol],
    [rowCounts],
  );

  // Auto-prune only the *clearly broken* picks: symbols with zero observations
  // in the window. The optimizer's higher threshold is enforced by the joint-
  // history endpoint when the user clicks Run, and surfaced visually in the
  // picker via dimming — pruning aggressively here would yank out symbols the
  // user picked deliberately for an equal-weight or manual run.
  useEffect(() => {
    if (!setSelected || !rowCounts) return;
    setSelected((prev) =>
      prev.filter((sym) => (rowCounts.symbols[sym]?.count ?? 0) > 0),
    );
  }, [rowCounts, setSelected]);

  // Also prune any selection that vanished from the price panel entirely
  // (e.g. ticker removed from prices.parquet between loads).
  useEffect(() => {
    if (!setSelected || !assetsQuery.isSuccess || priceSymbols.size === 0) return;
    setSelected((prev) => prev.filter((s) => priceSymbols.has(s)));
  }, [assetsQuery.isSuccess, priceSymbols, setSelected]);

  const jointKey = useMemo(() => [...selected].sort().join(","), [selected]);

  const jointQuery = useQuery({
    queryKey: ["portfolio-joint", startDate, jointKey],
    queryFn: () =>
      api.postPortfolioJointHistory({ symbols: selected, start_date: startDate }),
    enabled: selected.length >= 2,
  });

  const jointReady =
    selected.length < 2 || (!jointQuery.isFetching && jointQuery.data != null);
  const jointOk = selected.length < 2 || jointQuery.data?.eligible === true;
  const canRun = selected.length >= 2 && jointReady && jointOk;

  return {
    assetsQuery,
    rowCountsQuery,
    jointQuery,
    priceSymbols,
    minReq,
    lastPanelDate,
    coverage,
    jointReady,
    jointOk,
    canRun,
  };
}
