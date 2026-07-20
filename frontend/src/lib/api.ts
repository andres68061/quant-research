import type {
  AllVarResult,
  AssetsResponse,
  BacktestRequest,
  BacktestResponse,
  BenchmarkResponse,
  BootstrapResponse,
  CommoditiesListResponse,
  CommodityReturnsResponse,
  CorrelationResponse,
  DataCoverageResponse,
  ExclusionSummaryResponse,
  FactorsResponse,
  FF5SeriesResponse,
  FREDCatalogResponse,
  FREDSeriesResponse,
  GridSearchResponse,
  MeasuresLabRequest,
  MeasuresLabResponse,
  MLStrategyRequest,
  MLStrategyResponse,
  OptimizeRequest,
  OptimizeResponse,
  PairsBacktestRequest,
  PairsBacktestResponse,
  PairsIndexBacktestRequest,
  PairsIndexBacktestResponse,
  PairsPersistentBacktestRequest,
  PairsPersistentBacktestResponse,
  PairsScreenRequest,
  PairsScreenResponse,
  PortfolioJointHistoryResponse,
  PortfolioPriceRowCountsResponse,
  PricesResponse,
  RecessionPeriod,
  RegimeResponse,
  ReplayFramesResponse,
  SeasonalityResponse,
  SectorBreakdownResponse,
  SectorSummaryResponse,
  MultiRatioComparisonResponse,
  SimulateRequest,
  SimulateResponse,
  StockDetailResponse,
  WalkForwardOptimizeRequest,
  WalkForwardOptimizeResponse,
} from "./types.ts";

const BASE = "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`API ${res.status}: ${body}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => request<{ status: string }>("/health"),

  listFactors: () => request<FactorsResponse>("/data/factors"),

  getDataCoverage: () => request<DataCoverageResponse>("/data-coverage"),

  reviewQuarantine: (params: {
    symbol: string;
    check: string;
    status: "cleared" | "quarantined" | "flagged";
    note: string;
  }) =>
    request<{ symbol: string; check: string; status: string; review_note: string }>(
      "/data-coverage/quarantine/review",
      {
        method: "POST",
        body: JSON.stringify(params),
      },
    ),

  getFF5Series: (start?: string) =>
    request<FF5SeriesResponse>(`/fama-french/series${start ? `?start=${start}` : ""}`),

  runBacktest: (params: BacktestRequest) =>
    request<BacktestResponse>("/run-backtest", {
      method: "POST",
      body: JSON.stringify(params),
    }),

  runPairsBacktest: (params: PairsBacktestRequest) =>
    request<PairsBacktestResponse>("/run-pairs-backtest", {
      method: "POST",
      body: JSON.stringify(params),
    }),

  screenPairs: (params: PairsScreenRequest) =>
    request<PairsScreenResponse>("/screen-pairs", {
      method: "POST",
      body: JSON.stringify(params),
    }),

  runPairsIndexBacktest: (params: PairsIndexBacktestRequest) =>
    request<PairsIndexBacktestResponse>("/run-pairs-index-backtest", {
      method: "POST",
      body: JSON.stringify(params),
    }),

  runPairsPersistentBacktest: (params: PairsPersistentBacktestRequest) =>
    request<PairsPersistentBacktestResponse>("/run-pairs-persistent-backtest", {
      method: "POST",
      body: JSON.stringify(params),
    }),

  runMLStrategy: (params: MLStrategyRequest) =>
    request<MLStrategyResponse>("/run-ml-strategy", {
      method: "POST",
      body: JSON.stringify(params),
    }),

  getReplayFrames: (params: {
    factor: string;
    rebalance_freq?: string;
    transaction_cost_bps?: number;
    top_pct?: number;
    bottom_pct?: number;
    long_only?: boolean;
    start_date?: string;
    end_date?: string;
    survivorship_free?: boolean;
    min_stocks?: number;
    signal_lag_days?: number;
    tail?: number;
  }) => {
    const sp = new URLSearchParams();
    sp.set("factor", params.factor);
    if (params.rebalance_freq) sp.set("rebalance_freq", params.rebalance_freq);
    if (params.transaction_cost_bps != null) {
      sp.set("transaction_cost_bps", String(params.transaction_cost_bps));
    }
    if (params.top_pct != null) sp.set("top_pct", String(params.top_pct));
    if (params.bottom_pct != null) sp.set("bottom_pct", String(params.bottom_pct));
    if (params.long_only != null) sp.set("long_only", String(params.long_only));
    if (params.start_date) sp.set("start_date", params.start_date);
    if (params.end_date) sp.set("end_date", params.end_date);
    if (params.survivorship_free != null) {
      sp.set("survivorship_free", String(params.survivorship_free));
    }
    if (params.min_stocks != null) sp.set("min_stocks", String(params.min_stocks));
    if (params.signal_lag_days != null) {
      sp.set("signal_lag_days", String(params.signal_lag_days));
    }
    if (params.tail) sp.set("tail", String(params.tail));
    return request<ReplayFramesResponse>(`/replay/frames?${sp}`);
  },

  getMomentumGrid: (symbol: string) =>
    request<GridSearchResponse>(`/momentum/grid-search?symbol=${encodeURIComponent(symbol)}`),

  getMomentumBootstrap: (symbol: string, x: number, k: number) =>
    request<BootstrapResponse>(
      `/momentum/bootstrap?symbol=${encodeURIComponent(symbol)}&x=${x}&k=${k}`,
    ),

  getMomentumRegime: (symbol: string, x: number, k: number) =>
    request<RegimeResponse>(
      `/momentum/regime?symbol=${encodeURIComponent(symbol)}&x=${x}&k=${k}`,
    ),

  /* ── Assets / Prices / VaR ─────────────────────────────────── */

  listAssets: () => request<AssetsResponse>("/data/assets"),

  getPrices: (symbol: string, start?: string, end?: string) => {
    const sp = new URLSearchParams({ symbol });
    if (start) sp.set("start", start);
    if (end) sp.set("end", end);
    return request<PricesResponse>(`/data/prices?${sp}`);
  },

  getVar: (symbol: string, confidence?: number) => {
    const sp = new URLSearchParams({ symbol });
    if (confidence) sp.set("confidence", String(confidence));
    return request<AllVarResult>(`/metrics/var?${sp}`);
  },

  /* ── ETF Optimizer ─────────────────────────────────────────── */

  getPortfolioPriceRowCounts: (startDate?: string, endDate?: string) => {
    const sp = new URLSearchParams();
    if (startDate) sp.set("start_date", startDate);
    if (endDate) sp.set("end_date", endDate);
    const q = sp.toString();
    return request<PortfolioPriceRowCountsResponse>(
      `/portfolio/price-row-counts${q ? `?${q}` : ""}`,
    );
  },

  postPortfolioJointHistory: (params: {
    symbols: string[];
    start_date?: string;
    end_date?: string;
  }) =>
    request<PortfolioJointHistoryResponse>("/portfolio/joint-history", {
      method: "POST",
      body: JSON.stringify(params),
    }),

  optimizePortfolio: (params: OptimizeRequest) =>
    request<OptimizeResponse>("/portfolio/optimize", {
      method: "POST",
      body: JSON.stringify(params),
    }),

  simulatePortfolio: (params: SimulateRequest) =>
    request<SimulateResponse>("/portfolio/simulate", {
      method: "POST",
      body: JSON.stringify(params),
    }),

  walkForwardOptimizePortfolio: (params: WalkForwardOptimizeRequest) =>
    request<WalkForwardOptimizeResponse>("/portfolio/walk-forward-optimize", {
      method: "POST",
      body: JSON.stringify(params),
    }),

  getCetes28: () => request<{ rate: number; date: string }>("/banxico/cetes28"),

  /* ── Commodities ───────────────────────────────────────────── */

  listCommodities: () => request<CommoditiesListResponse>("/commodities/list"),

  getCommodityPrices: (symbols: string[], start?: string, end?: string) => {
    const sp = new URLSearchParams();
    symbols.forEach((s) => sp.append("symbols", s));
    if (start) sp.set("start", start);
    if (end) sp.set("end", end);
    return request<{ series: Record<string, { date: string; price: number }[]> }>(
      `/commodities/prices?${sp}`,
    );
  },

  getCommodityReturns: (symbols: string[], start?: string, end?: string) => {
    const sp = new URLSearchParams();
    symbols.forEach((s) => sp.append("symbols", s));
    if (start) sp.set("start", start);
    if (end) sp.set("end", end);
    return request<CommodityReturnsResponse>(`/commodities/returns?${sp}`);
  },

  getCommodityCorrelation: (symbols: string[], start?: string, end?: string) => {
    const sp = new URLSearchParams();
    symbols.forEach((s) => sp.append("symbols", s));
    if (start) sp.set("start", start);
    if (end) sp.set("end", end);
    return request<CorrelationResponse>(`/commodities/correlation?${sp}`);
  },

  getCommoditySeasonality: (symbol: string) =>
    request<SeasonalityResponse>(
      `/commodities/seasonality?symbol=${encodeURIComponent(symbol)}`,
    ),

  /* ── FRED / Economic Indicators ────────────────────────────── */

  getFREDCatalog: () => request<FREDCatalogResponse>("/fred/catalog"),

  getFREDSeries: (ids: string[], start?: string, end?: string, yoy?: boolean) => {
    const sp = new URLSearchParams();
    ids.forEach((id) => sp.append("ids", id));
    if (start) sp.set("start", start);
    if (end) sp.set("end", end);
    if (yoy) sp.set("yoy", "true");
    return request<FREDSeriesResponse>(`/fred/series?${sp}`);
  },

  getRecessions: (start?: string, end?: string) => {
    const sp = new URLSearchParams();
    if (start) sp.set("start", start);
    if (end) sp.set("end", end);
    return request<{ periods: RecessionPeriod[] }>(`/fred/recessions?${sp}`);
  },

  /* ── Sectors ───────────────────────────────────────────────── */

  getSectorSummary: () => request<SectorSummaryResponse>("/sectors/summary"),

  getSectorBreakdown: (sector?: string) => {
    const sp = sector ? `?sector=${encodeURIComponent(sector)}` : "";
    return request<SectorBreakdownResponse>(`/sectors/breakdown${sp}`);
  },

  /* ── Simulation (Sharpe Limitations) ──────────────────────── */

  sharpeComparison: (targetRatio: number, nDays: number, seed: number) =>
    request<MultiRatioComparisonResponse>(
      `/simulation/sharpe-comparison?target_sharpe=${targetRatio}&n_days=${nDays}&seed=${seed}`,
      { method: "POST" },
    ),

  /* ── Measures Lab ──────────────────────────────────────────── */

  measuresLab: (params: MeasuresLabRequest) =>
    request<MeasuresLabResponse>("/measures-lab", {
      method: "POST",
      body: JSON.stringify(params),
    }),

  /* ── Benchmarks ─────────────────────────────────────────── */

  getBenchmarkReturns: (params: {
    benchmark_type: string;
    start_date: string;
    end_date: string;
    sp500_weighting?: string;
    component1?: string;
    component2?: string;
    weight1?: number;
  }) => {
    const sp = new URLSearchParams({
      benchmark_type: params.benchmark_type,
      start_date: params.start_date,
      end_date: params.end_date,
    });
    if (params.sp500_weighting) sp.set("sp500_weighting", params.sp500_weighting);
    if (params.component1) sp.set("component1", params.component1);
    if (params.component2) sp.set("component2", params.component2);
    if (params.weight1 !== undefined) sp.set("weight1", String(params.weight1));
    return request<BenchmarkResponse>(`/benchmarks/returns?${sp}`);
  },

  /* ── Exclusions ──────────────────────────────────────────── */

  getExclusionSummary: (threshold: number, start?: string, end?: string) => {
    const sp = new URLSearchParams({ price_threshold: String(threshold) });
    if (start) sp.set("start_date", start);
    if (end) sp.set("end_date", end);
    return request<ExclusionSummaryResponse>(`/exclusions/summary?${sp}`);
  },

  getExclusionDetail: (symbol: string, threshold: number, start?: string, end?: string) => {
    const sp = new URLSearchParams({ price_threshold: String(threshold) });
    if (start) sp.set("start_date", start);
    if (end) sp.set("end_date", end);
    return request<StockDetailResponse>(`/exclusions/detail/${encodeURIComponent(symbol)}?${sp}`);
  },
} as const;
