import type {
  AllVarResult,
  BacktestRequest,
  BacktestResponse,
  BenchmarkRequest,
  BenchmarkResponse,
  BootstrapResponse,
  Cetes28Response,
  CommodityListResponse,
  CommodityPriceResponse,
  CommodityReturnsResponse,
  CorrelationResponse,
  ExclusionSummaryResponse,
  FactorsResponse,
  FREDCatalogResponse,
  FREDSeriesResponse,
  GridSearchResponse,
  MLStrategyRequest,
  MLStrategyResponse,
  OptimizeRequest,
  OptimizeResponse,
  PriceSeriesResponse,
  RecessionsResponse,
  RegimeResponse,
  ReplayResponse,
  SeasonalityResponse,
  SectorBreakdownResponse,
  SectorSummaryResponse,
  SharpeComparisonResponse,
  SimulateRequest,
  SimulateResponse,
  StockDetailResponse,
  AssetsResponse,
} from "@/lib/types.ts";

const API_BASE = "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const data = (await response.json()) as { detail?: string };
      if (data?.detail) {
        detail = data.detail;
      }
    } catch {
      // keep default detail if response body is not JSON
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

function withQuery(path: string, params: Record<string, string | number | boolean | undefined>): string {
  const qs = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined) {
      qs.set(key, String(value));
    }
  }
  const query = qs.toString();
  return query ? `${path}?${query}` : path;
}

function withArrayQuery(path: string, key: string, values: string[]): string {
  const qs = new URLSearchParams();
  for (const value of values) {
    qs.append(key, value);
  }
  const query = qs.toString();
  return query ? `${path}?${query}` : path;
}

export const api = {
  listAssets: () => request<AssetsResponse>("/data/assets"),
  listFactors: () => request<FactorsResponse>("/data/factors"),
  getPrices: (symbol: string, start?: string, end?: string) =>
    request<PriceSeriesResponse>(withQuery("/data/prices", { symbol, start, end })),

  runBacktest: (payload: BacktestRequest) =>
    request<BacktestResponse>("/run-backtest", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  runMLStrategy: (payload: MLStrategyRequest) =>
    request<MLStrategyResponse>("/run-ml-strategy", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  getReplayFrames: (params: {
    factor?: string;
    rebalance_freq?: string;
    transaction_cost_bps?: number;
    top_pct?: number;
    tail?: number;
    start_date?: string;
    end_date?: string;
    survivorship_free?: boolean;
  }) => request<ReplayResponse>(withQuery("/replay/frames", params)),
  getVar: (symbol: string, confidence = 95, start?: string, end?: string) =>
    request<AllVarResult>(withQuery("/metrics/var", { symbol, confidence, start, end })),

  getMomentumGrid: (symbol: string, sortino_window = 252) =>
    request<GridSearchResponse>(withQuery("/momentum/grid-search", { symbol, sortino_window })),
  getMomentumBootstrap: (symbol: string, x: number, k: number, sortino_window = 252) =>
    request<BootstrapResponse>(
      withQuery("/momentum/bootstrap", { symbol, x, k, sortino_window }),
    ),
  getMomentumRegime: (symbol: string, x: number, k: number, sortino_window = 252) =>
    request<RegimeResponse>(withQuery("/momentum/regime", { symbol, x, k, sortino_window })),

  getSectorSummary: () => request<SectorSummaryResponse>("/sectors/summary"),
  getSectorBreakdown: (sector?: string) =>
    request<SectorBreakdownResponse>(withQuery("/sectors/breakdown", { sector })),

  optimizePortfolio: (payload: OptimizeRequest) =>
    request<OptimizeResponse>("/portfolio/optimize", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  simulatePortfolio: (payload: SimulateRequest) =>
    request<SimulateResponse>("/portfolio/simulate", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  getBenchmarkReturns: (params: BenchmarkRequest) =>
    request<BenchmarkResponse>(withQuery("/benchmarks/returns", params)),

  getCetes28: () => request<Cetes28Response>("/banxico/cetes28"),

  listCommodities: () => request<CommodityListResponse>("/commodities/list"),
  getCommodityPrices: (symbols: string[], start?: string, end?: string) => {
    const base = withArrayQuery("/commodities/prices", "symbols", symbols);
    const suffix = new URLSearchParams();
    if (start) suffix.set("start", start);
    if (end) suffix.set("end", end);
    const extra = suffix.toString();
    return request<CommodityPriceResponse>(extra ? `${base}&${extra}` : base);
  },
  getCommodityReturns: (symbols: string[], start?: string, end?: string) => {
    const base = withArrayQuery("/commodities/returns", "symbols", symbols);
    const suffix = new URLSearchParams();
    if (start) suffix.set("start", start);
    if (end) suffix.set("end", end);
    const extra = suffix.toString();
    return request<CommodityReturnsResponse>(extra ? `${base}&${extra}` : base);
  },
  getCommodityCorrelation: (symbols: string[], start?: string, end?: string) => {
    const base = withArrayQuery("/commodities/correlation", "symbols", symbols);
    const suffix = new URLSearchParams();
    if (start) suffix.set("start", start);
    if (end) suffix.set("end", end);
    const extra = suffix.toString();
    return request<CorrelationResponse>(extra ? `${base}&${extra}` : base);
  },
  getCommoditySeasonality: (symbol: string, start?: string, end?: string) =>
    request<SeasonalityResponse>(withQuery("/commodities/seasonality", { symbol, start, end })),

  getFREDCatalog: () => request<FREDCatalogResponse>("/fred/catalog"),
  getFREDSeries: (ids: string[], start?: string, end?: string, yoy = false) => {
    const base = withArrayQuery("/fred/series", "ids", ids);
    const suffix = new URLSearchParams();
    if (start) suffix.set("start", start);
    if (end) suffix.set("end", end);
    suffix.set("yoy", String(yoy));
    const extra = suffix.toString();
    return request<FREDSeriesResponse>(extra ? `${base}&${extra}` : base);
  },
  getRecessions: (start?: string, end?: string) =>
    request<RecessionsResponse>(withQuery("/fred/recessions", { start, end })),

  getExclusionSummary: (price_threshold: number, start_date?: string, end_date?: string) =>
    request<ExclusionSummaryResponse>(
      withQuery("/exclusions/summary", { price_threshold, start_date, end_date }),
    ),
  getExclusionDetail: (symbol: string, price_threshold: number, start_date?: string, end_date?: string) =>
    request<StockDetailResponse>(
      withQuery(`/exclusions/detail/${encodeURIComponent(symbol)}`, {
        price_threshold,
        start_date,
        end_date,
      }),
    ),

  sharpeComparison: (target_sharpe: number, n_days: number, seed: number) =>
    request<SharpeComparisonResponse>(
      withQuery("/simulation/sharpe-comparison", { target_sharpe, n_days, seed }),
      { method: "POST" },
    ),
};
