/* ── Pydantic-mirrored types ──────────────────────────────────── */

export interface PerformanceMetrics {
  total_return: number;
  annualized_return: number;
  annualized_volatility: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  max_drawdown: number;
  calmar_ratio: number;
  n_periods: number;
  information_ratio?: number;
  beta?: number;
  alpha?: number;
}

/** `cumulative_return` is decimal return since series start (wealth − 1), not a wealth index. */
export interface EquityCurvePoint {
  date: string;
  cumulative_return: number;
}

export interface BacktestRequest {
  factor_col?: string;
  rebalance_freq?: string;
  transaction_cost_bps?: number;
  top_pct?: number;
  bottom_pct?: number;
  long_only?: boolean;
  start_date?: string;
  end_date?: string;
}

export interface BacktestResponse {
  metrics: PerformanceMetrics;
  equity_curve: EquityCurvePoint[];
  total_days: number;
}

export interface MLStrategyRequest {
  symbol: string;
  model_type: string;
  initial_train_days: number;
  test_days: number;
  max_splits: number;
}

export interface FoldResult {
  fold: number;
  train_size: number;
  test_size: number;
  accuracy: number;
  train_start: string;
  train_end: string;
  test_start: string;
  test_end: string;
}

export interface ConfusionMatrixResult {
  true_negatives: number;
  false_positives: number;
  false_negatives: number;
  true_positives: number;
}

export interface WalkForwardResult {
  model_type: string;
  n_splits: number;
  overall_accuracy: number;
  overall_precision?: number;
  overall_recall?: number;
  overall_f1?: number;
  overall_roc_auc?: number;
  confusion_matrix?: ConfusionMatrixResult;
  folds: FoldResult[];
}

export interface FeatureImportanceItem {
  feature: string;
  importance: number;
}

export interface MLStrategyResponse {
  walkforward: WalkForwardResult;
  feature_importance: FeatureImportanceItem[] | null;
  metadata: {
    symbol: string;
    total_features?: number;
    final_rows?: number;
  };
}

export interface ReplayFrame {
  date: string;
  position: string;
  signal: number | null;
  pnl_today: number;
  cumulative_pnl: number;
  drawdown: number;
  /** Annualised Sortino on trailing window from replay precompute. */
  rolling_sortino: number | null;
}

export interface FactorsResponse {
  count: number;
  factors: string[];
}

/* ── Momentum / Sortino ──────────────────────────────────────── */

export interface GridSearchRow {
  "X (lookback)": number;
  "K (forecast)": number;
  "Z (hit_rate)": number;
  CI_lower: number;
  CI_upper: number;
  Total_signals: number;
  Successful: number;
  Failed: number;
}

export interface GridSearchResponse {
  symbol: string;
  results: GridSearchRow[];
}

export interface BootstrapResponse {
  symbol: string;
  x: number;
  k: number;
  actual_hit_rate: number;
  random_mean: number;
  random_std: number;
  p_value: number;
  significant: boolean;
  n_signals: number;
  bootstrap_dist: number[];
}

export interface RegimeResponse {
  symbol: string;
  regime: {
    current_sortino: number;
    recent_slope: number;
    baseline_slope: number;
    strong_momentum: boolean;
    slope_ratio: number;
  } | null;
}

/* ── Assets / Prices ─────────────────────────────────────────── */

export interface Asset {
  symbol: string;
  type: string;
}

export interface AssetsResponse {
  count: number;
  assets: Asset[];
}

export interface PricePoint {
  date: string;
  price: number;
}

export interface PricesResponse {
  symbol: string;
  count: number;
  data: PricePoint[];
}

/* ── VaR ─────────────────────────────────────────────────────── */

export interface VarResult {
  var: number;
  cvar: number;
}

export interface AllVarResult {
  historical: VarResult;
  parametric: VarResult;
  monte_carlo: VarResult;
  confidence: number;
}

/* ── ETF Optimizer ───────────────────────────────────────────── */

export interface PortfolioPoint {
  volatility: number;
  ret: number;
}

export interface PortfolioWeights {
  [symbol: string]: number;
}

export interface PortfolioPriceRowCountsResponse {
  start_date: string | null;
  end_date: string | null;
  min_required: number;
  counts: Record<string, number>;
}

export interface PortfolioJointHistoryResponse {
  joint_rows: number;
  min_required: number;
  eligible: boolean;
  solo_row_counts: Record<string, number>;
}

export interface OptimizeRequest {
  symbols: string[];
  start_date?: string;
  end_date?: string;
  risk_free_rate: number;
  borrowing_rate: number;
}

export interface OptimizeResponse {
  tangency: {
    weights: PortfolioWeights;
    ret: number;
    volatility: number;
    sharpe: number;
  };
  min_vol: {
    weights: PortfolioWeights;
    ret: number;
    volatility: number;
    sharpe: number;
  };
  frontier: PortfolioPoint[];
  cal: PortfolioPoint[];
  individual: { symbol: string; ret: number; volatility: number }[];
}

export interface SimulateRequest {
  symbols: string[];
  weights: PortfolioWeights;
  freq: string;
  start_date?: string;
  end_date?: string;
}

export interface SimulateResponse {
  nav: { date: string; value: number }[];
  metrics: PerformanceMetrics;
}

/* ── Commodities ─────────────────────────────────────────────── */

export interface CommodityConfig {
  symbol: string;
  name: string;
  category: string;
  unit: string;
}

export interface CommoditiesListResponse {
  commodities: CommodityConfig[];
}

export interface CommodityPricePoint {
  date: string;
  [symbol: string]: string | number;
}

export interface CommodityReturnStats {
  symbol: string;
  mean: number;
  annualized: number;
  volatility: number;
  /** Annualised Sortino (downside deviation), same definition as core metrics. */
  sortino: number;
  skew: number;
  kurtosis: number;
  latest_price: number;
}

export interface CommodityReturnsResponse {
  stats: CommodityReturnStats[];
  series: { date: string; [symbol: string]: string | number }[];
}

export interface CorrelationResponse {
  symbols: string[];
  matrix: number[][];
}

export interface SeasonalityResponse {
  symbol: string;
  monthly_avg: { month: number; avg_return: number }[];
  heatmap: { year: number; month: number; ret: number }[];
}

/* ── FRED / Economic Indicators ──────────────────────────────── */

export interface FREDIndicator {
  id: string;
  name: string;
  unit: string;
  frequency: string;
}

export interface FREDCategory {
  category: string;
  indicators: FREDIndicator[];
}

export interface FREDCatalogResponse {
  categories: FREDCategory[];
}

export interface FREDSeriesPoint {
  date: string;
  [seriesId: string]: string | number;
}

export interface FREDSeriesResponse {
  series: FREDSeriesPoint[];
  metadata: Record<string, { name: string; unit: string }>;
}

export interface RecessionPeriod {
  start: string;
  end: string;
}

/* ── Simulation (Sharpe Limitations) ─────────────────────────── */

export interface InvestmentMetrics {
  sharpe: number;
  sortino: number;
  max_drawdown: number;
  calmar: number;
  total_return: number;
  annualized_vol: number;
  skewness: number;
  kurtosis: number;
  win_rate: number;
  best_day: number;
  worst_day: number;
}

export interface SimulatedInvestment {
  name: string;
  color: string;
  prices: number[];
  daily_returns: number[];
  metrics: InvestmentMetrics;
}

export interface MultiRatioComparisonResponse {
  /** Same numeric target applied to Sharpe, Sortino, and Calmar calibrations. */
  target: number;
  n_days: number;
  seed: number;
  by_sharpe: SimulatedInvestment[];
  by_sortino: SimulatedInvestment[];
  by_calmar: SimulatedInvestment[];
}

/* ── Exclusions ─────────────────────────────────────────────── */

export interface ExclusionStat {
  symbol: string;
  min_price: number;
  max_price: number;
  current_price: number;
  days_below: number;
  pct_below: number;
}

export interface ExclusionSummaryResponse {
  total: number;
  valid: number;
  excluded: number;
  threshold: number;
  stats: ExclusionStat[];
}

export interface StockDetailResponse {
  symbol: string;
  prices: { date: string; price: number; below: boolean }[];
  min_price: number;
  max_price: number;
  current_price: number;
  days_below: number;
  pct_below: number;
  annualized_vol: number;
  max_daily_gain: number;
  max_daily_loss: number;
  extreme_gains: number;
  extreme_losses: number;
}

/* ── Benchmarks ──────────────────────────────────────────────── */

export interface BenchmarkResponse {
  benchmark_name: string;
  dates: string[];
  returns: number[];
  cumulative_returns: number[];
  total_return: number;
  annualized_return: number;
  volatility: number;
  sharpe_ratio: number;
  max_drawdown: number;
  calmar_ratio: number;
}

/* ── Sectors ─────────────────────────────────────────────────── */

export interface SectorSummaryRow {
  sector: string;
  count: number;
  pct: number;
}

export interface SectorSummaryResponse {
  total_symbols: number;
  sectors: SectorSummaryRow[];
}

export interface SectorSymbol {
  symbol: string;
  sector: string;
  industry: string;
  type: string;
}

export interface SectorBreakdownResponse {
  symbols: SectorSymbol[];
}

/* ── Portfolio coverage / eligibility ───────────────────────── */

/** Per-symbol coverage entry from ``GET /portfolio/price-row-counts``. */
export interface PortfolioCoverageEntry {
  /** Number of non-null prices inside the ``[start, end]`` window. */
  count: number;
  /** ISO date of the symbol's first ever trade in the panel. */
  first: string;
  /** ISO date of the symbol's last ever trade in the panel. */
  last: string;
}

export interface PortfolioPriceRowCountsResponse {
  start_date: string | null;
  end_date: string | null;
  /** Optimizer's joint-history requirement (e.g. 60). */
  min_required: number;
  /** ISO date of the most recent row in the panel — used as the "is delisted?" reference. */
  last_panel_date: string | null;
  symbols: Record<string, PortfolioCoverageEntry>;
}

export interface PortfolioJointHistoryResponse {
  joint_rows: number;
  min_required: number;
  eligible: boolean;
  solo_row_counts: Record<string, number>;
}

/* ── Data coverage / quality ────────────────────────────────── */

export interface DatasetInfo {
  name: string;
  source: string;
  path: string;
  layer: "raw" | "derived";
  rows: number;
  columns: number;
  first_date: string | null;
  last_date: string | null;
  size_mb: number;
  description: string;
}

export interface YearCoverage {
  year: number;
  symbols_with_data: number;
  sp500_members: number;
  coverage_pct: number;
}

export interface QuarantineEntry {
  symbol: string;
  check: string;
  value: number;
  detail: string;
  status: "quarantined" | "flagged" | "cleared";
  review_note: string;
}

export interface SP500CsvInfo {
  filename: string;
  path: string;
  file_mtime_utc: string;
  age_days: number;
  last_membership_date: string | null;
  n_snapshots: number | null;
}

export interface DataCoverageResponse {
  datasets: DatasetInfo[];
  coverage_by_year: YearCoverage[];
  quarantine: QuarantineEntry[];
  quarantined_symbol_count: number;
  flagged_symbol_count: number;
  total_symbols_loaded: number;
  survivorship_note: string;
  sp500_csv: SP500CsvInfo | null;
}

/* ── Fama-French factors ────────────────────────────────────── */

export interface FactorStats {
  factor: string;
  annualized_return: number;
  annualized_volatility: number;
  sharpe_ratio: number;
}

export interface FF5SeriesResponse {
  dates: string[];
  growth: Record<string, number[]>;
  stats: FactorStats[];
  first_date: string;
  last_date: string;
}
