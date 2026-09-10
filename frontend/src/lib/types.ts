/* ── Pydantic-mirrored types ──────────────────────────────────── */

export interface PerformanceMetrics {
  total_return: number;
  annualized_return: number;
  annualized_volatility: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  max_drawdown: number;
  calmar_ratio: number;
  pain_index: number;
  pain_ratio: number;
  ulcer_index: number;
  martin_ratio: number;
  cid1_ratio: number;
  typical_period_return: number;
  cid2_ratio: number;
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
  survivorship_free?: boolean;
  min_stocks?: number;
  signal_lag_days?: number;
}

export interface InvestedCoverage {
  pct_days_invested: number;
  n_days: number;
  n_days_invested: number;
  n_days_flat: number;
  longest_flat_streak_days: number;
  min_stocks: number;
  cash_earns_zero: boolean;
  warning: string | null;
}

export interface RollingDiagPoint {
  date: string;
  sharpe: number | null;
  sortino: number | null;
  volatility: number | null;
}

export interface DrawdownPoint {
  date: string;
  drawdown: number;
}

export interface ReturnHistogram {
  bin_edges: number[];
  counts: number[];
}

export interface BacktestDiagnostics {
  rolling: RollingDiagPoint[];
  drawdown: DrawdownPoint[];
  histogram: ReturnHistogram;
  var: {
    historical: VarResult;
    parametric: VarResult;
    monte_carlo: VarResult;
  };
  rolling_window: number;
  var_confidence: number;
}

export interface BacktestResponse {
  metrics: PerformanceMetrics;
  equity_curve: EquityCurvePoint[];
  total_days: number;
  diagnostics?: BacktestDiagnostics;
  coverage?: InvestedCoverage;
}

export interface PairsBacktestRequest {
  symbol_y: string;
  symbol_x: string;
  start_date?: string;
  end_date?: string;
  hedge_window?: number;
  zscore_window?: number;
  entry_z?: number;
  exit_z?: number;
  transaction_cost_bps?: number;
  signal_lag_days?: number;
  train_frac?: number;
}

export interface PairsBacktestResponse {
  metrics: PerformanceMetrics;
  equity_curve: EquityCurvePoint[];
  total_days: number;
  diagnostics: {
    symbol_y: string;
    symbol_x: string;
    engle_granger: {
      hedge_ratio: number;
      intercept: number;
      adf_stat: number;
      adf_pvalue: number;
      n_obs: number;
    };
    hedge_window: number;
    zscore_window: number;
    entry_z: number;
    exit_z: number;
    transaction_cost: number;
    signal_lag_days: number;
    n_days: number;
    pct_days_in_trade: number;
  };
  spread_series: { date: string; zscore: number; position: number }[];
  is_held_out: boolean;
  train_start_date?: string | null;
  train_end_date?: string | null;
  held_out_start_date?: string | null;
  train_diagnostics?: {
    hedge_ratio: number;
    intercept: number;
    adf_stat: number;
    adf_pvalue: number;
    n_obs: number;
  } | null;
}

export interface PairsScreenRequest {
  sector?: string;
  symbols?: string[];
  method?: "gatev" | "engle_granger";
  use_adv?: boolean;
  max_symbols?: number;
  start_date?: string;
  end_date?: string;
  train_frac?: number;
  min_train_corr?: number;
  max_train_adf_pvalue?: number;
  max_oos_backtests?: number;
  hedge_window?: number;
  zscore_window?: number;
  entry_z?: number;
  exit_z?: number;
  transaction_cost_bps?: number;
}

export interface PairsScreenRow {
  symbol_y: string;
  symbol_x: string;
  formation_ssd?: number | null;
  train_corr?: number | null;
  train_adf_pvalue?: number | null;
  train_hedge_ratio?: number | null;
  oos_sharpe: number;
  oos_annualized_return: number;
  oos_max_drawdown: number;
  oos_n_days: number;
  oos_pct_days_in_trade: number;
}

export interface PairsScreenResponse {
  symbols: string[];
  split_date: string;
  train_frac: number;
  method?: string;
  n_pairs_tested: number;
  n_pairs_passed_train: number;
  results: PairsScreenRow[];
}

export interface PairsIndexBacktestRequest {
  sector_names: string[];
  start_date?: string;
  end_date?: string;
  formation_months?: number;
  trading_months?: number;
  top_n_pairs?: number;
  max_symbols_per_sector?: number;
  use_adv?: boolean;
  hedge_window?: number;
  zscore_window?: number;
  entry_z?: number;
  exit_z?: number;
  transaction_cost_bps?: number;
  signal_lag_days?: number;
}

export interface PairsIndexPairRow {
  symbol_y: string;
  symbol_x: string;
  sector: string;
  formation_ssd: number;
  formation_adf_pvalue?: number | null;
  period_sharpe: number;
  period_n_days: number;
}

export interface PairsIndexPeriodRow {
  formation_start: string;
  formation_end: string;
  trading_start: string;
  trading_end: string;
  n_candidates_formed: number;
  n_pairs_selected: number;
  avg_active_pairs: number;
  blended_sharpe?: number | null;
  selected_pairs: PairsIndexPairRow[];
}

export interface PairsIndexBacktestResponse {
  metrics: PerformanceMetrics;
  equity_curve: EquityCurvePoint[];
  total_days: number;
  universe: string[];
  periods: PairsIndexPeriodRow[];
}

export interface PairsPersistentBacktestRequest {
  sector_names: string[];
  start_date?: string;
  end_date?: string;
  formation_months?: number;
  rescreen_months?: number;
  top_n_pairs?: number;
  max_symbols_per_sector?: number;
  use_adv?: boolean;
  min_crossings?: number;
  max_adf_pvalue?: number;
  hedge_window?: number;
  zscore_window?: number;
  entry_z?: number;
  exit_z?: number;
  transaction_cost_bps?: number;
  signal_lag_days?: number;
  monitor_window?: number;
  check_every_days?: number;
  stop_max_pvalue?: number;
  persistence_checks?: number;
  freeze_hedge_in_trade?: boolean;
}

export interface PairsPersistentPairRow {
  symbol_y: string;
  symbol_x: string;
  sector: string;
  formation_adf_pvalue: number;
  formation_crossings: number;
  trading_start: string;
  stop_date?: string | null;
  stopped_early: boolean;
  n_days: number;
}

export interface PairsPersistentScreenRow {
  formation_start: string;
  formation_end: string;
  active_before: number;
  free_slots: number;
  n_candidates_found: number;
  n_selected: number;
}

export interface PairsPersistentBacktestResponse {
  metrics: PerformanceMetrics;
  equity_curve: EquityCurvePoint[];
  total_days: number;
  screens: PairsPersistentScreenRow[];
  pair_history: PairsPersistentPairRow[];
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
  shap_importance: FeatureImportanceItem[] | null;
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
  n_long?: number | null;
  n_short?: number | null;
}

export interface ReplayFramesResponse {
  total_frames: number;
  returned_frames: number;
  frames: ReplayFrame[];
  coverage: InvestedCoverage;
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
  n_trials: number;
}

export interface BootstrapResponse {
  symbol: string;
  x: number;
  k: number;
  actual_hit_rate: number;
  random_mean: number;
  random_std: number;
  p_value: number;
  p_value_adjusted: number;
  n_trials: number;
  significant: boolean;
  significant_after_correction: boolean;
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

export interface WalkForwardOptimizeRequest {
  symbols: string[];
  start_date: string;
  end_date?: string;
  lookback_months?: number;
  rebalance_months?: number;
  risk_free_rate?: number;
  portfolio_kind?: "tangency" | "min_variance";
}

export interface WalkForwardPeriod {
  fit_start: string;
  hold_start: string;
  hold_end: string;
  fit_n_obs: number;
  weights: PortfolioWeights;
}

export interface WalkForwardOptimizeResponse {
  metrics: PerformanceMetrics;
  equity_curve: EquityCurvePoint[];
  total_days: number;
  periods: WalkForwardPeriod[];
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

/* ── Measures Lab ─────────────────────────────────────────────── */

export interface MeasureSet {
  cid1_ratio: number;
  cid2_ratio: number;
  total_return: number;
  typical_period_return: number;
  annualized_return: number;
  annualized_volatility: number;
  max_drawdown: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  calmar_ratio: number;
  pain_ratio: number;
  martin_ratio: number;
}

export interface MeasuresLabSeries {
  name: string;
  color: string;
  prices: number[];
  measures: MeasureSet;
}

export interface MeasuresLabRequest {
  n_days?: number;
  n_relationship_draws?: number;
  seed?: number;
  portfolio_weight_a?: number;
  portfolio_a?: string;
  portfolio_b?: string;
}

export interface MeasuresLabResponse {
  single_stock_examples: MeasuresLabSeries[];
  portfolio_example: MeasuresLabSeries;
  portfolio_legs: MeasuresLabSeries[];
  relationship_scatter: Record<string, number[]>;
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
  edgar_note?: string;
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

/* ── Simple Top-500 Index + Cid-1 study ─────────────────────── */

export interface Top500RebalanceSummary {
  date: string;
  n_constituents: number;
  turnover: number | null;
  top_weight: number;
  top10_weight_share: number;
}

export interface Top500PerformanceResponse {
  dates: string[];
  index_cumulative: number[];
  benchmark_cumulative: number[] | null;
  metrics: Record<string, number | null>;
  benchmark_metrics: Record<string, number | null> | null;
  correlation_vs_benchmark: number | null;
  tracking_error_ann: number | null;
  rebalances: Top500RebalanceSummary[];
}

export interface Top500HoldingRow {
  symbol: string;
  sector: string | null;
  market_cap: number;
  weight: number;
  total_return: number | null;
  cost_basis_pain: number | null;
  cid1_ratio: number | null;
  cid1_angle: number | null;
}

export interface Top500HoldingsResponse {
  date: string;
  window_days: number;
  holdings: Top500HoldingRow[];
}

export interface Cid1StatSummary {
  mean: number | null;
  tstat: number | null;
  pvalue: number | null;
  n_obs: number;
}

export interface Cid1StudyDateRow {
  date: string;
  n_symbols: number;
  ic: number | null;
  persistence_vs_prev: number | null;
  median_cid1_angle: number | null;
  share_pain_free: number | null;
  quantile_spread: number | null;
}

export interface Cid1SensitivityRow {
  start_year: number;
  n_quarters: number;
  mean_ic: number | null;
  ic_tstat: number | null;
  mean_spread: number | null;
  spread_tstat: number | null;
}

export interface Cid1StudyResponse {
  per_date: Cid1StudyDateRow[];
  quantile_avg_forward_returns: (number | null)[];
  ic_summary: Cid1StatSummary;
  persistence_summary: Cid1StatSummary;
  quantile_spread_summary: Cid1StatSummary;
  fama_macbeth_univariate: Cid1StatSummary;
  fama_macbeth_multivariate: Record<string, Cid1StatSummary>;
  start_year_sensitivity: Cid1SensitivityRow[];
  config: Record<string, unknown>;
}

// ---------- Data health (GET /data-health) ----------

export interface FunnelStage {
  stage: string;
  count: number;
  scope: "vendor catalog" | "our universe" | "what we hold";
  definition: string;
  why_smaller: string | null;
  note: string;
}

export interface RegistryCaveat {
  id: string;
  kind: "data" | "method";
  severity: "high" | "medium" | "low";
  title: string;
  detail: string;
  remediation: string | null;
}

export interface GlossaryTerm {
  term: string;
  definition: string;
}

export interface DataFlaw {
  id: string;
  severity: "high" | "medium" | "low";
  title: string;
  detail: string;
}

export interface DatasetCoverage {
  files?: number;
  universe_with_data?: number;
  universe_empty_file?: number;
  universe_missing_file?: number;
  corrupt_files?: number;
  status?: string;
}

export interface PanelEntry {
  file: string;
  status?: string;
  rows?: number;
  columns?: number;
  symbols?: number | null;
  modified?: string;
  size_mb?: number;
}

export interface DelistEraCoverage {
  delist_year: string;
  names: number;
  with_prices_pct: number;
}

export interface DataHealthSnapshot {
  generated_at: string;
  universe: {
    total: number;
    live: number;
    delisted: number;
    carried_over: number;
    by_exchange: Record<string, number>;
    market_cap_floor_usd: number;
  };
  funnel: FunnelStage[];
  prices: {
    fetch_outcomes: Record<string, number>;
    failed_symbols: string[];
    history_years: { median: number; p10: number; p90: number };
    symbols_starting_by_1990: number;
    last_date_max: string;
  };
  survivorship: {
    delisted_total: number;
    with_prices_pct: number;
    by_delist_era: DelistEraCoverage[];
    price_end_within_30d_of_delisting_pct: number;
    missing_symbols_sample: string[];
    missing_count: number;
  };
  calendar: {
    canonical_trading_days?: number;
    expanded_dates?: number;
    non_trading_dates?: number;
    weekend_dates?: number;
    holiday_or_other_dates?: number;
    sample?: string[];
    status?: string;
  };
  publication_dates: {
    sampled_symbols?: number;
    sampled_rows?: number;
    placeholder_accepted_date_pct?: number;
    by_decade?: { decade: number; rows: number; placeholder_pct: number }[];
    remediation?: string;
    status?: string;
  };
  datasets: Record<string, DatasetCoverage>;
  intraday: Record<
    string,
    { symbols: number; symbol_year_files: number; empty_symbol_years: number; year_range: string | null }
  >;
  panels: PanelEntry[];
  flaws: DataFlaw[];
  registry_caveats: RegistryCaveat[];
  glossary: GlossaryTerm[];
}

export interface SymbolStatementInfo {
  quarters: number;
  first_period?: string;
  last_period?: string;
  placeholder_filing_dates_pct?: number;
}

export interface SymbolDetail {
  symbol: string;
  universe?: {
    company_name: string | null;
    exchange: string | null;
    sector: string | null;
    is_delisted: boolean | null;
    ipo_date: string | null;
    delisted_date: string | null;
    market_cap: number | null;
  };
  prices?: { rows: number; first: string | null; last: string | null; has_ohlc: boolean };
  statements: Record<string, SymbolStatementInfo | null>;
  market_caps_rows: number | null;
  datasets: Record<string, number | null>;
  intraday: Record<string, { years: string[]; rows: number }>;
}

// ---------- Sector performance (GET /sectors/performance) ----------

export interface SectorSeriesPoint {
  date: string;
  level: number;
}

export interface SectorPerformanceEntry {
  sector: string;
  ann_return_pct: number;
  members_latest: number;
  series: SectorSeriesPoint[];
}

export interface SectorMethodology {
  rebalance: string;
  membership: string;
  returns: string;
  min_members_per_day: number;
  universe_labeled_symbols: number;
}

export interface SectorPerformanceResponse {
  weighting: "cap" | "equal";
  sp500_membership_filter: boolean;
  start: string;
  granularity: string;
  methodology: SectorMethodology;
  sectors: SectorPerformanceEntry[];
  caveats: RegistryCaveat[];
}

// ---------- PEAD event study (GET /backtest/events/pead-study) ----------

export interface PeadQuantilePath {
  quantile: string;
  car_pct: number[];
}

export interface PeadStudyResponse {
  signal: string;
  horizon_days: number;
  n_quantiles: number;
  n_events: number;
  event_counts: Record<string, number>;
  first_event: string;
  last_event: string;
  spread_t_stat: number;
  spread_final_pct: number;
  quantile_paths: PeadQuantilePath[];
  event_days: number[];
  caveats: RegistryCaveat[];
}

/* ── Watchdog ─────────────────────────────────────────────── */

export interface WatchdogCheck {
  name: string;
  status: "ok" | "warning" | "error";
  detail: string;
  facts: Record<string, unknown>;
}

export interface WatchdogStatus {
  status: "ok" | "warning" | "error" | "unknown";
  generated_at: string | null;
  n_errors: number;
  n_warnings: number;
  deep?: boolean;
  checks: WatchdogCheck[];
  summary: string;
}

/* ── Data explorer ────────────────────────────────────────── */

export interface ExplorerDataset {
  name: string;
  description: string;
  grain: string;
  family: "factor" | "reference";
  columns: string[];
  n_columns: number;
}

export interface ExplorerSearchResult {
  symbol: string;
  company_name: string | null;
  exchange: string | null;
  sector: string | null;
  industry: string | null;
  is_delisted: boolean | null;
}

export interface CompanyFactorValue {
  family: string;
  factor: string;
  value: number;
  as_of: string | null;
}

export interface CompanyPricePoint {
  date: string;
  adj_close: number;
}

export interface CompanyProfile {
  symbol: string;
  identity: Record<string, unknown>;
  classification: Record<string, unknown> | null;
  identifiers: Record<string, unknown> | null;
  index_membership: Record<string, unknown>[];
  factors: CompanyFactorValue[];
  prices: CompanyPricePoint[];
}

export interface ExplorerFilterSpec {
  column: string;
  operator: string;
  value: unknown;
  value2?: unknown;
}

export interface ExplorerScreenRequest {
  columns: string[];
  filters: ExplorerFilterSpec[];
  panels?: string[];
  as_of?: string;
  order_by?: string;
  descending?: boolean;
  limit?: number;
  preview_only?: boolean;
}

export interface ExplorerQueryResult {
  sql: string;
  columns: string[];
  rows: Record<string, unknown>[];
  row_count: number;
  truncated: boolean;
  elapsed_ms?: number;
  dtypes?: Record<string, string>;
}

/* ── Glossary ─────────────────────────────────────────────── */

export interface GlossaryRegistryTerm {
  term: string;
  category: string;
  definition: string;
  see_also: string[];
}

export interface GlossaryResponse {
  categories: { id: string; label: string }[];
  terms: GlossaryRegistryTerm[];
}

/* ── Research notes ───────────────────────────────────────── */

export interface ResearchResultRow {
  variant: string;
  gross_sharpe: number | null;
  net_sharpe: number | null;
  net_annual_return: number | null;
  t_stat: number | null;
  hit_rate: number | null;
  max_drawdown: number | null;
  note: string | null;
  is_control: boolean;
  is_headline: boolean;
}

export interface ResearchNoteSummary {
  id: string;
  title: string;
  run_date: string;
  verdict: "validated" | "interesting" | "no_edge" | "real_not_tradable";
  verdict_label: string;
  one_liner: string;
}

export interface ResearchNote extends ResearchNoteSummary {
  question: string;
  hypothesis: string;
  reference: string;
  method: string;
  results: ResearchResultRow[];
  control_reading: string;
  what_it_means: string;
  caveats: string[];
  reproduce: string;
  decade_table: string[][];
  glossary_terms: string[];
  logged_in: string;
}

export interface ResearchNotesIndex {
  notes: ResearchNoteSummary[];
  verdicts: { id: string; label: string }[];
}

// ─── Data Monitor ─────────────────────────────────────────────────────

export type MonitorTransform = "level" | "diff" | "pct_change" | "log_return";
export type StaleStatus = "fresh" | "late" | "stale" | "empty";

export interface MonitoredSeries {
  id: string;
  name: string;
  group: string;
  source: "fmp" | "fred";
  unit: string;
  frequency: string;
  default_transform: MonitorTransform;
  lag_days: number;
  expected_max_gap_days: number;
  rolling_window: number;
}

export interface MonitorCatalogResponse {
  groups: { id: string; label: string; series: MonitoredSeries[] }[];
  transforms: MonitorTransform[];
}

export interface DistributionSummary {
  n: number;
  start: string | null;
  end: string | null;
  last_value: number | null;
  mean: number | null;
  std: number | null;
  min: number | null;
  p05: number | null;
  p25: number | null;
  median: number | null;
  p75: number | null;
  p95: number | null;
  max: number | null;
  skew: number | null;
  kurtosis: number | null;
  autocorr_lag1: number | null;
  percentile_of_last: number | null;
  zscore_of_last: number | null;
  adf_pvalue: number | null;
}

export interface MonitorHistogram {
  edges: number[];
  counts: number[];
  mark: number | null;
  mark_bin: number | null;
}

export interface MonitorLevelPoint {
  date: string;
  level: number | null;
  rolling_mean: number | null;
  upper: number | null;
  lower: number | null;
  zscore: number | null;
}

export interface MonthStat {
  month: number;
  n: number;
  mean: number | null;
  median: number | null;
  hit_rate: number | null;
}

export interface SeasonalityTable {
  years: number[];
  matrix: (number | null)[][];
  month_stats: MonthStat[];
  transform: MonitorTransform;
}

export interface AnnualPaths {
  years: number[];
  paths: Record<string, { doy: number; value: number }[]>;
  normalized: boolean;
}

export interface StalenessReport {
  series_id: string;
  last_date: string | null;
  days_since_last: number | null;
  expected_max_gap_days: number;
  status: StaleStatus;
}

export interface Caveat {
  id: string;
  kind: string;
  severity: string;
  title: string;
  detail: string;
  remediation: string | null;
}

export interface MonitorSeriesResponse {
  series: MonitoredSeries;
  transform: MonitorTransform;
  summary: DistributionSummary;
  histogram: MonitorHistogram;
  levels: MonitorLevelPoint[];
  seasonality: SeasonalityTable;
  annual_paths: AnnualPaths;
  staleness: StalenessReport;
  methodology: Record<string, string>;
  caveats: Caveat[];
}

export interface CurveSnapshot {
  requested: string;
  date: string;
  points: { id: string; tenor_years: number; yield: number | null }[];
}

export interface CurveShapePoint {
  date: string;
  spread_2s10s?: number | null;
  spread_3m10y?: number | null;
  spread_5s30s?: number | null;
  butterfly_2_5_10?: number | null;
  level?: number | null;
}

export interface YieldCurveResponse {
  tenors: { id: string; tenor_years: number; name: string }[];
  snapshots: CurveSnapshot[];
  shape_history: CurveShapePoint[];
  methodology: Record<string, string>;
  caveats: Caveat[];
}

export interface StalenessBoardRow extends StalenessReport {
  name: string;
  group: string;
  source: "fmp" | "fred";
  frequency: string;
}

export interface StalenessBoardResponse {
  as_of: string;
  counts: Record<StaleStatus, number>;
  series: StalenessBoardRow[];
}
