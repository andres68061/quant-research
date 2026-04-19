export interface Asset {
  symbol: string;
  type: string;
}

export interface AssetsResponse {
  count: number;
  assets: Asset[];
}

export interface FactorsResponse {
  count: number;
  factors: string[];
}

export interface PricePoint {
  date: string;
  price: number;
}

export interface PriceSeriesResponse {
  symbol: string;
  count: number;
  data: PricePoint[];
}

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

export interface EquityCurvePoint {
  date: string;
  cumulative_return: number;
}

export interface BacktestRequest {
  universe?: string;
  strategy_type?: string;
  factor_col?: string;
  model_type?: string;
  start_date?: string;
  end_date?: string;
  rebalance_freq?: string;
  transaction_cost_bps?: number;
  top_pct?: number;
  bottom_pct?: number;
  long_only?: boolean;
  survivorship_free?: boolean;
}

export interface BacktestResponse {
  metrics: PerformanceMetrics;
  equity_curve: EquityCurvePoint[];
  total_days: number;
}

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

export interface ReplayFrame {
  date: string;
  signal?: number | null;
  position?: string | null;
  pnl_today?: number | null;
  cumulative_pnl?: number | null;
  drawdown?: number | null;
  rolling_sharpe?: number | null;
}

export interface ReplayResponse {
  total_frames: number;
  returned_frames: number;
  frames: ReplayFrame[];
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
  confusion_matrix?: ConfusionMatrixResult | null;
  folds: FoldResult[];
}

export interface FeatureImportanceItem {
  feature: string;
  importance: number;
}

export interface MLStrategyRequest {
  symbol: string;
  model_type: string;
  initial_train_days: number;
  test_days: number;
  max_splits: number;
}

export interface MLStrategyResponse {
  walkforward: WalkForwardResult;
  feature_importance?: FeatureImportanceItem[] | null;
  metadata: {
    symbol?: string;
    total_features?: number;
    final_rows?: number;
  };
}

export interface GridResult {
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
  results: GridResult[];
}

export interface BootstrapResponse {
  symbol: string;
  x: number;
  k: number;
  p_value: number;
  significant: boolean;
  actual_hit_rate: number;
  random_mean: number;
  n_signals: number;
  bootstrap_dist: number[];
}

export interface RegimeResult {
  strong_momentum: boolean;
  current_sortino: number;
  recent_slope: number;
  baseline_slope: number;
  [key: string]: number | boolean;
}

export interface RegimeResponse {
  symbol: string;
  regime: RegimeResult;
}

export interface SectorSummaryItem {
  sector: string;
  count: number;
  pct: number;
}

export interface SectorSummaryResponse {
  total_symbols: number;
  sectors: SectorSummaryItem[];
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

export interface OptimizeRequest {
  symbols: string[];
  start_date?: string;
  end_date?: string;
  risk_free_rate: number;
  borrowing_rate: number;
}

export interface SimulateRequest {
  symbols: string[];
  weights: Record<string, number>;
  freq: string;
  start_date?: string;
  end_date?: string;
}

export interface PortfolioPoint {
  date: string;
  value: number;
}

export interface SimulateResponse {
  nav: PortfolioPoint[];
  metrics: PerformanceMetrics;
}

export interface FrontierPoint {
  ret: number;
  volatility: number;
}

export interface PortfolioWithWeights extends FrontierPoint {
  sharpe: number;
  weights: Record<string, number>;
}

export interface IndividualAssetPoint {
  symbol: string;
  ret: number;
  volatility: number;
}

export interface OptimizeResponse {
  tangency: PortfolioWithWeights;
  min_vol: PortfolioWithWeights;
  frontier: FrontierPoint[];
  cal: FrontierPoint[];
  individual: IndividualAssetPoint[];
}

export interface BenchmarkRequest {
  benchmark_type: string;
  start_date: string;
  end_date: string;
  sp500_weighting?: string;
  component1?: string;
  component2?: string;
  weight1?: number;
}

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

export interface Cetes28Response {
  rate: number;
  date: string;
}

export interface CommodityInfo {
  symbol: string;
  name: string;
  category: string;
  unit: string;
}

export interface CommodityListResponse {
  commodities: CommodityInfo[];
}

export interface CommodityPriceResponse {
  series: Record<string, PricePoint[]>;
}

export interface CommodityReturnStats {
  symbol: string;
  mean: number;
  annualized: number;
  volatility: number;
  sharpe: number;
  skew: number;
  kurtosis: number;
  latest_price: number;
}

export interface CommodityReturnsResponse {
  stats: CommodityReturnStats[];
  series: Array<Record<string, string | number | null>>;
}

export interface CorrelationResponse {
  symbols: string[];
  matrix: number[][];
}

export interface SeasonalityPoint {
  month: number;
  avg_return: number;
}

export interface SeasonalityHeatmapPoint {
  year: number;
  month: number;
  ret: number;
}

export interface SeasonalityResponse {
  symbol: string;
  monthly_avg: SeasonalityPoint[];
  heatmap: SeasonalityHeatmapPoint[];
}

export interface FREDIndicator {
  id: string;
  name: string;
  unit: string;
}

export interface FREDCategory {
  category: string;
  indicators: FREDIndicator[];
}

export interface FREDCatalogResponse {
  categories: FREDCategory[];
}

export interface FREDSeriesResponse {
  series: Array<Record<string, string | number | null>>;
  metadata: Record<string, { name: string; unit: string }>;
}

export interface RecessionPeriod {
  start: string;
  end: string;
}

export interface RecessionsResponse {
  periods: RecessionPeriod[];
}

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

export interface StockPriceDetail {
  date: string;
  price: number;
  below: boolean;
}

export interface StockDetailResponse {
  symbol: string;
  prices: StockPriceDetail[];
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

export interface SharpeSimulationMetrics {
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
  metrics: SharpeSimulationMetrics;
}

export interface SharpeComparisonResponse {
  target_sharpe: number;
  n_days: number;
  seed: number;
  investments: SimulatedInvestment[];
}
