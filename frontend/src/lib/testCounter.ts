/**
 * Session-scoped count of backtests run, and the significance bar it implies.
 *
 * The idea: if you try 30 parameter combinations and report the best one, the
 * usual |t| >= 2 bar is meaningless — the best of 30 tries clears it by luck
 * alone. The Sidak correction states the bar that keeps the chance of ANY false
 * positive across the family at 5%.
 *
 * Two honest caveats, surfaced in the UI rather than buried:
 *
 * 1. A formal correction applies to a *pre-specified* family of tests. Counting
 *    clicks approximates research effort; it is not a rigorous family.
 * 2. Backtests over overlapping data are correlated, and Sidak assumes
 *    independence, so the bar it gives is conservative.
 *
 * It exists because the count is the number people forget. Seeing "23 tests;
 * the bar is now 3.09" at the moment of judging a result is the intervention.
 *
 * Lives outside the component file so the component module exports only a
 * component (React Fast Refresh requirement).
 */

const STORAGE_KEY = "quant.test-counter";
const EVENT = "quant:test-counted";

export function getTestCount(): number {
  const stored = Number(sessionStorage.getItem(STORAGE_KEY) ?? "0");
  return Number.isFinite(stored) ? stored : 0;
}

/** Record that a test was run. Call from any page that runs a backtest. */
export function countTest(): void {
  sessionStorage.setItem(STORAGE_KEY, String(getTestCount() + 1));
  window.dispatchEvent(new Event(EVENT));
}

export function resetTestCount(): void {
  sessionStorage.setItem(STORAGE_KEY, "0");
  window.dispatchEvent(new Event(EVENT));
}

export function subscribeTestCount(callback: () => void): () => void {
  window.addEventListener(EVENT, callback);
  return () => window.removeEventListener(EVENT, callback);
}

/**
 * The |t| threshold implied by the Sidak correction for n tests.
 *
 * per-test alpha = 1 - (1 - familyAlpha)^(1/n), two-sided, converted to a
 * z-score. A normal quantile is used rather than a t quantile because backtest
 * samples here run to thousands of days, where the two agree to two decimals.
 */
export function sidakTBar(n: number, familyAlpha = 0.05): number {
  const perTest = 1 - Math.pow(1 - familyAlpha, 1 / Math.max(n, 1));
  return Math.abs(normalQuantile(perTest / 2));
}

/** Inverse standard normal CDF (Acklam's rational approximation, ~1e-9 accurate). */
function normalQuantile(p: number): number {
  if (p <= 0 || p >= 1) return 0;
  const a = [
    -3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.38357751867269e2,
    -3.066479806614716e1, 2.506628277459239,
  ];
  const b = [
    -5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1,
    -1.328068155288572e1,
  ];
  const c = [
    -7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734,
    4.374664141464968, 2.938163982698783,
  ];
  const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];
  const pLow = 0.02425;

  if (p < pLow) {
    const q = Math.sqrt(-2 * Math.log(p));
    return (
      (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    );
  }
  if (p > 1 - pLow) {
    const q = Math.sqrt(-2 * Math.log(1 - p));
    return -(
      (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    );
  }
  const q = p - 0.5;
  const r = q * q;
  return (
    ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) /
    (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
  );
}
