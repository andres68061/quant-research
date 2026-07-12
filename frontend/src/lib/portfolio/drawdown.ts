/** Pure helpers for drawdown series, used by NAV-vs-benchmark charts. */

export interface NavPoint {
  date: string;
  value: number;
}

export interface DrawdownPoint {
  date: string;
  dd: number;
}

/**
 * Drawdown from a NAV series ``[{date, value}, ...]``.
 *
 * dd[t] = (value[t] - running_max(value[..t])) / running_max
 */
export function drawdownFromNav(nav: NavPoint[]): DrawdownPoint[] {
  let peak = nav[0]?.value ?? 1;
  return nav.map((p) => {
    if (p.value > peak) peak = p.value;
    return { date: p.date, dd: peak > 0 ? (p.value - peak) / peak : 0 };
  });
}

/**
 * Drawdown from parallel ``dates`` and cumulative-return arrays (e.g. benchmark
 * payload's ``cumulative_returns`` which are wealth indices, not net returns).
 */
export function drawdownFromCum(
  dates: string[],
  cumulative: number[],
): DrawdownPoint[] {
  let peak = cumulative[0] ?? 1;
  return dates.map((d, i) => {
    const v = cumulative[i] ?? peak;
    if (v > peak) peak = v;
    return { date: d, dd: peak > 0 ? (v - peak) / peak : 0 };
  });
}
