/**
 * Right-rail KPI block for a ``BenchmarkResponse``.
 *
 * Mirrors the structure of ``SimMetricsKPIs`` so the two read consistently
 * when stacked in the right sidebar.
 */
import KPICard from "@/components/cards/KPICard.tsx";
import type { BenchmarkResponse } from "@/lib/types.ts";
import { fmtPct, fmtRatio } from "@/lib/utils.ts";

interface Props {
  benchmark: BenchmarkResponse;
}

export default function BenchmarkKPIs({ benchmark }: Props) {
  return (
    <>
      <div className="mt-3 text-[10px] uppercase tracking-wider text-zinc-500 font-semibold">
        {benchmark.benchmark_name}
      </div>
      <KPICard
        label="Total Return"
        value={fmtPct(benchmark.total_return)}
        accent={benchmark.total_return >= 0 ? "positive" : "negative"}
      />
      <KPICard
        label="Sharpe"
        value={fmtRatio(benchmark.sharpe_ratio)}
        accent={benchmark.sharpe_ratio >= 0 ? "positive" : "negative"}
      />
      <KPICard label="Max DD" value={fmtPct(benchmark.max_drawdown)} accent="negative" />
      <KPICard label="Volatility" value={fmtPct(benchmark.volatility)} />
    </>
  );
}
