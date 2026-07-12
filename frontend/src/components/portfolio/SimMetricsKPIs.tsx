/**
 * Right-rail KPI block for ``SimulateResponse.metrics``.
 *
 * Used by both the ETF Optimizer and Custom Portfolio pages so they show the
 * same set in the same order.
 */
import KPICard from "@/components/cards/KPICard.tsx";
import type { SimulateResponse } from "@/lib/types.ts";
import { fmtPct, fmtRatio } from "@/lib/utils.ts";

interface Props {
  metrics: SimulateResponse["metrics"];
  /** Heading shown above the cards (e.g. "Portfolio"). */
  title?: string;
}

export default function SimMetricsKPIs({ metrics, title = "Portfolio" }: Props) {
  return (
    <>
      <div className="text-[10px] uppercase tracking-wider text-zinc-500 font-semibold">
        {title}
      </div>
      <KPICard
        label="Total Return"
        value={fmtPct(metrics.total_return)}
        accent={metrics.total_return >= 0 ? "positive" : "negative"}
      />
      <KPICard
        label="Sharpe"
        value={fmtRatio(metrics.sharpe_ratio)}
        accent={metrics.sharpe_ratio >= 0 ? "positive" : "negative"}
      />
      <KPICard
        label="Sortino"
        value={fmtRatio(metrics.sortino_ratio)}
        accent={metrics.sortino_ratio >= 0 ? "positive" : "negative"}
      />
      <KPICard label="Max DD" value={fmtPct(metrics.max_drawdown)} accent="negative" />
      <KPICard label="Volatility" value={fmtPct(metrics.annualized_volatility)} />
      <KPICard
        label="Calmar"
        value={fmtRatio(metrics.calmar_ratio)}
        accent={metrics.calmar_ratio >= 0 ? "positive" : "negative"}
      />
    </>
  );
}
