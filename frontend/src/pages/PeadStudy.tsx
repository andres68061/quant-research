import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import Plot from "react-plotly.js";

import KPICard from "@/components/cards/KPICard.tsx";
import AppLayout from "@/components/layout/AppLayout.tsx";
import LeftSidebar from "@/components/layout/LeftSidebar.tsx";
import RightSidebar from "@/components/layout/RightSidebar.tsx";
import { api } from "@/lib/api.ts";
import { cn } from "@/lib/utils.ts";

// Quintiles are ORDERED (worst -> best surprise), so color is a sequential
// single-hue ramp, light -> dark, not categorical hues.
const QUINTILE_RAMP = ["#bfdbfe", "#93c5fd", "#60a5fa", "#3b82f6", "#1d4ed8"];

const SIGNALS = [
  { id: "sue_price_scaled", label: "EPS surprise / price (default)" },
  { id: "sue_std_scaled", label: "EPS surprise / own history σ" },
  { id: "revenue_surprise_pct", label: "Revenue surprise %" },
];

export default function PeadStudy() {
  const [signal, setSignal] = useState("sue_price_scaled");
  const [horizon, setHorizon] = useState(60);

  const { data, isLoading, error } = useQuery({
    queryKey: ["pead-study", signal, horizon],
    queryFn: () => api.getPeadStudy({ signal, horizonDays: horizon }),
    staleTime: 30 * 60 * 1000,
  });

  const significant = data != null && Math.abs(data.spread_t_stat) >= 2;

  return (
    <AppLayout
      left={
        <LeftSidebar>
          <div className="space-y-3">
            <div>
              <label className="text-[10px] uppercase tracking-wider text-zinc-500 mb-1 block">
                Surprise definition
              </label>
              <select
                value={signal}
                onChange={(e) => setSignal(e.target.value)}
                className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5 text-xs text-zinc-200 font-mono"
              >
                {SIGNALS.map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.label}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="text-[10px] uppercase tracking-wider text-zinc-500 mb-1 block">
                Drift horizon (trading days)
              </label>
              <select
                value={horizon}
                onChange={(e) => setHorizon(Number(e.target.value))}
                className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5 text-xs text-zinc-200 font-mono"
              >
                {[20, 40, 60, 90].map((h) => (
                  <option key={h} value={h}>
                    {h} days
                  </option>
                ))}
              </select>
            </div>
            <p className="text-[10px] text-zinc-600 leading-relaxed">
              Event-time study: each stock&apos;s clock starts at its own announcement. This is
              the correct design for PEAD — the calendar-rebalanced factor runner dilutes it.
            </p>
          </div>
        </LeftSidebar>
      }
      right={
        <RightSidebar>
          {data ? (
            <div className="flex flex-col gap-2">
              <div className="text-[10px] uppercase tracking-wider text-zinc-500">Result</div>
              <KPICard
                label={`Q${data.n_quantiles}−Q1 spread @ day ${data.horizon_days}`}
                value={`${data.spread_final_pct > 0 ? "+" : ""}${data.spread_final_pct.toFixed(2)}%`}
              />
              <KPICard label="t-statistic" value={data.spread_t_stat.toFixed(2)} />
              <KPICard label="Events" value={data.n_events.toLocaleString()} />
              <KPICard label="Span" value={`${data.first_event} → ${data.last_event}`} />
              <div
                className={cn(
                  "text-xs rounded px-2 py-1.5",
                  significant
                    ? "text-emerald-400 bg-emerald-950/30"
                    : "text-zinc-400 bg-zinc-800/60",
                )}
              >
                {significant
                  ? "Spread is statistically significant (|t| ≥ 2)."
                  : "NOT statistically significant — on this large-cap universe the drift is indistinguishable from zero."}
              </div>
            </div>
          ) : null}
        </RightSidebar>
      }
    >
      <div className="p-4">
        <h1 className="text-base text-zinc-100 mb-1">Post-Earnings Announcement Drift</h1>
        <p className="text-xs text-zinc-500 mb-4 max-w-2xl">
          Average cumulative abnormal return after an earnings announcement, grouped into
          surprise quintiles (Q1 = biggest miss, Q5 = biggest beat; breakpoints within each
          calendar quarter). If PEAD exists here, Q5 keeps drifting up and Q1 down after day 0.
        </p>

        {isLoading ? <p className="text-xs text-zinc-500">Running event study…</p> : null}
        {error ? <p className="text-xs text-red-400">Failed to load study.</p> : null}

        {data ? (
          <>
            <Plot
              data={data.quantile_paths.map((path, i) => ({
                x: data.event_days,
                y: path.car_pct,
                type: "scatter" as const,
                mode: "lines" as const,
                name: `${path.quantile}${i === 0 ? " (miss)" : i === data.quantile_paths.length - 1 ? " (beat)" : ""}`,
                line: { color: QUINTILE_RAMP[i] ?? "#71717a", width: i === 0 || i === data.quantile_paths.length - 1 ? 2 : 1.2 },
                hovertemplate: `${path.quantile}<br>day +%{x}<br>CAR %{y:.2f}%<extra></extra>`,
              }))}
              layout={{
                paper_bgcolor: "transparent",
                plot_bgcolor: "transparent",
                font: { color: "#a1a1aa", family: "JetBrains Mono, monospace", size: 11 },
                margin: { t: 8, r: 16, b: 40, l: 56 },
                height: 400,
                xaxis: {
                  gridcolor: "#27272a",
                  linecolor: "#27272a",
                  title: { text: "trading days after announcement", font: { size: 10 } },
                },
                yaxis: {
                  gridcolor: "#27272a",
                  linecolor: "#27272a",
                  zeroline: true,
                  zerolinecolor: "#3f3f46",
                  title: { text: "avg cumulative abnormal return (%)", font: { size: 10 } },
                },
                legend: { orientation: "h", y: -0.18, font: { size: 10 } },
                hovermode: "x unified" as const,
              }}
              config={{ displayModeBar: false, responsive: true }}
              className="w-full"
              useResizeHandler
              style={{ width: "100%" }}
            />

            <details className="mt-3 text-[11px] text-zinc-500 border border-zinc-800 rounded p-2 max-w-2xl">
              <summary className="cursor-pointer text-zinc-400">
                Methodology &amp; caveats ({data.caveats.length})
              </summary>
              <ul className="mt-2 space-y-1 list-disc pl-4">
                {data.caveats.map((c) => (
                  <li key={c}>{c}</li>
                ))}
              </ul>
            </details>
          </>
        ) : null}
      </div>
    </AppLayout>
  );
}
