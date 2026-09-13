import { useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";

import DisclosureBlock from "@/components/cards/DisclosureBlock.tsx";
import KPICard from "@/components/cards/KPICard.tsx";
import AppLayout from "@/components/layout/AppLayout.tsx";
import BottomPanel from "@/components/layout/BottomPanel.tsx";
import LeftSidebar from "@/components/layout/LeftSidebar.tsx";
import RightSidebar from "@/components/layout/RightSidebar.tsx";
import AnnualPathsChart from "@/components/monitor/AnnualPathsChart.tsx";
import DistributionHistogram from "@/components/monitor/DistributionHistogram.tsx";
import LevelProfileChart from "@/components/monitor/LevelProfileChart.tsx";
import SeasonalityHeatmap from "@/components/monitor/SeasonalityHeatmap.tsx";
import StalenessTable from "@/components/monitor/StalenessTable.tsx";
import { CurveShapeHistory, YieldCurveSnapshots } from "@/components/monitor/YieldCurveChart.tsx";
import { api } from "@/lib/api.ts";
import type { DistributionSummary, MonitorTransform } from "@/lib/types.ts";
import { cn } from "@/lib/utils.ts";

const TRANSFORM_LABEL: Record<MonitorTransform, string> = {
  level: "level",
  diff: "daily change",
  pct_change: "simple return",
  log_return: "log return",
};

const START_OPTIONS: { label: string; years: number | null }[] = [
  { label: "All", years: null },
  { label: "20y", years: 20 },
  { label: "10y", years: 10 },
  { label: "5y", years: 5 },
  { label: "2y", years: 2 },
];

type Tab = "series" | "curve";

function fmt(v: number | null | undefined, digits = 3): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "–";
  if (Math.abs(v) >= 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
  return v.toFixed(digits);
}

function fmtValue(v: number | null | undefined, isReturn: boolean): string {
  if (v === null || v === undefined) return "–";
  return isReturn ? `${(v * 100).toFixed(2)}%` : fmt(v);
}

function StatRow({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="flex items-baseline justify-between text-[11px] py-0.5 border-b border-zinc-800/60">
      <span className="text-zinc-500">
        {label}
        {hint ? <span className="text-zinc-700 ml-1">{hint}</span> : null}
      </span>
      <span className="font-mono tabular-nums text-zinc-200">{value}</span>
    </div>
  );
}

function SummaryPanel({ s, isReturn }: { s: DistributionSummary; isReturn: boolean }) {
  const pctClass =
    s.percentile_of_last === null
      ? "neutral"
      : s.percentile_of_last >= 90 || s.percentile_of_last <= 10
        ? "negative"
        : "neutral";
  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-2 gap-2">
        <KPICard label="Latest" value={fmtValue(s.last_value, isReturn)} />
        <KPICard
          label="Percentile"
          value={s.percentile_of_last === null ? "–" : `p${s.percentile_of_last.toFixed(0)}`}
          accent={pctClass}
        />
        <KPICard label="Z-score" value={fmt(s.zscore_of_last, 2)} />
        <KPICard
          label="ADF p"
          value={s.adf_pvalue === null ? "–" : s.adf_pvalue < 0.001 ? "<0.001" : s.adf_pvalue.toFixed(3)}
          accent={s.adf_pvalue !== null && s.adf_pvalue > 0.1 ? "negative" : "neutral"}
        />
      </div>
      <div>
        <StatRow label="n" value={String(s.n)} hint={s.start && s.end ? `${s.start} → ${s.end}` : ""} />
        <StatRow label="mean" value={fmtValue(s.mean, isReturn)} />
        <StatRow label="std" value={fmtValue(s.std, isReturn)} />
        <StatRow label="min / max" value={`${fmtValue(s.min, isReturn)} / ${fmtValue(s.max, isReturn)}`} />
        <StatRow label="p5 / p95" value={`${fmtValue(s.p05, isReturn)} / ${fmtValue(s.p95, isReturn)}`} />
        <StatRow label="p25 / median / p75" value={`${fmtValue(s.p25, isReturn)} / ${fmtValue(s.median, isReturn)} / ${fmtValue(s.p75, isReturn)}`} />
        <StatRow label="skew" value={fmt(s.skew, 2)} />
        <StatRow label="excess kurtosis" value={fmt(s.kurtosis, 2)} />
        <StatRow label="autocorr (lag 1)" value={fmt(s.autocorr_lag1, 2)} />
      </div>
    </div>
  );
}

export default function DataMonitor() {
  // Tab and series live in the URL so a row on the freshness board, a chat
  // message, or a screenshot script can point at exactly one view.
  const [params, setParams] = useSearchParams();
  const [tab, setTab] = useState<Tab>(params.get("tab") === "curve" ? "curve" : "series");
  const [seriesId, setSeriesId] = useState(params.get("series") ?? "GLD");
  useEffect(() => {
    const next = new URLSearchParams();
    if (tab === "curve") next.set("tab", "curve");
    if (seriesId !== "GLD") next.set("series", seriesId);
    if (next.toString() !== params.toString()) setParams(next, { replace: true });
  }, [tab, seriesId, params, setParams]);
  const [transform, setTransform] = useState<MonitorTransform | undefined>(undefined);
  const [startYears, setStartYears] = useState<number | null>(null);

  const start = useMemo(() => {
    if (startYears === null) return undefined;
    const d = new Date();
    d.setFullYear(d.getFullYear() - startYears);
    return d.toISOString().slice(0, 10);
  }, [startYears]);

  const catalog = useQuery({ queryKey: ["monitor-catalog"], queryFn: api.getMonitorCatalog });
  const series = useQuery({
    queryKey: ["monitor-series", seriesId, transform ?? "default", start ?? "all"],
    queryFn: () => api.getMonitorSeries(seriesId, transform, start),
    enabled: tab === "series",
  });
  const curve = useQuery({
    queryKey: ["monitor-curve"],
    queryFn: () => api.getYieldCurve(),
    enabled: tab === "curve",
  });
  const board = useQuery({ queryKey: ["monitor-staleness"], queryFn: api.getStalenessBoard });

  const data = series.data;
  const usedTransform = data?.transform ?? transform ?? "level";
  const spec = data?.series;
  // Returns and ratio-valued levels (YoY inflation stored as 0.033) both read as percentages.
  const isReturn =
    usedTransform === "pct_change" ||
    usedTransform === "log_return" ||
    (usedTransform !== "diff" && spec?.unit === "ratio");

  const selectSeries = (id: string) => {
    setSeriesId(id);
    setTransform(undefined);
    setTab("series");
  };

  return (
    <AppLayout
      left={
        <LeftSidebar>
          <div className="flex gap-1">
            {(["series", "curve"] as Tab[]).map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className={cn(
                  "flex-1 text-[11px] py-1 border",
                  tab === t
                    ? "border-blue-500 text-blue-400 bg-blue-950/30"
                    : "border-zinc-800 text-zinc-400 hover:text-zinc-200",
                )}
              >
                {t === "series" ? "Series" : "Yield curve"}
              </button>
            ))}
          </div>

          {tab === "series" ? (
            <>
              <div>
                <label className="text-[10px] uppercase tracking-wider text-zinc-500">Series</label>
                <select
                  value={seriesId}
                  onChange={(e) => {
                    setSeriesId(e.target.value);
                    setTransform(undefined);
                  }}
                  className="mt-1 w-full bg-zinc-900 border border-zinc-800 text-zinc-200 text-[11px] px-2 py-1 font-mono"
                >
                  {(catalog.data?.groups ?? []).map((g) => (
                    <optgroup key={g.id} label={g.label}>
                      {g.series.map((s) => (
                        <option key={s.id} value={s.id}>
                          {s.id} — {s.name}
                        </option>
                      ))}
                    </optgroup>
                  ))}
                </select>
              </div>

              <div>
                <label className="text-[10px] uppercase tracking-wider text-zinc-500">
                  Distribution of
                </label>
                <div className="mt-1 grid grid-cols-2 gap-1">
                  {(catalog.data?.transforms ?? []).map((t) => (
                    <button
                      key={t}
                      onClick={() => setTransform(t)}
                      className={cn(
                        "text-[10px] py-1 border",
                        usedTransform === t
                          ? "border-blue-500 text-blue-400 bg-blue-950/30"
                          : "border-zinc-800 text-zinc-400 hover:text-zinc-200",
                      )}
                    >
                      {TRANSFORM_LABEL[t]}
                    </button>
                  ))}
                </div>
                {spec ? (
                  <p className="text-[10px] text-zinc-600 mt-1">
                    default for this series: {TRANSFORM_LABEL[spec.default_transform]}
                  </p>
                ) : null}
              </div>

              <div>
                <label className="text-[10px] uppercase tracking-wider text-zinc-500">History</label>
                <div className="mt-1 flex gap-1">
                  {START_OPTIONS.map((o) => (
                    <button
                      key={o.label}
                      onClick={() => setStartYears(o.years)}
                      className={cn(
                        "flex-1 text-[10px] py-1 border",
                        startYears === o.years
                          ? "border-blue-500 text-blue-400 bg-blue-950/30"
                          : "border-zinc-800 text-zinc-400 hover:text-zinc-200",
                      )}
                    >
                      {o.label}
                    </button>
                  ))}
                </div>
              </div>

              {spec ? (
                <div className="text-[10px] text-zinc-500 space-y-0.5 border-t border-zinc-800 pt-2">
                  <div>
                    source <span className="text-zinc-300">{spec.source.toUpperCase()}</span> ·{" "}
                    {spec.frequency} · {spec.unit}
                  </div>
                  <div>
                    publication lag <span className="text-zinc-300">{spec.lag_days}d</span> · late
                    after <span className="text-zinc-300">{spec.expected_max_gap_days}d</span>
                  </div>
                </div>
              ) : null}
            </>
          ) : (
            <p className="text-[11px] text-zinc-500 leading-relaxed">
              Constant-maturity Treasury yields from FRED. Snapshots: today, 1M, 3M, 1Y and 2Y
              ago, each resolved to the last observation on or before that date.
            </p>
          )}
        </LeftSidebar>
      }
      right={
        <RightSidebar>
          {tab === "series" && data ? (
            <SummaryPanel s={data.summary} isReturn={isReturn} />
          ) : tab === "curve" && curve.data ? (
            <div className="flex flex-col gap-2">
              {curve.data.snapshots.map((s) => {
                const by = Object.fromEntries(s.points.map((p) => [p.id, p.yield]));
                const ten = by["dgs10"];
                const two = by["dgs2"];
                const three = by["dgs3mo"];
                return (
                  <div key={s.date} className="text-[11px] border-b border-zinc-800/60 pb-1">
                    <div className="text-zinc-300 font-mono">{s.date}</div>
                    <div className="text-zinc-500 font-mono tabular-nums">
                      10y {fmt(ten ?? null, 2)} · 2s10s{" "}
                      {ten != null && two != null ? (ten - two).toFixed(2) : "–"} · 3m10y{" "}
                      {ten != null && three != null ? (ten - three).toFixed(2) : "–"}
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="text-[11px] text-zinc-600">loading…</div>
          )}
        </RightSidebar>
      }
      bottom={
        <BottomPanel>
          {board.data ? (
            <StalenessTable
              rows={board.data.series}
              asOf={board.data.as_of}
              snapshots={board.data.snapshots}
              onSelect={selectSeries}
            />
          ) : (
            <div className="text-[11px] text-zinc-600">checking freshness…</div>
          )}
        </BottomPanel>
      }
    >
      {tab === "series" ? (
        series.isLoading ? (
          <div className="text-[11px] text-zinc-600">loading {seriesId}…</div>
        ) : series.isError ? (
          <div className="text-[11px] text-red-400">{String(series.error)}</div>
        ) : data ? (
          <div className="flex flex-col gap-4">
            <div className="flex items-baseline justify-between">
              <h2 className="text-sm text-zinc-200">
                {data.series.id}{" "}
                <span className="text-zinc-500 font-normal">{data.series.name}</span>
              </h2>
              <span
                className={cn(
                  "text-[10px] font-mono",
                  data.staleness.status === "fresh"
                    ? "text-emerald-400"
                    : data.staleness.status === "late"
                      ? "text-amber-400"
                      : "text-red-400",
                )}
              >
                last {data.staleness.last_date} · {data.staleness.days_since_last}d ·{" "}
                {data.staleness.status}
                {data.staleness.snapshot_as_of
                  ? ` · frozen at snapshot ${data.staleness.snapshot_as_of}`
                  : ""}
              </span>
            </div>

            <section className="bg-zinc-900 border border-zinc-800 p-2">
              <div className="text-[10px] uppercase tracking-wider text-zinc-500 mb-1">
                Level · rolling mean and ±2σ band ({data.series.rolling_window} obs)
              </div>
              <LevelProfileChart
                points={data.levels}
                unit={data.series.unit}
                windowLabel={`${data.series.rolling_window} obs`}
              />
            </section>

            <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
              <section className="bg-zinc-900 border border-zinc-800 p-2">
                <div className="text-[10px] uppercase tracking-wider text-zinc-500 mb-1">
                  Distribution of {TRANSFORM_LABEL[usedTransform]} · latest marked
                </div>
                <DistributionHistogram
                  histogram={data.histogram}
                  transformLabel={TRANSFORM_LABEL[usedTransform]}
                  percentile={data.summary.percentile_of_last}
                />
              </section>
              <section className="bg-zinc-900 border border-zinc-800 p-2">
                <div className="text-[10px] uppercase tracking-wider text-zinc-500 mb-1">
                  Years overlaid · {data.annual_paths.normalized ? "rebased to 100" : "levels"}
                </div>
                <AnnualPathsChart paths={data.annual_paths} unit={data.series.unit} />
              </section>
            </div>

            <section className="bg-zinc-900 border border-zinc-800 p-2">
              <div className="text-[10px] uppercase tracking-wider text-zinc-500 mb-1">
                Seasonality · year × month {TRANSFORM_LABEL[usedTransform]} · column mean, hit-rate, n
              </div>
              <SeasonalityHeatmap
                table={data.seasonality}
                isReturn={isReturn}
                height={Math.min(520, 120 + 14 * data.seasonality.years.length)}
              />
            </section>

            <DisclosureBlock methodology={data.methodology} caveats={data.caveats} />
          </div>
        ) : null
      ) : curve.isLoading ? (
        <div className="text-[11px] text-zinc-600">loading curve…</div>
      ) : curve.isError ? (
        <div className="text-[11px] text-red-400">{String(curve.error)}</div>
      ) : curve.data ? (
        <div className="flex flex-col gap-4">
          <section className="bg-zinc-900 border border-zinc-800 p-2">
            <div className="text-[10px] uppercase tracking-wider text-zinc-500 mb-1">
              Treasury curve · snapshots
            </div>
            <YieldCurveSnapshots snapshots={curve.data.snapshots} />
          </section>
          <section className="bg-zinc-900 border border-zinc-800 p-2">
            <div className="text-[10px] uppercase tracking-wider text-zinc-500 mb-1">
              Curve shape through time · below zero = inverted
            </div>
            <CurveShapeHistory history={curve.data.shape_history} />
          </section>
          <DisclosureBlock methodology={curve.data.methodology} caveats={curve.data.caveats} />
        </div>
      ) : null}
    </AppLayout>
  );
}
