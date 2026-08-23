import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

import AppLayout from "@/components/layout/AppLayout.tsx";
import LeftSidebar from "@/components/layout/LeftSidebar.tsx";
import { api } from "@/lib/api.ts";
import type {
  DataFlaw,
  DataHealthSnapshot,
  FunnelStage as FunnelStageType,
  RegistryCaveat,
  SymbolDetail,
} from "@/lib/types.ts";
import { cn } from "@/lib/utils.ts";

type Section = "universe" | "datasets" | "flaws" | "symbol";

const SEVERITY_STYLES: Record<DataFlaw["severity"], string> = {
  high: "text-red-400 bg-red-950/40",
  medium: "text-amber-400 bg-amber-950/30",
  low: "text-zinc-400 bg-zinc-800/60",
};

const SECTION_LABELS: Record<Section, string> = {
  universe: "Universe & Survivorship",
  datasets: "Datasets & Panels",
  flaws: "Known Flaws",
  symbol: "Symbol Drilldown",
};

function formatCount(value: number | null | undefined): string {
  return value == null ? "—" : value.toLocaleString();
}

/** Single-measure horizontal bar on a muted track, value direct-labeled. */
function CoverageBar({
  label,
  value,
  max,
  display,
  accent = "bg-blue-400",
  note,
}: {
  label: string;
  value: number;
  max: number;
  display: string;
  accent?: string;
  note?: string;
}) {
  const pct = max > 0 ? Math.max((value / max) * 100, 0.5) : 0;
  return (
    <div className="group" title={note ? `${label}: ${display} — ${note}` : `${label}: ${display}`}>
      <div className="flex items-baseline justify-between gap-2 text-xs">
        <span className="text-zinc-400 truncate">{label}</span>
        <span className="font-mono tabular-nums text-zinc-100 shrink-0">{display}</span>
      </div>
      <div className="mt-1 h-2 rounded-sm bg-zinc-800 overflow-hidden">
        <div className={cn("h-full rounded-sm", accent)} style={{ width: `${pct}%` }} />
      </div>
      {note ? <div className="mt-0.5 text-[10px] text-zinc-500">{note}</div> : null}
    </div>
  );
}

const SCOPE_STYLES: Record<string, string> = {
  "vendor catalog": "text-zinc-500 bg-zinc-800/60",
  "our universe": "text-blue-400 bg-blue-950/30",
  "what we hold": "text-emerald-400 bg-emerald-950/30",
};

/** One funnel stage: the number, what population it describes, and why it shrank. */
function FunnelStage({ stage, max }: { stage: FunnelStageType; max: number }) {
  const pct = max > 0 ? Math.max((stage.count / max) * 100, 0.5) : 0;
  const accent =
    stage.scope === "what we hold"
      ? "bg-emerald-400"
      : stage.scope === "our universe"
        ? "bg-blue-400"
        : "bg-zinc-600";
  return (
    <div className="border-b border-zinc-800/60 pb-3 last:border-0">
      <div className="flex items-baseline justify-between gap-3">
        <div className="flex items-center gap-2 min-w-0">
          <span className="text-xs text-zinc-200">{stage.stage}</span>
          <span
            className={cn("px-1.5 py-0.5 rounded text-[9px] uppercase shrink-0", SCOPE_STYLES[stage.scope])}
          >
            {stage.scope}
          </span>
        </div>
        <span className="font-mono tabular-nums text-sm text-zinc-100 shrink-0">
          {stage.count.toLocaleString()}
        </span>
      </div>
      <div className="mt-1.5 h-2 rounded-sm bg-zinc-800 overflow-hidden">
        <div className={cn("h-full rounded-sm", accent)} style={{ width: `${pct}%` }} />
      </div>
      <p className="mt-1.5 text-[11px] text-zinc-400 leading-relaxed">{stage.definition}</p>
      {stage.why_smaller ? (
        <p className="mt-1 text-[11px] text-amber-500/80 leading-relaxed">
          ↓ {stage.why_smaller}
        </p>
      ) : null}
    </div>
  );
}

function UniverseSection({ snapshot }: { snapshot: DataHealthSnapshot }) {
  const funnelMax = Math.max(...snapshot.funnel.map((s) => s.count));
  const surv = snapshot.survivorship;
  const pub = snapshot.publication_dates;
  return (
    <div className="space-y-6">
      <section className="bg-zinc-900 border border-zinc-800 rounded p-4">
        <h2 className="text-sm text-zinc-100 mb-1">Universe funnel — what each number means</h2>
        <p className="text-xs text-zinc-500 mb-4 leading-relaxed">
          These counts describe <em>different populations</em>, which is why they differ by so
          much. The grey stages are the vendor&apos;s catalog (what exists), blue is the universe
          we chose to download, green is what we actually hold on disk. Only the green numbers
          bound what a backtest can use.
        </p>
        <div className="space-y-3">
          {snapshot.funnel.map((stage) => (
            <FunnelStage key={stage.stage} stage={stage} max={funnelMax} />
          ))}
        </div>
      </section>

      {snapshot.glossary?.length ? (
        <section className="bg-zinc-900 border border-zinc-800 rounded p-4">
          <h2 className="text-sm text-zinc-100 mb-1">Glossary</h2>
          <p className="text-xs text-zinc-500 mb-3">
            Terms used across this page and the docs, defined so nothing here requires prior
            context.
          </p>
          <dl className="space-y-2.5">
            {snapshot.glossary.map((entry) => (
              <div key={entry.term}>
                <dt className="text-xs text-zinc-200 font-mono">{entry.term}</dt>
                <dd className="text-[11px] text-zinc-400 leading-relaxed mt-0.5">
                  {entry.definition}
                </dd>
              </div>
            ))}
          </dl>
        </section>
      ) : null}

      <section className="bg-zinc-900 border border-zinc-800 rounded p-4">
        <h2 className="text-sm text-zinc-100 mb-1">Survivorship</h2>
        <p className="text-xs text-zinc-500 mb-4">
          {surv.with_prices_pct}% of {surv.delisted_total.toLocaleString()} delisted names have
          price history; {surv.price_end_within_30d_of_delisting_pct}% of those end within 30 days
          of the recorded delisting. Coverage by delisting era:
        </p>
        <div className="space-y-3">
          {surv.by_delist_era.map((era) => (
            <CoverageBar
              key={era.delist_year}
              label={`${era.delist_year} (${era.names} names)`}
              value={era.with_prices_pct}
              max={100}
              display={`${era.with_prices_pct}%`}
              accent="bg-emerald-400"
            />
          ))}
        </div>
        {surv.missing_count > 0 ? (
          <p className="mt-4 text-xs text-zinc-500">
            {surv.missing_count} delisted names have no prices, e.g.{" "}
            <span className="font-mono text-zinc-400">
              {surv.missing_symbols_sample.slice(0, 8).join(", ")}
            </span>
          </p>
        ) : null}
      </section>

      {pub.by_decade ? (
        <section className="bg-zinc-900 border border-zinc-800 rounded p-4">
          <h2 className="text-sm text-zinc-100 mb-1">Filing-date integrity</h2>
          <p className="text-xs text-zinc-500 mb-4">
            Share of statement rows whose vendor acceptedDate was a placeholder (period end).
            Remediated with a 45-day fallback lag + per-row flag (ADR 0012).
          </p>
          <div className="space-y-3">
            {pub.by_decade.map((row) => (
              <CoverageBar
                key={row.decade}
                label={`${row.decade}s (${row.rows.toLocaleString()} rows)`}
                value={row.placeholder_pct}
                max={100}
                display={`${row.placeholder_pct}%`}
                accent="bg-red-400"
              />
            ))}
          </div>
        </section>
      ) : null}
    </div>
  );
}

function DatasetsSection({ snapshot }: { snapshot: DataHealthSnapshot }) {
  const universeTotal = snapshot.universe.total;
  const datasetEntries = Object.entries(snapshot.datasets).filter(
    ([, stats]) => stats.universe_with_data != null,
  );
  return (
    <div className="space-y-6">
      <section className="bg-zinc-900 border border-zinc-800 rounded p-4">
        <h2 className="text-sm text-zinc-100 mb-1">Dataset coverage vs universe</h2>
        <p className="text-xs text-zinc-500 mb-4">
          Symbols (of {universeTotal.toLocaleString()}) with non-empty data per dataset. Short bars
          are not necessarily failures — event/vendor datasets were fetched for the 774-name
          universe first; the expanded fetch backfills them.
        </p>
        <div className="space-y-3">
          {datasetEntries.map(([name, stats]) => (
            <CoverageBar
              key={name}
              label={name}
              value={stats.universe_with_data ?? 0}
              max={universeTotal}
              display={`${formatCount(stats.universe_with_data)} · ${formatCount(
                stats.universe_empty_file,
              )} empty · ${formatCount(stats.universe_missing_file)} missing`}
            />
          ))}
        </div>
      </section>

      <section className="bg-zinc-900 border border-zinc-800 rounded p-4">
        <h2 className="text-sm text-zinc-100 mb-1">Panel artifacts</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-zinc-500 border-b border-zinc-800">
                <th className="text-left py-1.5 pr-3 font-normal">File</th>
                <th className="text-right py-1.5 px-3 font-normal">Rows</th>
                <th className="text-right py-1.5 px-3 font-normal">Symbols</th>
                <th className="text-right py-1.5 px-3 font-normal">Size</th>
                <th className="text-right py-1.5 pl-3 font-normal">Modified</th>
              </tr>
            </thead>
            <tbody className="font-mono tabular-nums">
              {snapshot.panels.map((panel) => (
                <tr key={panel.file} className="border-b border-zinc-800/60">
                  <td className="py-1.5 pr-3 text-zinc-300 font-sans">{panel.file}</td>
                  <td className="py-1.5 px-3 text-right text-zinc-100">
                    {panel.status === "missing" ? (
                      <span className="text-red-400 font-sans">missing</span>
                    ) : (
                      formatCount(panel.rows)
                    )}
                  </td>
                  <td className="py-1.5 px-3 text-right text-zinc-400">
                    {formatCount(panel.symbols)}
                  </td>
                  <td className="py-1.5 px-3 text-right text-zinc-400">
                    {panel.size_mb != null ? `${panel.size_mb} MB` : "—"}
                  </td>
                  <td className="py-1.5 pl-3 text-right text-zinc-500">{panel.modified ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {Object.keys(snapshot.intraday).length > 0 && !("status" in snapshot.intraday) ? (
        <section className="bg-zinc-900 border border-zinc-800 rounded p-4">
          <h2 className="text-sm text-zinc-100 mb-2">Intraday</h2>
          {Object.entries(snapshot.intraday).map(([interval, stats]) => (
            <p key={interval} className="text-xs text-zinc-400">
              <span className="text-zinc-100 font-mono">{interval}</span>: {stats.symbols} symbols,{" "}
              {stats.symbol_year_files.toLocaleString()} symbol-years ({stats.empty_symbol_years}{" "}
              empty), {stats.year_range}.{" "}
              <span className="text-amber-400">
                Bars are unadjusted for splits — see Known Flaws.
              </span>
            </p>
          ))}
        </section>
      ) : null}
    </div>
  );
}

function FlawsSection({
  flaws,
  registryCaveats,
}: {
  flaws: DataFlaw[];
  registryCaveats: RegistryCaveat[];
}) {
  return (
    <div className="space-y-3">
      <p className="text-[11px] text-zinc-500 px-1 leading-relaxed">
        Two sources, one place. <span className="text-zinc-300">Measured flaws</span> are
        recomputed from disk on every audit (core/data/health.py). <span className="text-zinc-300">
        Registry caveats</span> are the standing disclosures shared with every research surface
        (core/research/caveats.py) — the same entries appear on the sector and PEAD pages.
      </p>
      <h3 className="text-xs uppercase tracking-wider text-zinc-500 pt-1">Measured flaws</h3>
      {flaws.map((flaw) => (
        <section key={flaw.id} className="bg-zinc-900 border border-zinc-800 rounded p-4">
          <div className="flex items-center gap-2 mb-1.5">
            <span
              className={cn(
                "px-1.5 py-0.5 rounded text-[10px] uppercase tracking-wide",
                SEVERITY_STYLES[flaw.severity],
              )}
            >
              {flaw.severity}
            </span>
            <h3 className="text-sm text-zinc-100">{flaw.title}</h3>
          </div>
          <p className="text-xs text-zinc-400 leading-relaxed">{flaw.detail}</p>
        </section>
      ))}
      <h3 className="text-xs uppercase tracking-wider text-zinc-500 pt-3">
        Registry caveats (shared across surfaces)
      </h3>
      {registryCaveats.map((caveat) => (
        <section key={caveat.id} className="bg-zinc-900 border border-zinc-800 rounded p-4">
          <div className="flex items-center gap-2 mb-1.5">
            <span
              className={cn(
                "px-1.5 py-0.5 rounded text-[10px] uppercase tracking-wide",
                SEVERITY_STYLES[caveat.severity],
              )}
            >
              {caveat.severity}
            </span>
            <span className="px-1.5 py-0.5 rounded text-[10px] uppercase text-zinc-400 bg-zinc-800/60">
              {caveat.kind}
            </span>
            <h3 className="text-sm text-zinc-100">{caveat.title}</h3>
          </div>
          <p className="text-xs text-zinc-400 leading-relaxed">{caveat.detail}</p>
          {caveat.remediation ? (
            <p className="text-[11px] text-blue-400/80 mt-1.5">Fix: {caveat.remediation}</p>
          ) : null}
        </section>
      ))}
    </div>
  );
}

function SymbolStatRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-3 text-xs py-1 border-b border-zinc-800/60">
      <span className="text-zinc-500">{label}</span>
      <span className="font-mono tabular-nums text-zinc-200 text-right">{value}</span>
    </div>
  );
}

function SymbolSection() {
  const [input, setInput] = useState("AAPL");
  const [symbol, setSymbol] = useState("AAPL");
  const { data, isLoading, error } = useQuery<SymbolDetail>({
    queryKey: ["symbol-data-detail", symbol],
    queryFn: () => api.getSymbolDataDetail(symbol),
    staleTime: 5 * 60 * 1000,
    retry: false,
  });

  const universe = data?.universe;
  return (
    <div className="space-y-4">
      <form
        className="flex gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          setSymbol(input.trim().toUpperCase());
        }}
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Symbol (e.g. AAPL, or a delisted name like AET)"
          className="flex-1 bg-zinc-900 border border-zinc-800 rounded px-3 py-1.5 text-sm text-zinc-100 font-mono placeholder:text-zinc-600 focus:outline-none focus:border-zinc-600"
        />
        <button
          type="submit"
          className="px-3 py-1.5 text-xs bg-zinc-800 text-zinc-200 rounded hover:bg-zinc-700"
        >
          Inspect
        </button>
      </form>

      {isLoading ? <p className="text-xs text-zinc-500">Loading {symbol}…</p> : null}
      {error ? <p className="text-xs text-red-400">No data held for {symbol}.</p> : null}

      {data ? (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <section className="bg-zinc-900 border border-zinc-800 rounded p-4">
            <h3 className="text-sm text-zinc-100 mb-2">
              {data.symbol}
              {universe?.company_name ? (
                <span className="text-zinc-500 font-normal"> — {universe.company_name}</span>
              ) : null}
              {universe?.is_delisted ? (
                <span className="ml-2 px-1.5 py-0.5 rounded text-[10px] uppercase text-red-400 bg-red-950/40">
                  delisted
                </span>
              ) : null}
            </h3>
            <SymbolStatRow label="Exchange / sector" value={`${universe?.exchange ?? "—"} · ${universe?.sector ?? "—"}`} />
            <SymbolStatRow
              label="Listed span"
              value={`${universe?.ipo_date ?? "?"} → ${universe?.delisted_date ?? "present"}`}
            />
            <SymbolStatRow
              label="Market cap (today)"
              value={
                universe?.market_cap != null
                  ? `$${(universe.market_cap / 1e9).toFixed(1)}B`
                  : "—"
              }
            />
            <SymbolStatRow
              label="Daily bars"
              value={
                data.prices
                  ? `${formatCount(data.prices.rows)} (${data.prices.first} → ${data.prices.last})${data.prices.has_ohlc ? " · OHLCV" : " · close only"}`
                  : "none"
              }
            />
            <SymbolStatRow label="Market-cap rows" value={formatCount(data.market_caps_rows)} />
            {Object.entries(data.statements).map(([name, info]) => (
              <SymbolStatRow
                key={name}
                label={name.replace("_", " ")}
                value={
                  info == null
                    ? "no file"
                    : info.quarters === 0
                      ? "empty"
                      : `${info.quarters}q (${info.first_period} → ${info.last_period}), ${info.placeholder_filing_dates_pct}% imputed dates`
                }
              />
            ))}
            {Object.entries(data.intraday).map(([interval, stats]) => (
              <SymbolStatRow
                key={interval}
                label={`intraday ${interval}`}
                value={`${formatCount(stats.rows)} bars, ${stats.years.length} years`}
              />
            ))}
          </section>

          <section className="bg-zinc-900 border border-zinc-800 rounded p-4">
            <h3 className="text-sm text-zinc-100 mb-2">Per-symbol datasets</h3>
            <div className="space-y-1">
              {Object.entries(data.datasets).map(([name, rows]) => (
                <div key={name} className="flex items-center gap-2 text-xs py-0.5">
                  <span
                    className={cn(
                      "inline-block w-2 h-2 rounded-full shrink-0",
                      rows == null ? "bg-zinc-700" : rows > 0 ? "bg-emerald-400" : "bg-zinc-500",
                    )}
                  />
                  <span className="text-zinc-400 flex-1">{name}</span>
                  <span className="font-mono tabular-nums text-zinc-300">
                    {rows == null ? "not fetched" : rows === 0 ? "empty" : `${formatCount(rows)} rows`}
                  </span>
                </div>
              ))}
            </div>
          </section>
        </div>
      ) : null}
    </div>
  );
}

export default function DataHealth() {
  const [section, setSection] = useState<Section>("universe");
  const { data, isLoading, error } = useQuery({
    queryKey: ["data-health"],
    queryFn: api.getDataHealth,
    staleTime: 5 * 60 * 1000,
  });

  return (
    <AppLayout
      left={
        <LeftSidebar>
          <div className="space-y-1">
            {(Object.keys(SECTION_LABELS) as Section[]).map((key) => (
              <button
                key={key}
                onClick={() => setSection(key)}
                className={cn(
                  "w-full text-left px-2.5 py-1.5 text-xs rounded transition-colors",
                  section === key
                    ? "bg-zinc-800 text-zinc-100"
                    : "text-zinc-500 hover:text-zinc-300 hover:bg-zinc-900",
                )}
              >
                {SECTION_LABELS[key]}
                {key === "flaws" && data ? (
                  <span className="ml-1.5 text-[10px] text-red-400">
                    {data.flaws.filter((f) => f.severity === "high").length} high
                  </span>
                ) : null}
              </button>
            ))}
          </div>
          {data ? (
            <p className="mt-4 px-2.5 text-[10px] text-zinc-600">
              Audit: {data.generated_at}
              <br />
              Rerun: scripts/audit_data_health.py
            </p>
          ) : null}
        </LeftSidebar>
      }
    >
      <div className="p-4 max-w-4xl">
        <h1 className="text-base text-zinc-100 mb-1">Data Health</h1>
        <p className="text-xs text-zinc-500 mb-5">
          What we hold, how complete it is, and every known flaw — recomputed from disk by the
          audit script, not asserted. Prose caveats live in docs/DATA_HEALTH.md.
        </p>
        {isLoading ? <p className="text-xs text-zinc-500">Loading audit snapshot…</p> : null}
        {error ? (
          <p className="text-xs text-red-400">
            No audit snapshot. Run scripts/audit_data_health.py, then reload.
          </p>
        ) : null}
        {data ? (
          <>
            {section === "universe" ? <UniverseSection snapshot={data} /> : null}
            {section === "datasets" ? <DatasetsSection snapshot={data} /> : null}
            {section === "flaws" ? (
              <FlawsSection flaws={data.flaws} registryCaveats={data.registry_caveats ?? []} />
            ) : null}
            {section === "symbol" ? <SymbolSection /> : null}
          </>
        ) : null}
      </div>
    </AppLayout>
  );
}
