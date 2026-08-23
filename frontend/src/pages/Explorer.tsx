import { useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import Plot from "react-plotly.js";

import AppLayout from "@/components/layout/AppLayout.tsx";
import { api } from "@/lib/api.ts";
import { cn } from "@/lib/utils.ts";
import type {
  CompanyProfile,
  ExplorerDataset,
  ExplorerFilterSpec,
  ExplorerQueryResult,
} from "@/lib/types.ts";

type Tab = "company" | "screen" | "query";

const OPERATORS = [">", ">=", "<", "<=", "=", "!="] as const;

const PLOT_THEME = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "transparent",
  font: { color: "#a1a1aa", size: 11, family: "JetBrains Mono, monospace" },
  xaxis: { gridcolor: "#27272a", zerolinecolor: "#3f3f46" },
  yaxis: { gridcolor: "#27272a", zerolinecolor: "#3f3f46" },
  margin: { l: 56, r: 16, t: 24, b: 40 },
} as const;

export default function Explorer() {
  const [tab, setTab] = useState<Tab>("company");

  return (
    <AppLayout>
      <div className="max-w-6xl mx-auto py-4 px-2 space-y-4">
        <header>
          <h1 className="text-lg font-semibold text-zinc-200 tracking-tight">Data Explorer</h1>
          <p className="text-xs text-zinc-500 mt-1">
            Look up a company, screen the universe, or query the panels directly. Every
            screen shows the SQL it generated — the query is the documentation.
          </p>
        </header>

        <nav className="flex gap-1 border-b border-zinc-800">
          {(
            [
              ["company", "Company"],
              ["screen", "Screener"],
              ["query", "SQL"],
            ] as const
          ).map(([id, label]) => (
            <button
              key={id}
              onClick={() => setTab(id)}
              className={cn(
                "px-3 py-1.5 text-xs border-b-2 -mb-px transition-colors cursor-pointer",
                tab === id
                  ? "border-blue-400 text-zinc-100"
                  : "border-transparent text-zinc-500 hover:text-zinc-300",
              )}
            >
              {label}
            </button>
          ))}
        </nav>

        {tab === "company" && <CompanyTab />}
        {tab === "screen" && <ScreenTab />}
        {tab === "query" && <QueryTab />}
      </div>
    </AppLayout>
  );
}

/* ── Company lookup ───────────────────────────────────────── */

function CompanyTab() {
  const [term, setTerm] = useState("");
  const [symbol, setSymbol] = useState<string | null>(null);

  const search = useQuery({
    queryKey: ["explorer-search", term],
    queryFn: () => api.searchCompanies(term),
    enabled: term.trim().length >= 2 && !symbol,
  });

  const profile = useQuery({
    queryKey: ["explorer-company", symbol],
    queryFn: () => api.getCompanyProfile(symbol as string),
    enabled: !!symbol,
  });

  return (
    <div className="space-y-4">
      <div className="flex gap-2">
        <input
          value={term}
          onChange={(e) => {
            setTerm(e.target.value);
            setSymbol(null);
          }}
          placeholder="Ticker or company name — try MCD or 'mcdonald'"
          className="flex-1 bg-zinc-900 border border-zinc-800 rounded px-3 py-1.5 text-xs text-zinc-200 font-mono focus:outline-none focus:border-zinc-600"
        />
      </div>

      {!symbol && search.data && search.data.results.length > 0 && (
        <div className="border border-zinc-800 rounded divide-y divide-zinc-800">
          {search.data.results.slice(0, 8).map((r) => (
            <button
              key={r.symbol}
              onClick={() => setSymbol(r.symbol)}
              className="w-full text-left px-3 py-1.5 hover:bg-zinc-900 flex gap-3 items-baseline cursor-pointer"
            >
              <span className="font-mono text-xs text-zinc-200 w-16 shrink-0">{r.symbol}</span>
              <span className="text-xs text-zinc-400 flex-1 truncate">{r.company_name}</span>
              <span className="text-[10px] text-zinc-600 shrink-0">{r.sector ?? "—"}</span>
              {r.is_delisted && (
                <span className="text-[10px] text-red-400 shrink-0">delisted</span>
              )}
            </button>
          ))}
        </div>
      )}

      {profile.isLoading && <div className="text-xs text-zinc-500">Loading…</div>}
      {profile.data && <CompanyDetail profile={profile.data} />}
    </div>
  );
}

function CompanyDetail({ profile }: { profile: CompanyProfile }) {
  const identity = profile.identity as Record<string, string | number | boolean | null>;
  const ids = profile.identifiers as Record<string, string | null> | null;

  const families = useMemo(() => {
    const grouped = new Map<string, typeof profile.factors>();
    for (const f of profile.factors) {
      if (!grouped.has(f.family)) grouped.set(f.family, []);
      grouped.get(f.family)!.push(f);
    }
    return [...grouped.entries()];
  }, [profile]);

  return (
    <div className="space-y-4">
      <section className="border border-zinc-800 rounded p-3">
        <div className="flex items-baseline gap-3 mb-2">
          <h2 className="text-sm text-zinc-100 font-mono">{profile.symbol}</h2>
          <span className="text-xs text-zinc-400">{String(identity.company_name ?? "")}</span>
          {ids?.qid && (
            <span
              className="ml-auto text-[10px] text-zinc-500 font-mono"
              title="Permanent internal security id — survives ticker changes and vendor switches"
            >
              qid {ids.qid}
              {ids.issuer_id ? ` · issuer ${ids.issuer_id}` : ""}
            </span>
          )}
        </div>
        <dl className="grid grid-cols-2 sm:grid-cols-4 gap-x-4 gap-y-1.5 text-xs">
          {(
            [
              ["Exchange", identity.exchange],
              ["Sector", identity.sector],
              ["Industry", identity.industry],
              ["Market cap", fmtBig(identity.market_cap as number | null)],
              ["Listed", identity.is_delisted ? "Delisted" : "Live"],
              ["IPO", identity.ipo_date ?? "—"],
              ["Delisted", identity.delisted_date ?? "—"],
              [
                "S&P 500",
                profile.index_membership.length
                  ? profile.index_membership
                      .map((m) => `${m.valid_from} → ${m.valid_to ?? "now"}`)
                      .join(", ")
                  : "never",
              ],
            ] as const
          ).map(([label, value]) => (
            <div key={label}>
              <dt className="text-[10px] uppercase tracking-wider text-zinc-600">{label}</dt>
              <dd className="text-zinc-300 font-mono tabular-nums truncate">
                {value == null || value === "" ? "—" : String(value)}
              </dd>
            </div>
          ))}
        </dl>
      </section>

      {profile.prices.length > 0 && (
        <section className="border border-zinc-800 rounded p-3">
          <h3 className="text-[11px] uppercase tracking-wider text-zinc-500 mb-2">
            Adjusted close · {profile.prices.length} trading days
          </h3>
          <Plot
            data={[
              {
                x: profile.prices.map((p) => p.date),
                y: profile.prices.map((p) => p.adj_close),
                type: "scatter",
                mode: "lines",
                line: { color: "#60a5fa", width: 1.2 },
                hovertemplate: "%{x}<br>$%{y:.2f}<extra></extra>",
              },
            ]}
            layout={{ ...PLOT_THEME, height: 260, showlegend: false }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: "100%" }}
          />
        </section>
      )}

      <section className="border border-zinc-800 rounded p-3">
        <h3 className="text-[11px] uppercase tracking-wider text-zinc-500 mb-2">
          Latest factor values · {profile.factors.length} across {families.length} families
        </h3>
        <div className="grid gap-4 sm:grid-cols-2">
          {families.map(([family, values]) => (
            <div key={family}>
              <div className="text-[10px] font-mono text-zinc-600 mb-1">
                {family} · as of {values[0]?.as_of ?? "—"}
              </div>
              <table className="w-full text-xs">
                <tbody>
                  {values.map((f) => (
                    <tr key={f.factor} className="border-t border-zinc-900">
                      <td className="py-0.5 text-zinc-400 pr-2 truncate">{f.factor}</td>
                      <td className="py-0.5 text-right font-mono tabular-nums text-zinc-200">
                        {fmtNum(f.value)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

/* ── Screener ─────────────────────────────────────────────── */

function ScreenTab() {
  const datasets = useQuery({
    queryKey: ["explorer-datasets"],
    queryFn: api.getExplorerDatasets,
  });

  const [selected, setSelected] = useState<string[]>([]);
  const [filters, setFilters] = useState<ExplorerFilterSpec[]>([
    { column: "gross_profitability", operator: ">", value: 0.3 },
  ]);
  const [columns, setColumns] = useState<string[]>(["gross_profitability", "mom_12_1"]);
  const [orderBy, setOrderBy] = useState("mom_12_1");
  const [result, setResult] = useState<ExplorerQueryResult | null>(null);

  const run = useMutation({
    mutationFn: () =>
      api.runScreen({
        columns,
        filters,
        panels: selected.length ? selected : undefined,
        order_by: orderBy || undefined,
        limit: 200,
      }),
    onSuccess: setResult,
  });

  const allColumns = useMemo(
    () => (datasets.data?.datasets ?? []).flatMap((d) => d.columns),
    [datasets.data],
  );

  return (
    <div className="space-y-4">
      <section>
        <h3 className="text-[11px] uppercase tracking-wider text-zinc-500 mb-2">
          Panels — click to force a join; leave empty to infer from the columns used
        </h3>
        <div className="flex flex-wrap gap-1.5">
          {(datasets.data?.datasets ?? []).map((d: ExplorerDataset) => (
            <button
              key={d.name}
              title={`${d.description}\nGrain: ${d.grain}\n${d.n_columns} columns`}
              onClick={() =>
                setSelected((s) =>
                  s.includes(d.name) ? s.filter((x) => x !== d.name) : [...s, d.name],
                )
              }
              className={cn(
                "px-2 py-1 text-[11px] font-mono rounded border transition-colors cursor-pointer",
                selected.includes(d.name)
                  ? "border-blue-500/50 bg-blue-500/10 text-blue-200"
                  : "border-zinc-800 text-zinc-500 hover:text-zinc-300",
              )}
            >
              {d.name}
              <span className="ml-1.5 text-zinc-600">{d.n_columns}</span>
            </button>
          ))}
        </div>
      </section>

      <section className="space-y-2">
        <h3 className="text-[11px] uppercase tracking-wider text-zinc-500">Filters</h3>
        {filters.map((f, i) => (
          <div key={i} className="flex gap-2 items-center">
            <input
              list="explorer-columns"
              value={f.column}
              onChange={(e) =>
                setFilters((s) => s.map((x, j) => (j === i ? { ...x, column: e.target.value } : x)))
              }
              className="flex-1 bg-zinc-900 border border-zinc-800 rounded px-2 py-1 text-xs font-mono text-zinc-200 focus:outline-none focus:border-zinc-600"
            />
            <select
              value={f.operator}
              onChange={(e) =>
                setFilters((s) =>
                  s.map((x, j) => (j === i ? { ...x, operator: e.target.value } : x)),
                )
              }
              className="bg-zinc-900 border border-zinc-800 rounded px-2 py-1 text-xs font-mono text-zinc-200"
            >
              {OPERATORS.map((op) => (
                <option key={op}>{op}</option>
              ))}
            </select>
            <input
              value={String(f.value)}
              onChange={(e) =>
                setFilters((s) =>
                  s.map((x, j) =>
                    j === i ? { ...x, value: parseFloat(e.target.value) || e.target.value } : x,
                  ),
                )
              }
              className="w-28 bg-zinc-900 border border-zinc-800 rounded px-2 py-1 text-xs font-mono tabular-nums text-zinc-200 focus:outline-none focus:border-zinc-600"
            />
            <button
              onClick={() => setFilters((s) => s.filter((_, j) => j !== i))}
              className="text-zinc-600 hover:text-red-400 px-1 cursor-pointer"
            >
              ×
            </button>
          </div>
        ))}
        <datalist id="explorer-columns">
          {allColumns.map((c) => (
            <option key={c} value={c} />
          ))}
        </datalist>
        <button
          onClick={() =>
            setFilters((s) => [...s, { column: "", operator: ">", value: 0 }])
          }
          className="text-[11px] text-zinc-500 hover:text-zinc-300 cursor-pointer"
        >
          + add filter
        </button>
      </section>

      <section className="flex gap-2 items-end">
        <label className="flex-1">
          <span className="block text-[10px] uppercase tracking-wider text-zinc-600 mb-1">
            Columns shown (comma separated)
          </span>
          <input
            value={columns.join(",")}
            onChange={(e) => setColumns(e.target.value.split(",").map((s) => s.trim()).filter(Boolean))}
            className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1 text-xs font-mono text-zinc-200 focus:outline-none focus:border-zinc-600"
          />
        </label>
        <label>
          <span className="block text-[10px] uppercase tracking-wider text-zinc-600 mb-1">
            Sort by
          </span>
          <input
            value={orderBy}
            onChange={(e) => setOrderBy(e.target.value)}
            className="w-40 bg-zinc-900 border border-zinc-800 rounded px-2 py-1 text-xs font-mono text-zinc-200 focus:outline-none focus:border-zinc-600"
          />
        </label>
        <button
          onClick={() => run.mutate()}
          disabled={run.isPending}
          className="px-3 py-1 text-xs bg-zinc-800 hover:bg-zinc-700 text-zinc-100 rounded border border-zinc-700 disabled:opacity-50 cursor-pointer"
        >
          {run.isPending ? "Running…" : "Run screen"}
        </button>
      </section>

      {run.isError && (
        <div className="text-xs text-red-400 border border-red-500/30 rounded p-2">
          {(run.error as Error).message}
        </div>
      )}
      {result && <ResultPanel result={result} />}
    </div>
  );
}

/* ── Raw SQL ──────────────────────────────────────────────── */

function QueryTab() {
  const [sql, setSql] = useState(
    "SELECT sector, count(*) AS n\nFROM universe\nWHERE is_delisted = FALSE\nGROUP BY sector\nORDER BY n DESC",
  );
  const [result, setResult] = useState<ExplorerQueryResult | null>(null);

  const run = useMutation({
    mutationFn: () => api.runExplorerQuery(sql),
    onSuccess: setResult,
  });

  return (
    <div className="space-y-3">
      <p className="text-xs text-zinc-500">
        Read-only. SELECT and WITH only, one statement, capped at 5,000 rows. Paste a
        screen's generated SQL here to edit and re-run it.
      </p>
      <textarea
        value={sql}
        onChange={(e) => setSql(e.target.value)}
        rows={8}
        spellCheck={false}
        className="w-full bg-zinc-900 border border-zinc-800 rounded px-3 py-2 text-xs font-mono text-zinc-200 focus:outline-none focus:border-zinc-600"
      />
      <button
        onClick={() => run.mutate()}
        disabled={run.isPending}
        className="px-3 py-1 text-xs bg-zinc-800 hover:bg-zinc-700 text-zinc-100 rounded border border-zinc-700 disabled:opacity-50 cursor-pointer"
      >
        {run.isPending ? "Running…" : "Run query"}
      </button>
      {run.isError && (
        <div className="text-xs text-red-400 border border-red-500/30 rounded p-2 font-mono">
          {(run.error as Error).message}
        </div>
      )}
      {result && <ResultPanel result={result} />}
    </div>
  );
}

/* ── Shared result rendering ──────────────────────────────── */

/**
 * Picks a default chart from the shape of the result, then lets the user change it.
 *
 * The default is inferred rather than fixed because the right chart depends on
 * what came back: a date column means a time series, one label plus one number
 * means a bar chart, two numbers means a scatter.
 */
function ResultPanel({ result }: { result: ExplorerQueryResult }) {
  const numeric = useMemo(
    () =>
      result.columns.filter((c) =>
        result.rows.some((r) => typeof r[c] === "number" && Number.isFinite(r[c] as number)),
      ),
    [result],
  );
  const dateColumn = result.columns.find((c) => /date/i.test(c));
  const labelColumn = result.columns.find((c) => !numeric.includes(c) && c !== dateColumn);

  const defaultChart: "none" | "line" | "bar" | "scatter" | "histogram" =
    dateColumn && numeric.length >= 1 && result.rows.length > 5
      ? "line"
      : labelColumn && numeric.length === 1
        ? "bar"
        : numeric.length >= 2
          ? "scatter"
          : numeric.length === 1
            ? "histogram"
            : "none";

  const [chart, setChart] = useState(defaultChart);
  const [showSql, setShowSql] = useState(false);

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-3 text-[11px] text-zinc-500">
        <span className="font-mono">
          {result.row_count} row{result.row_count === 1 ? "" : "s"}
          {result.truncated && " (capped)"}
          {result.elapsed_ms != null && ` · ${result.elapsed_ms}ms`}
        </span>
        <button
          onClick={() => setShowSql((v) => !v)}
          className="text-zinc-500 hover:text-zinc-300 cursor-pointer"
        >
          {showSql ? "hide SQL" : "show SQL"}
        </button>
        <span className="ml-auto flex gap-1">
          {(["none", "line", "bar", "scatter", "histogram"] as const).map((c) => (
            <button
              key={c}
              onClick={() => setChart(c)}
              className={cn(
                "px-1.5 py-0.5 rounded cursor-pointer",
                chart === c ? "bg-zinc-800 text-zinc-200" : "text-zinc-600 hover:text-zinc-400",
              )}
            >
              {c}
            </button>
          ))}
        </span>
      </div>

      {showSql && (
        <pre className="text-[11px] font-mono text-zinc-400 bg-zinc-900 border border-zinc-800 rounded p-2 overflow-x-auto whitespace-pre">
          {result.sql}
        </pre>
      )}

      {chart !== "none" && numeric.length > 0 && (
        <div className="border border-zinc-800 rounded p-2">
          <Plot
            data={buildTrace(chart, result, numeric, dateColumn, labelColumn)}
            layout={{ ...PLOT_THEME, height: 280, showlegend: false }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: "100%" }}
          />
        </div>
      )}

      <div className="border border-zinc-800 rounded overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-zinc-800 bg-zinc-900/50">
              {result.columns.map((c) => (
                <th
                  key={c}
                  className="text-left px-2 py-1 text-[10px] uppercase tracking-wider text-zinc-500 font-normal whitespace-nowrap"
                >
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {result.rows.map((row, i) => (
              <tr key={i} className="border-b border-zinc-900 hover:bg-zinc-900/50">
                {result.columns.map((c) => (
                  <td
                    key={c}
                    className="px-2 py-0.5 font-mono tabular-nums text-zinc-300 whitespace-nowrap"
                  >
                    {typeof row[c] === "number" ? fmtNum(row[c] as number) : String(row[c] ?? "—")}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function buildTrace(
  chart: string,
  result: ExplorerQueryResult,
  numeric: string[],
  dateColumn?: string,
  labelColumn?: string,
): Plotly.Data[] {
  const rows = result.rows;
  if (chart === "line" && dateColumn) {
    return [
      {
        x: rows.map((r) => r[dateColumn] as string),
        y: rows.map((r) => r[numeric[0]] as number),
        type: "scatter",
        mode: "lines",
        line: { color: "#60a5fa", width: 1.2 },
      },
    ];
  }
  if (chart === "bar") {
    return [
      {
        x: rows.map((r) => String(r[labelColumn ?? result.columns[0]])),
        y: rows.map((r) => r[numeric[0]] as number),
        type: "bar",
        marker: { color: "#3b82f6" },
      },
    ];
  }
  if (chart === "scatter" && numeric.length >= 2) {
    return [
      {
        x: rows.map((r) => r[numeric[0]] as number),
        y: rows.map((r) => r[numeric[1]] as number),
        text: rows.map((r) => String(r[labelColumn ?? "symbol"] ?? "")),
        type: "scatter",
        mode: "markers",
        marker: { color: "#60a5fa", size: 5, opacity: 0.7 },
        hovertemplate: `%{text}<br>${numeric[0]}: %{x}<br>${numeric[1]}: %{y}<extra></extra>`,
      },
    ];
  }
  return [
    {
      x: rows.map((r) => r[numeric[0]] as number),
      type: "histogram",
      marker: { color: "#3b82f6" },
      nbinsx: 40,
    },
  ];
}

function fmtNum(value: number | null): string {
  if (value == null || !Number.isFinite(value)) return "—";
  const abs = Math.abs(value);
  if (abs >= 1e9) return `${(value / 1e9).toFixed(2)}B`;
  if (abs >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
  if (abs >= 1000) return value.toFixed(0);
  if (abs >= 1) return value.toFixed(3);
  return value.toFixed(4);
}

function fmtBig(value: number | null): string {
  if (value == null) return "—";
  if (value >= 1e12) return `$${(value / 1e12).toFixed(2)}T`;
  if (value >= 1e9) return `$${(value / 1e9).toFixed(1)}B`;
  if (value >= 1e6) return `$${(value / 1e6).toFixed(1)}M`;
  return `$${value.toFixed(0)}`;
}
