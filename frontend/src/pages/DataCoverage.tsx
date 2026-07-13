import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import KPICard from "@/components/cards/KPICard.tsx";
import AppLayout from "@/components/layout/AppLayout.tsx";
import LeftSidebar from "@/components/layout/LeftSidebar.tsx";
import { api } from "@/lib/api.ts";
import type { DatasetInfo, QuarantineEntry, SP500CsvInfo, YearCoverage } from "@/lib/types.ts";
import { cn } from "@/lib/utils.ts";

type Tab = "datasets" | "coverage" | "quarantine";

const STATUS_STYLES: Record<QuarantineEntry["status"], string> = {
  quarantined: "text-red-400 bg-red-950/40",
  flagged: "text-amber-400 bg-amber-950/30",
  cleared: "text-emerald-400 bg-emerald-950/30",
};

function formatCsvAge(csv: SP500CsvInfo): string {
  if (csv.age_days < 1) return "<1d";
  if (csv.age_days < 45) return `${Math.round(csv.age_days)}d`;
  return `${(csv.age_days / 30).toFixed(1)}mo`;
}

export default function DataCoverage() {
  const [activeTab, setActiveTab] = useState<Tab>("datasets");
  const queryClient = useQueryClient();
  const { data, isLoading, error } = useQuery({
    queryKey: ["data-coverage"],
    queryFn: api.getDataCoverage,
    staleTime: 5 * 60 * 1000,
  });

  const reviewMut = useMutation({
    mutationFn: api.reviewQuarantine,
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["data-coverage"] });
    },
  });

  const csv = data?.sp500_csv ?? null;
  const csvStale = csv != null && csv.age_days >= 90;

  return (
    <AppLayout
      left={
        <LeftSidebar>
          <div className="space-y-1">
            {(["datasets", "coverage", "quarantine"] as Tab[]).map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={cn(
                  "w-full text-left px-2.5 py-1.5 text-xs rounded transition-colors capitalize",
                  activeTab === tab
                    ? "bg-zinc-800 text-zinc-100"
                    : "text-zinc-500 hover:text-zinc-300 hover:bg-zinc-900",
                )}
              >
                {tab === "coverage" ? "Universe Coverage" : tab}
              </button>
            ))}
          </div>

          <div className="mt-6 text-[11px] text-zinc-500 leading-relaxed space-y-2">
            <p>
              Primary vendor: <span className="text-zinc-300">FMP Premium</span> (dividend-adjusted
              EOD). Raw per-symbol files are immutable; every derived panel is rebuildable from
              them.
            </p>
            <p>
              <span className="text-red-400">Quarantined</span> symbols are excluded from all
              loaded data. <span className="text-amber-400">Flagged</span> symbols stay in but
              await review. Overrides persist to parquet; restart the API for exclusions to reload.
            </p>
            {csv && (
              <p>
                S&amp;P CSV:{" "}
                <span className="text-zinc-300 font-mono">{csv.filename}</span>
                <br />
                Age {formatCsvAge(csv)}
                {csv.last_membership_date
                  ? ` · last row ${csv.last_membership_date}`
                  : null}
              </p>
            )}
            {data?.edgar_note && <p>{data.edgar_note}</p>}
          </div>
        </LeftSidebar>
      }
    >
      <div className="p-4 space-y-4 overflow-y-auto h-full">
        <h1 className="text-sm font-semibold text-zinc-200">Data Coverage &amp; Quality</h1>

        {isLoading && <p className="text-xs text-zinc-500">Loading inventory…</p>}
        {error && (
          <p className="text-xs text-red-400">Failed to load: {(error as Error).message}</p>
        )}

        {data && (
          <>
            <div className="grid grid-cols-5 gap-3">
              <KPICard label="Symbols Loaded" value={String(data.total_symbols_loaded)} />
              <KPICard label="Datasets" value={String(data.datasets.length)} />
              <KPICard
                label="Quarantined"
                value={String(data.quarantined_symbol_count)}
                accent={data.quarantined_symbol_count > 0 ? "negative" : "positive"}
              />
              <KPICard label="Flagged for Review" value={String(data.flagged_symbol_count)} />
              <KPICard
                label="S&P CSV Age"
                value={csv ? formatCsvAge(csv) : "—"}
                accent={csvStale ? "negative" : "positive"}
              />
            </div>

            {activeTab === "datasets" && <DatasetsTable datasets={data.datasets} />}
            {activeTab === "coverage" && (
              <CoverageTable
                coverage={data.coverage_by_year}
                note={data.survivorship_note}
                sp500Csv={csv}
              />
            )}
            {activeTab === "quarantine" && (
              <QuarantineTable
                entries={data.quarantine}
                onReview={(payload) => reviewMut.mutate(payload)}
                reviewing={reviewMut.isPending}
                error={reviewMut.isError ? (reviewMut.error as Error).message : null}
              />
            )}
          </>
        )}
      </div>
    </AppLayout>
  );
}

function DatasetsTable({ datasets }: { datasets: DatasetInfo[] }) {
  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="text-zinc-500 border-b border-zinc-800">
            <Th>Dataset</Th>
            <Th>Source</Th>
            <Th>Layer</Th>
            <Th className="text-right">Rows</Th>
            <Th className="text-right">Cols</Th>
            <Th>Range</Th>
            <Th className="text-right">Size (MB)</Th>
            <Th>Description</Th>
          </tr>
        </thead>
        <tbody>
          {datasets.map((d) => (
            <tr key={d.name} className="border-b border-zinc-800/50 hover:bg-zinc-800/30">
              <Td className="font-mono text-zinc-200">{d.name}</Td>
              <Td className="text-zinc-400">{d.source}</Td>
              <Td>
                <span
                  className={cn(
                    "px-1.5 py-0.5 rounded text-[10px] font-mono",
                    d.layer === "raw" ? "text-blue-400 bg-blue-950/40" : "text-zinc-400 bg-zinc-800",
                  )}
                >
                  {d.layer}
                </span>
              </Td>
              <Td className="text-right font-mono tabular-nums text-zinc-300">
                {d.rows.toLocaleString()}
              </Td>
              <Td className="text-right font-mono tabular-nums text-zinc-300">
                {d.columns > 0 ? d.columns.toLocaleString() : "—"}
              </Td>
              <Td className="font-mono text-zinc-400 whitespace-nowrap">
                {d.first_date ? `${d.first_date} → ${d.last_date}` : "—"}
              </Td>
              <Td className="text-right font-mono tabular-nums text-zinc-300">{d.size_mb}</Td>
              <Td className="text-zinc-500 max-w-md">{d.description}</Td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function CoverageTable({
  coverage,
  note,
  sp500Csv,
}: {
  coverage: YearCoverage[];
  note?: string;
  sp500Csv: SP500CsvInfo | null;
}) {
  const recentFirst = [...coverage].sort((a, b) => b.year - a.year);
  return (
    <div className="space-y-3">
      {sp500Csv && (
        <p
          className={cn(
            "text-xs px-3 py-2 leading-relaxed border font-mono",
            sp500Csv.age_days >= 90
              ? "text-amber-400/90 bg-amber-950/20 border-amber-900/40"
              : "text-zinc-400 bg-zinc-900 border-zinc-800",
          )}
        >
          Membership CSV: {sp500Csv.filename} · file age {formatCsvAge(sp500Csv)} (mtime{" "}
          {sp500Csv.file_mtime_utc.slice(0, 10)})
          {sp500Csv.last_membership_date
            ? ` · last snapshot ${sp500Csv.last_membership_date}`
            : ""}
          {sp500Csv.n_snapshots != null ? ` · ${sp500Csv.n_snapshots} rows` : ""}
        </p>
      )}
      {note && (
        <p className="text-xs text-amber-400/90 bg-amber-950/20 border border-amber-900/40 px-3 py-2 leading-relaxed">
          {note}
        </p>
      )}
      <div className="bg-zinc-900 border border-zinc-800 rounded overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="text-zinc-500 border-b border-zinc-800">
            <Th>Year</Th>
            <Th className="text-right">Symbols With Data</Th>
            <Th className="text-right">S&amp;P 500 Members</Th>
            <Th className="text-right">Membership Covered</Th>
            <Th>Coverage</Th>
          </tr>
        </thead>
        <tbody>
          {recentFirst.map((c) => (
            <tr key={c.year} className="border-b border-zinc-800/50 hover:bg-zinc-800/30">
              <Td className="font-mono text-zinc-200">{c.year}</Td>
              <Td className="text-right font-mono tabular-nums text-zinc-300">
                {c.symbols_with_data}
              </Td>
              <Td className="text-right font-mono tabular-nums text-zinc-300">
                {c.sp500_members || "—"}
              </Td>
              <Td
                className={cn(
                  "text-right font-mono tabular-nums",
                  c.coverage_pct >= 95
                    ? "text-emerald-400"
                    : c.coverage_pct >= 80
                      ? "text-amber-400"
                      : "text-red-400",
                )}
              >
                {c.sp500_members ? `${c.coverage_pct}%` : "—"}
              </Td>
              <Td className="w-40">
                <div className="h-1.5 bg-zinc-800 rounded overflow-hidden">
                  <div
                    className={cn(
                      "h-full",
                      c.coverage_pct >= 95
                        ? "bg-emerald-500/70"
                        : c.coverage_pct >= 80
                          ? "bg-amber-500/70"
                          : "bg-red-500/70",
                    )}
                    style={{ width: `${Math.min(c.coverage_pct, 100)}%` }}
                  />
                </div>
              </Td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
    </div>
  );
}

function QuarantineTable({
  entries,
  onReview,
  reviewing,
  error,
}: {
  entries: QuarantineEntry[];
  onReview: (payload: {
    symbol: string;
    check: string;
    status: "cleared" | "quarantined" | "flagged";
    note: string;
  }) => void;
  reviewing: boolean;
  error: string | null;
}) {
  const ordered = [...entries].sort((a, b) =>
    a.status === b.status ? a.symbol.localeCompare(b.symbol) : a.status.localeCompare(b.status),
  );
  const [drafts, setDrafts] = useState<Record<string, { status: QuarantineEntry["status"]; note: string }>>(
    {},
  );

  return (
    <div className="space-y-2">
      {error && <p className="text-xs text-red-400">{error}</p>}
      <p className="text-[11px] text-zinc-500">
        Manual override: set status + note, then Save. Quarantined symbols drop from loaded panels
        after an API restart.
      </p>
      <div className="bg-zinc-900 border border-zinc-800 rounded overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="text-zinc-500 border-b border-zinc-800">
              <Th>Symbol</Th>
              <Th>Status</Th>
              <Th>Check</Th>
              <Th className="text-right">Value</Th>
              <Th>Detail</Th>
              <Th>Review Note</Th>
              <Th>Override</Th>
            </tr>
          </thead>
          <tbody>
            {ordered.map((q, i) => {
              const key = `${q.symbol}::${q.check}`;
              const draft = drafts[key] ?? { status: q.status, note: q.review_note || "" };
              return (
                <tr
                  key={`${q.symbol}-${q.check}-${i}`}
                  className="border-b border-zinc-800/50 hover:bg-zinc-800/30"
                >
                  <Td className="font-mono text-zinc-200">{q.symbol}</Td>
                  <Td>
                    <span
                      className={cn(
                        "px-1.5 py-0.5 rounded text-[10px] font-mono",
                        STATUS_STYLES[q.status],
                      )}
                    >
                      {q.status}
                    </span>
                  </Td>
                  <Td className="font-mono text-zinc-400">{q.check}</Td>
                  <Td className="text-right font-mono tabular-nums text-zinc-300">{q.value}</Td>
                  <Td className="text-zinc-500">{q.detail}</Td>
                  <Td className="text-zinc-500 italic">{q.review_note || "—"}</Td>
                  <Td>
                    <div className="flex flex-col gap-1 min-w-[160px]">
                      <select
                        value={draft.status}
                        onChange={(e) =>
                          setDrafts((prev) => ({
                            ...prev,
                            [key]: {
                              ...draft,
                              status: e.target.value as QuarantineEntry["status"],
                            },
                          }))
                        }
                        className="bg-zinc-950 border border-zinc-800 rounded px-1.5 py-1 text-[10px] font-mono text-zinc-300"
                      >
                        <option value="flagged">flagged</option>
                        <option value="cleared">cleared</option>
                        <option value="quarantined">quarantined</option>
                      </select>
                      <input
                        value={draft.note}
                        onChange={(e) =>
                          setDrafts((prev) => ({
                            ...prev,
                            [key]: { ...draft, note: e.target.value },
                          }))
                        }
                        placeholder="note"
                        className="bg-zinc-950 border border-zinc-800 rounded px-1.5 py-1 text-[10px] text-zinc-300"
                      />
                      <button
                        type="button"
                        disabled={reviewing}
                        onClick={() =>
                          onReview({
                            symbol: q.symbol,
                            check: q.check,
                            status: draft.status,
                            note: draft.note,
                          })
                        }
                        className="text-[10px] uppercase tracking-wider py-1 rounded bg-zinc-800 text-zinc-300 hover:bg-zinc-700 disabled:opacity-50"
                      >
                        Save
                      </button>
                    </div>
                  </Td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Th({ children, className }: { children: React.ReactNode; className?: string }) {
  return <th className={cn("px-3 py-2 text-left font-medium", className)}>{children}</th>;
}

function Td({ children, className }: { children: React.ReactNode; className?: string }) {
  return <td className={cn("px-3 py-1.5", className)}>{children}</td>;
}
