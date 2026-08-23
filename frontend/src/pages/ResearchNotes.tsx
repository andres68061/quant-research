import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import Plot from "react-plotly.js";

import AppLayout from "@/components/layout/AppLayout.tsx";
import { api } from "@/lib/api.ts";
import { cn } from "@/lib/utils.ts";
import type { ResearchNote, ResearchResultRow } from "@/lib/types.ts";

const VERDICT_TONE: Record<string, string> = {
  validated: "border-emerald-500/40 bg-emerald-500/10 text-emerald-300",
  interesting: "border-blue-500/40 bg-blue-500/10 text-blue-300",
  real_not_tradable: "border-amber-500/40 bg-amber-500/10 text-amber-300",
  no_edge: "border-zinc-700 bg-zinc-800/60 text-zinc-400",
};

/**
 * Research notes: the readable record of what each experiment found.
 *
 * Every note answers the same questions in the same order — what was asked, why
 * it might work, what was run, every variant's numbers, what the control showed,
 * what it means, what would change the conclusion, and how to re-run it. The
 * fixed shape is the point: it makes notes comparable, and it makes an
 * unfinished one obvious.
 */
export default function ResearchNotes() {
  const index = useQuery({ queryKey: ["research-notes"], queryFn: api.listResearchNotes });
  const [selected, setSelected] = useState<string | null>(null);

  const activeId = selected ?? index.data?.notes[0]?.id ?? null;
  const note = useQuery({
    queryKey: ["research-note", activeId],
    queryFn: () => api.getResearchNote(activeId as string),
    enabled: !!activeId,
  });

  return (
    <AppLayout>
      <div className="max-w-4xl mx-auto py-6 px-4 space-y-5">
        <header>
          <h1 className="text-lg font-semibold text-zinc-200 tracking-tight">Research Notes</h1>
          <p className="text-xs text-zinc-500 mt-1">
            One note per experiment, each with every variant tried — not just the
            best one — and the control it has to be read against. The figures here
            are asserted against the stored experiment output by the test suite.
          </p>
        </header>

        <div className="flex flex-wrap gap-2">
          {(index.data?.notes ?? []).map((n) => (
            <button
              key={n.id}
              onClick={() => setSelected(n.id)}
              className={cn(
                "text-left px-3 py-2 rounded border max-w-xs cursor-pointer transition-colors",
                activeId === n.id
                  ? "border-zinc-600 bg-zinc-900"
                  : "border-zinc-800 hover:border-zinc-700",
              )}
            >
              <div className="text-xs text-zinc-200 leading-snug">{n.title}</div>
              <div className="flex gap-2 items-center mt-1">
                <span
                  className={cn(
                    "text-[9px] px-1.5 py-0.5 rounded border",
                    VERDICT_TONE[n.verdict] ?? VERDICT_TONE.no_edge,
                  )}
                >
                  {n.verdict_label}
                </span>
                <span className="text-[10px] text-zinc-600 font-mono">{n.run_date}</span>
              </div>
            </button>
          ))}
        </div>

        {note.isLoading && <div className="text-xs text-zinc-500">Loading…</div>}
        {note.data && <NoteBody note={note.data} />}
      </div>
    </AppLayout>
  );
}

function NoteBody({ note }: { note: ResearchNote }) {
  return (
    <article className="space-y-5">
      <div
        className={cn(
          "border rounded p-3",
          VERDICT_TONE[note.verdict] ?? VERDICT_TONE.no_edge,
        )}
      >
        <div className="text-[10px] uppercase tracking-wider opacity-70">
          {note.verdict_label}
        </div>
        <p className="text-xs mt-1 leading-relaxed">{note.one_liner}</p>
      </div>

      <Section title="What was asked">{note.question}</Section>
      <Section title="Why it might work">
        {note.hypothesis}
        <span className="block text-[10px] text-zinc-600 mt-1.5 font-mono">
          {note.reference}
        </span>
      </Section>
      <Section title="What was run">{note.method}</Section>

      <section>
        <h2 className="text-sm font-medium text-zinc-300 border-b border-zinc-800 pb-1 mb-3">
          Every variant tried
        </h2>
        <ResultsTable rows={note.results} />
        <SharpeChart rows={note.results} />
      </section>

      <Section title="Reading the control" tone="amber">
        {note.control_reading}
      </Section>

      {note.decade_table.length > 0 && (
        <section>
          <h2 className="text-sm font-medium text-zinc-300 border-b border-zinc-800 pb-1 mb-3">
            Sharpe by decade
          </h2>
          <p className="text-[11px] text-zinc-500 mb-2">
            Decay is the fingerprint of an anomaly being competed away; stability is what a
            structural effect looks like.
          </p>
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-zinc-800">
                {["Variant", "2000s", "2010s", "2020s"].map((h) => (
                  <th
                    key={h}
                    className="text-left px-2 py-1 text-[10px] uppercase tracking-wider text-zinc-500 font-normal"
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {note.decade_table.map((row) => (
                <tr key={row[0]} className="border-b border-zinc-900">
                  <td className="px-2 py-1 text-zinc-300 font-mono text-[11px]">{row[0]}</td>
                  {row.slice(1).map((cell, i) => (
                    <td
                      key={i}
                      className={cn(
                        "px-2 py-1 font-mono tabular-nums",
                        Number(cell) > 0 ? "text-emerald-400" : "text-red-400",
                      )}
                    >
                      {cell}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      <Section title="What it means">{note.what_it_means}</Section>

      <section>
        <h2 className="text-sm font-medium text-zinc-300 border-b border-zinc-800 pb-1 mb-2">
          What would change this
        </h2>
        <ul className="space-y-1.5">
          {note.caveats.map((c) => (
            <li key={c} className="text-[11px] text-zinc-400 leading-relaxed flex gap-2">
              <span className="text-zinc-600 shrink-0">·</span>
              <span>{c}</span>
            </li>
          ))}
        </ul>
      </section>

      <section className="border-t border-zinc-800 pt-3 space-y-2">
        <div>
          <div className="text-[10px] uppercase tracking-wider text-zinc-600 mb-1">Reproduce</div>
          <code className="block text-[11px] font-mono text-zinc-400 bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5 overflow-x-auto">
            {note.reproduce}
          </code>
        </div>
        {note.logged_in && (
          <div className="text-[10px] text-zinc-600">
            Logged in <span className="font-mono">{note.logged_in}</span>
          </div>
        )}
        {note.glossary_terms.length > 0 && (
          <div className="flex flex-wrap gap-1.5 items-baseline">
            <span className="text-[10px] text-zinc-600">Terms:</span>
            {note.glossary_terms.map((t) => (
              <a
                key={t}
                href={`/glossary#${t.toLowerCase().replace(/[^a-z0-9]+/g, "-")}`}
                className="text-[10px] text-blue-400/80 hover:text-blue-300 font-mono"
              >
                {t}
              </a>
            ))}
          </div>
        )}
      </section>
    </article>
  );
}

function Section({
  title,
  children,
  tone,
}: {
  title: string;
  children: React.ReactNode;
  tone?: "amber";
}) {
  return (
    <section>
      <h2 className="text-sm font-medium text-zinc-300 border-b border-zinc-800 pb-1 mb-2">
        {title}
      </h2>
      <div
        className={cn(
          "text-[11px] leading-relaxed whitespace-pre-line",
          tone === "amber" ? "text-amber-100/80" : "text-zinc-400",
        )}
      >
        {children}
      </div>
    </section>
  );
}

function ResultsTable({ rows }: { rows: ResearchResultRow[] }) {
  const showGross = rows.some((r) => r.gross_sharpe != null);
  const showT = rows.some((r) => r.t_stat != null);

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-zinc-800">
            <th className="text-left px-2 py-1 text-[10px] uppercase tracking-wider text-zinc-500 font-normal">
              Variant
            </th>
            {showGross && <Th>Gross Sharpe</Th>}
            <Th>Net Sharpe</Th>
            {showT && <Th>t-stat</Th>}
            <Th>Net return</Th>
            <Th>Max DD</Th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr
              key={r.variant}
              className={cn(
                "border-b border-zinc-900",
                r.is_control && "bg-amber-500/5",
                r.is_headline && "bg-zinc-900/60",
              )}
            >
              <td className="px-2 py-1.5 text-zinc-300 align-top">
                <div className="flex items-baseline gap-1.5">
                  <span>{r.variant}</span>
                  {r.is_control && (
                    <span className="text-[9px] text-amber-400 shrink-0">control</span>
                  )}
                </div>
                {r.note && (
                  <div className="text-[10px] text-zinc-600 mt-0.5 leading-snug">{r.note}</div>
                )}
              </td>
              {showGross && <Td value={r.gross_sharpe} />}
              <Td value={r.net_sharpe} highlight />
              {showT && <Td value={r.t_stat} />}
              <Td value={r.net_annual_return} percent />
              <Td value={r.max_drawdown} percent />
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Th({ children }: { children: React.ReactNode }) {
  return (
    <th className="text-right px-2 py-1 text-[10px] uppercase tracking-wider text-zinc-500 font-normal whitespace-nowrap">
      {children}
    </th>
  );
}

function Td({
  value,
  percent,
  highlight,
}: {
  value: number | null;
  percent?: boolean;
  highlight?: boolean;
}) {
  if (value == null) {
    return <td className="px-2 py-1.5 text-right text-zinc-700 font-mono">—</td>;
  }
  return (
    <td
      className={cn(
        "px-2 py-1.5 text-right font-mono tabular-nums align-top whitespace-nowrap",
        value > 0 ? "text-emerald-400" : "text-red-400",
        highlight && "font-medium",
      )}
    >
      {percent ? `${(value * 100).toFixed(1)}%` : value.toFixed(3)}
    </td>
  );
}

function SharpeChart({ rows }: { rows: ResearchResultRow[] }) {
  const usable = rows.filter((r) => r.net_sharpe != null);
  if (usable.length < 2) return null;

  return (
    <div className="mt-3">
      <Plot
        data={[
          {
            x: usable.map((r) => r.net_sharpe as number),
            y: usable.map((r) => r.variant),
            type: "bar",
            orientation: "h",
            marker: {
              color: usable.map((r) =>
                r.is_control ? "#f59e0b" : (r.net_sharpe as number) > 0 ? "#10b981" : "#ef4444",
              ),
            },
            hovertemplate: "%{y}<br>net Sharpe %{x:.3f}<extra></extra>",
          },
        ]}
        layout={{
          paper_bgcolor: "transparent",
          plot_bgcolor: "transparent",
          font: { color: "#a1a1aa", size: 10, family: "JetBrains Mono, monospace" },
          xaxis: {
            title: { text: "Net Sharpe (control in amber)" },
            gridcolor: "#27272a",
            zerolinecolor: "#52525b",
          },
          yaxis: { automargin: true, autorange: "reversed" },
          margin: { l: 8, r: 16, t: 8, b: 40 },
          height: 40 * usable.length + 80,
          showlegend: false,
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%" }}
      />
    </div>
  );
}
