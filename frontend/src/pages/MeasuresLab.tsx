import { useMutation } from "@tanstack/react-query";
import { useState } from "react";
import Plot from "react-plotly.js";

import RunButton from "@/components/controls/RunButton.tsx";
import AppLayout from "@/components/layout/AppLayout.tsx";
import LeftSidebar from "@/components/layout/LeftSidebar.tsx";
import RightSidebar from "@/components/layout/RightSidebar.tsx";
import TeX from "@/components/TeX.tsx";
import { api } from "@/lib/api.ts";
import type { MeasuresLabResponse, MeasuresLabSeries } from "@/lib/types.ts";
import { cn } from "@/lib/utils.ts";

const PL: Partial<Plotly.Layout> = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "transparent",
  font: { color: "#a1a1aa", family: "JetBrains Mono, monospace", size: 11 },
  margin: { t: 36, r: 16, b: 44, l: 52 },
};

const ARCHETYPE_NAMES = [
  "Steady Eddie",
  "Rollercoaster",
  "Sneaky Losses",
  "Fat Tails",
  "Crash & Recover",
];

const MEASURE_COLS: { key: keyof MeasuresLabSeries["measures"]; label: string; pct?: boolean }[] = [
  { key: "cid1_ratio", label: "Cid-1 Ratio" },
  { key: "cid2_ratio", label: "Cid-2 Ratio" },
  { key: "total_return", label: "Total Ret", pct: true },
  { key: "typical_period_return", label: "Typical Period Ret", pct: true },
  { key: "sharpe_ratio", label: "Sharpe" },
  { key: "sortino_ratio", label: "Sortino" },
  { key: "calmar_ratio", label: "Calmar" },
  { key: "pain_ratio", label: "Pain Ratio" },
  { key: "martin_ratio", label: "Martin Ratio" },
  { key: "max_drawdown", label: "Max DD", pct: true },
];

export default function MeasuresLab() {
  const [nDays, setNDays] = useState(756);
  const [seed, setSeed] = useState(42);
  const [nDraws, setNDraws] = useState(80);
  const [legA, setLegA] = useState("Rollercoaster");
  const [legB, setLegB] = useState("Sneaky Losses");
  const [weightA, setWeightA] = useState(0.5);

  const mut = useMutation({
    mutationFn: () =>
      api.measuresLab({
        n_days: nDays,
        n_relationship_draws: nDraws,
        seed,
        portfolio_a: legA,
        portfolio_b: legB,
        portfolio_weight_a: weightA,
      }),
  });

  const data = mut.data as MeasuresLabResponse | undefined;

  return (
    <AppLayout
      left={
        <LeftSidebar>
          <p className="text-[11px] text-zinc-500 leading-relaxed mb-2">
            Same made-up data, every measure. First a single stock (5 risk shapes), then a 2-asset
            portfolio blend of two of them, then many random single-stock draws to see which
            measures move together (linear) and which don&apos;t.
          </p>

          <Field label="Trading days">
            <select
              value={nDays}
              onChange={(e) => setNDays(+e.target.value)}
              className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5 text-xs text-zinc-200 font-mono"
            >
              {[252, 504, 756, 1260, 2520].map((d) => (
                <option key={d} value={d}>
                  {d} ({(d / 252).toFixed(0)}y)
                </option>
              ))}
            </select>
          </Field>
          <Field label="Seed">
            <input
              type="number"
              min={1}
              max={99999}
              value={seed}
              onChange={(e) => setSeed(+e.target.value)}
              className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5 text-xs font-mono text-zinc-200"
            />
          </Field>

          <div className="mt-3 pt-3 border-t border-zinc-800">
            <p className="text-[10px] uppercase tracking-wider text-zinc-500 mb-2">
              Portfolio blend
            </p>
            <Field label="Leg A">
              <select
                value={legA}
                onChange={(e) => setLegA(e.target.value)}
                className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5 text-xs text-zinc-200 font-mono"
              >
                {ARCHETYPE_NAMES.map((n) => (
                  <option key={n} value={n}>
                    {n}
                  </option>
                ))}
              </select>
            </Field>
            <Field label="Leg B">
              <select
                value={legB}
                onChange={(e) => setLegB(e.target.value)}
                className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5 text-xs text-zinc-200 font-mono"
              >
                {ARCHETYPE_NAMES.map((n) => (
                  <option key={n} value={n}>
                    {n}
                  </option>
                ))}
              </select>
            </Field>
            <Field label={`Weight on Leg A (${(weightA * 100).toFixed(0)}%)`}>
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={weightA}
                onChange={(e) => setWeightA(+e.target.value)}
                className="w-full accent-blue-500"
              />
            </Field>
          </div>

          <div className="mt-3 pt-3 border-t border-zinc-800">
            <Field label={`Relationship draws (${nDraws})`}>
              <input
                type="range"
                min={10}
                max={300}
                step={10}
                value={nDraws}
                onChange={(e) => setNDraws(+e.target.value)}
                className="w-full accent-blue-500"
              />
            </Field>
          </div>

          <RunButton onClick={() => mut.mutate()} loading={mut.isPending} label="Generate" />
          {mut.isError && (
            <p className="text-xs text-red-400 mt-2">{(mut.error as Error).message}</p>
          )}

          <div className="mt-4 pt-3 border-t border-zinc-800">
            <div className="text-[10px] uppercase tracking-wider text-zinc-500 mb-3">
              Formulas (wealth relative to $1, w<sub>0</sub>=1)
            </div>
            <div className="flex flex-col gap-4 text-[11px] text-zinc-400">
              <div>
                <div className="text-[10px] text-zinc-600 mb-1">Main diagnostic measures</div>
                <TeX display math="\text{CostBasisShortfall} = \sum_{t=1}^{T} \max(0,\, 1 - w_t)" />
                <TeX display math="\text{Cid-1} = \frac{w_T - 1}{\text{CostBasisShortfall}}" />
                <TeX display math="\text{Typical}_{n} = \tfrac{1}{K}\sum_{k=1}^{K}\left(\prod_{i \in \text{block}_k}(1+r_i) - 1\right)" />
                <TeX display math="\text{Cid-2} = \frac{\text{Typical}_n}{\text{CostBasisShortfall}}" />
              </div>
              <div className="pt-2 border-t border-zinc-800/60">
                <div className="text-[10px] text-zinc-600 mb-1">Reference measures</div>
                <TeX display math="S = \frac{\bar r - r_f}{\sigma}\cdot\sqrt{252}" />
                <TeX display math="\text{Sortino} = \frac{\bar r - r_f}{\sigma_{\text{down}}}\cdot\sqrt{252}" />
                <TeX display math="\text{Calmar} = \frac{R_{ann}}{|\text{max DD}|}" />
                <TeX display math="\text{PainIndex} = \tfrac{1}{T}\sum_t |w_t - \text{peak}_t| / \text{peak}_t" />
                <TeX display math="\text{PainRatio} = \frac{R_{ann}}{\text{PainIndex}}" />
                <TeX display math="\text{Ulcer} = \sqrt{\tfrac{1}{T}\sum_t \left(\frac{w_t-\text{peak}_t}{\text{peak}_t}\right)^2}" />
                <TeX display math="\text{Martin} = \frac{R_{ann}}{\text{Ulcer}}" />
              </div>
            </div>
          </div>
        </LeftSidebar>
      }
      right={
        <RightSidebar>
          {!data ? (
            <p className="text-xs text-zinc-600">Generate to see measures</p>
          ) : (
            <p className="text-[11px] text-zinc-500 leading-relaxed">
              Cid-1/Cid-2/Pain/Martin measures are all diagnostic-only here — none of these
              synthetic paths were optimized against any of them.
            </p>
          )}
        </RightSidebar>
      }
    >
      {!data ? (
        <EmptyState />
      ) : (
        <div className="space-y-16 pb-8">
          <Section title="1. Single stock — five risk shapes, same window">
            <div className="w-full h-[420px]">
              <GrowthChart series={data.single_stock_examples} title="Growth of $100" />
            </div>
            <MeasuresTable rows={data.single_stock_examples} />
          </Section>

          <Section
            title="2. Portfolio — blend of two of those shapes"
            subtitle="Compare the blend's row to the two legs' own rows: a measure that isn't a linear
              function of its inputs (every ratio here) will not sit at the weighted average of the
              legs' values."
          >
            <div className="w-full h-[420px]">
              <GrowthChart
                series={[...data.portfolio_legs, data.portfolio_example]}
                title="Growth of $100 — legs + blend"
              />
            </div>
            <MeasuresTable rows={[...data.portfolio_legs, data.portfolio_example]} />
          </Section>

          <Section
            title="3. Relationships across many random single-stock draws"
            subtitle="Each point is one random synthetic path. A tight diagonal band = the two measures
              move together almost linearly; a scattered cloud = they capture different things."
          >
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <ScatterPanel
                data={data.relationship_scatter}
                xKey="cid1_ratio"
                yKey="cid2_ratio"
                xLabel="Cid-1 Ratio"
                yLabel="Cid-2 Ratio"
              />
              <ScatterPanel
                data={data.relationship_scatter}
                xKey="sharpe_ratio"
                yKey="cid1_ratio"
                xLabel="Sharpe"
                yLabel="Cid-1 Ratio"
              />
              <ScatterPanel
                data={data.relationship_scatter}
                xKey="sharpe_ratio"
                yKey="pain_ratio"
                xLabel="Sharpe"
                yLabel="Pain Ratio"
              />
              <ScatterPanel
                data={data.relationship_scatter}
                xKey="pain_ratio"
                yKey="martin_ratio"
                xLabel="Pain Ratio"
                yLabel="Martin Ratio"
              />
            </div>
          </Section>
        </div>
      )}
    </AppLayout>
  );
}

function GrowthChart({ series, title }: { series: MeasuresLabSeries[]; title: string }) {
  return (
    <Plot
      data={series.map((s) => ({
        y: s.prices,
        type: "scatter" as const,
        mode: "lines" as const,
        name: s.name,
        line: { color: s.color, width: 2 },
      }))}
      layout={{
        ...PL,
        autosize: true,
        title: { text: title, font: { size: 13, color: "#d4d4d8" } } as Partial<
          Plotly.Layout["title"]
        >,
        yaxis: { title: { text: "Value ($)" } as Partial<Plotly.LayoutAxis["title"]>, gridcolor: "#27272a" },
        xaxis: { title: { text: "Trading Day" } as Partial<Plotly.LayoutAxis["title"]>, gridcolor: "#27272a" },
        showlegend: true,
        legend: { font: { size: 10 } },
      }}
      useResizeHandler
      className="w-full h-full"
      style={{ width: "100%", height: "100%" }}
      config={{ displayModeBar: false }}
    />
  );
}

function ScatterPanel({
  data,
  xKey,
  yKey,
  xLabel,
  yLabel,
}: {
  data: Record<string, number[]>;
  xKey: string;
  yKey: string;
  xLabel: string;
  yLabel: string;
}) {
  return (
    <div className="h-[320px] bg-zinc-900 border border-zinc-800 rounded p-2">
      <Plot
        data={[
          {
            x: data[xKey],
            y: data[yKey],
            type: "scatter",
            mode: "markers",
            marker: { color: "#38bdf8", size: 6, opacity: 0.75 },
          },
        ]}
        layout={{
          ...PL,
          autosize: true,
          title: { text: `${xLabel} vs ${yLabel}`, font: { size: 12, color: "#d4d4d8" } } as Partial<
            Plotly.Layout["title"]
          >,
          xaxis: { title: { text: xLabel } as Partial<Plotly.LayoutAxis["title"]>, gridcolor: "#27272a" },
          yaxis: { title: { text: yLabel } as Partial<Plotly.LayoutAxis["title"]>, gridcolor: "#27272a" },
          showlegend: false,
        }}
        useResizeHandler
        className="w-full h-full"
        style={{ width: "100%", height: "100%" }}
        config={{ displayModeBar: false }}
      />
    </div>
  );
}

function MeasuresTable({ rows }: { rows: MeasuresLabSeries[] }) {
  return (
    <div className="overflow-x-auto mt-4">
      <table className="text-xs font-mono w-full">
        <thead>
          <tr className="text-zinc-500 text-left">
            <th className="pr-3 pb-1">Name</th>
            {MEASURE_COLS.map((c) => (
              <th key={c.key} className="pr-3 pb-1 text-right">
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.name} className="text-zinc-300 border-t border-zinc-800/50">
              <td className="pr-3 py-1 font-semibold whitespace-nowrap" style={{ color: r.color }}>
                {r.name}
              </td>
              {MEASURE_COLS.map((c) => {
                const v = r.measures[c.key];
                return (
                  <td
                    key={c.key}
                    className={cn(
                      "pr-3 py-1 text-right tabular-nums",
                      v >= 0 ? "text-zinc-300" : "text-red-400",
                    )}
                  >
                    {c.pct ? `${(v * 100).toFixed(1)}%` : v.toFixed(2)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Section({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
}) {
  return (
    <section className="border-t border-zinc-800 pt-6 first:border-t-0 first:pt-0">
      <h2 className="text-sm font-semibold text-zinc-200 mb-1">{title}</h2>
      {subtitle && (
        <p className="text-[11px] text-zinc-500 mb-4 leading-relaxed max-w-3xl">{subtitle}</p>
      )}
      <div className="flex flex-col gap-4">{children}</div>
    </section>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="mb-2">
      <label className="block text-[10px] uppercase tracking-wider text-zinc-500 mb-1">{label}</label>
      {children}
    </div>
  );
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center h-full text-zinc-600 gap-4">
      <div className="text-lg font-semibold">Measures Lab</div>
      <p className="text-xs max-w-md text-center leading-relaxed">
        Pick a window and portfolio blend on the left, then Generate to see every measure on the
        same synthetic data — single stock, portfolio, and cross-measure relationships.
      </p>
    </div>
  );
}
