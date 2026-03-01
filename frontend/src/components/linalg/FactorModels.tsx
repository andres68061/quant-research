import { useMemo, useState } from "react";
import Plot from "react-plotly.js";

import TeX from "@/components/TeX.tsx";

const PL: Partial<Plotly.Layout> = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "transparent",
  font: { color: "#a1a1aa", family: "JetBrains Mono, monospace", size: 11 },
  margin: { t: 32, r: 16, b: 44, l: 52 },
};

const FACTORS = ["Market", "Size", "Value"] as const;
const FACTOR_COLORS: Record<string, string> = {
  Alpha: "#71717a",
  Market: "#3b82f6",
  Size: "#22c55e",
  Value: "#f97316",
  Idiosyncratic: "#ef4444",
};

export default function FactorModels() {
  const [nStocks] = useState(5);
  const [factorReturns, setFactorReturns] = useState([5, 1, 2]);
  const [seed, setSeed] = useState(1);

  const { B, alpha, epsilon, decomp } = useMemo(() => {
    const rng = mulberry32(seed);
    const loadings = Array.from({ length: nStocks }, () =>
      FACTORS.map((_, fi) => {
        const v = (rng() - 0.5) * 2;
        return fi === 0 ? Math.abs(v) : v;
      }),
    );
    const a = Array.from({ length: nStocks }, () => (rng() - 0.5) * 0.02);
    const e = Array.from({ length: nStocks }, () => (rng() - 0.5) * 0.04);

    const F = factorReturns.map((r) => r / 100);

    const decomposition = Array.from({ length: nStocks }, (_, i) => {
      const factorContribs = FACTORS.map(
        (_, fi) => loadings[i][fi] * F[fi],
      );
      const total = a[i] + factorContribs.reduce((s, v) => s + v, 0) + e[i];
      return {
        stock: `Stock ${i + 1}`,
        Alpha: a[i],
        Market: factorContribs[0],
        Size: factorContribs[1],
        Value: factorContribs[2],
        Idiosyncratic: e[i],
        Total: total,
      };
    });

    return { B: loadings, alpha: a, epsilon: e, decomp: decomposition };
  }, [nStocks, factorReturns, seed]);

  const components = ["Alpha", "Market", "Size", "Value", "Idiosyncratic"] as const;

  return (
    <div className="space-y-4 overflow-y-auto h-full">
      <h2 className="text-sm font-semibold text-zinc-200">Factor Models</h2>

      <div className="bg-zinc-900/50 border border-zinc-800 rounded p-3 space-y-1">
        <TeX tex="R_i = \alpha_i + \sum_j \beta_{ij} F_j + \varepsilon_i" />
        <p className="text-[10px] text-zinc-500 mt-1">
          Single period: returns decomposed into alpha, factor exposures, and idiosyncratic risk.
        </p>
      </div>

      <div className="grid grid-cols-3 gap-3">
        {FACTORS.map((f, i) => (
          <Field key={f} label={`${f} Return (%)`}>
            <input
              type="range"
              min={-10}
              max={10}
              step={0.5}
              value={factorReturns[i]}
              onChange={(e) => {
                const next = [...factorReturns];
                next[i] = +e.target.value;
                setFactorReturns(next);
              }}
              className="w-full accent-blue-500"
            />
            <span className="text-xs font-mono text-zinc-400">{factorReturns[i].toFixed(1)}%</span>
          </Field>
        ))}
      </div>

      <button
        onClick={() => setSeed((s) => s + 1)}
        className="text-[10px] text-zinc-500 hover:text-zinc-300 underline"
      >
        Regenerate loadings
      </button>

      {/* Stacked bar */}
      <Plot
        data={[
          ...components.map((comp) => ({
            x: decomp.map((d) => d.stock),
            y: decomp.map((d) => d[comp] * 100),
            type: "bar" as const,
            name: comp,
            marker: { color: FACTOR_COLORS[comp] },
          })),
          {
            x: decomp.map((d) => d.stock),
            y: decomp.map((d) => d.Total * 100),
            type: "scatter" as const,
            mode: "markers" as const,
            name: "Total",
            marker: { color: "#d4d4d8", size: 10, symbol: "diamond" },
          },
        ]}
        layout={{
          ...PL,
          barmode: "relative",
          title: { text: "Return Decomposition", font: { size: 13, color: "#d4d4d8" } } as Partial<Plotly.Layout["title"]>,
          yaxis: { title: { text: "Return (%)" } as Partial<Plotly.LayoutAxis["title"]>, gridcolor: "#27272a" },
          showlegend: true,
          legend: { font: { size: 9 }, orientation: "h", y: -0.2 },
          height: 380,
        }}
        useResizeHandler
        className="w-full"
        config={{ displayModeBar: false }}
      />

      {/* Loadings table */}
      <div className="text-[10px] uppercase text-zinc-500">Factor Loadings (beta)</div>
      <table className="text-xs font-mono w-full">
        <thead>
          <tr className="text-zinc-500 text-left">
            <th className="pr-3 pb-1">Stock</th>
            {FACTORS.map((f) => (
              <th key={f} className="pr-3 pb-1">{f}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {B.map((row, i) => (
            <tr key={i} className="text-zinc-300 border-t border-zinc-800/50">
              <td className="pr-3 py-1">Stock {i + 1}</td>
              {row.map((v, j) => (
                <td key={j} className="pr-3 py-1">{v.toFixed(3)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ── Helpers ────────────────────────────────────────────────── */

function mulberry32(seed: number) {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="block text-[10px] uppercase tracking-wider text-zinc-500 mb-1">
        {label}
      </label>
      {children}
    </div>
  );
}
