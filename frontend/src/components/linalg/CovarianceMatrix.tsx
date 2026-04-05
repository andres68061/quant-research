import { useQuery } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import Plot from "react-plotly.js";

import TeX from "@/components/TeX.tsx";
import { api } from "@/lib/api.ts";

const PL: Partial<Plotly.Layout> = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "transparent",
  font: { color: "#a1a1aa", family: "JetBrains Mono, monospace", size: 11 },
  margin: { t: 32, r: 16, b: 40, l: 52 },
};

export default function CovarianceMatrix() {
  const assetsQ = useQuery({ queryKey: ["assets"], queryFn: api.listAssets });
  const allSymbols = (assetsQ.data?.assets ?? []).map((a) => a.symbol);
  const [selected, setSelected] = useState<string[]>([]);
  const [filter, setFilter] = useState("");

  const filtered = filter
    ? allSymbols.filter((s) => s.includes(filter.toUpperCase()))
    : allSymbols.slice(0, 50);

  const priceQueries = useQuery({
    queryKey: ["cov-prices", selected],
    queryFn: async () => {
      if (selected.length < 2) return null;
      const results = await Promise.all(selected.map((s) => api.getPrices(s)));
      return results;
    },
    enabled: selected.length >= 2,
  });

  const { cov, corr, vols, eigenvalues, varExplained } = useMemo(() => {
    if (!priceQueries.data) return { cov: null, corr: null, vols: null, eigenvalues: null, varExplained: null };
    const series = priceQueries.data;
    const minLen = Math.min(...series.map((s) => s.data.length));
    if (minLen < 10) return { cov: null, corr: null, vols: null, eigenvalues: null, varExplained: null };

    const returns: number[][] = series.map((s) => {
      const prices = s.data.slice(-minLen).map((p) => p.price);
      return prices.slice(1).map((p, i) => p / prices[i] - 1);
    });

    const n = returns.length;
    const T = returns[0].length;

    const means = returns.map((r) => r.reduce((s, v) => s + v, 0) / T);
    const covMat: number[][] = Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (_, j) => {
        let sum = 0;
        for (let t = 0; t < T; t++) sum += (returns[i][t] - means[i]) * (returns[j][t] - means[j]);
        return (sum / (T - 1)) * 252;
      }),
    );

    const volArr = covMat.map((_, i) => Math.sqrt(covMat[i][i]));
    const corrMat = covMat.map((row, i) =>
      row.map((v, j) => v / (volArr[i] * volArr[j] || 1)),
    );

    const eigs = powerIteration(covMat);
    const eigSum = eigs.reduce((s, v) => s + v, 0);
    const varExp = eigs.map((e) => (e / eigSum) * 100);

    return { cov: covMat, corr: corrMat, vols: volArr, eigenvalues: eigs, varExplained: varExp };
  }, [priceQueries.data]);

  const toggleSymbol = (sym: string) => {
    setSelected((prev) =>
      prev.includes(sym) ? prev.filter((s) => s !== sym) : [...prev, sym],
    );
  };

  return (
    <div className="space-y-4 overflow-y-auto h-full">
      <h2 className="text-sm font-semibold text-zinc-200">Covariance Matrix</h2>

      <div className="bg-zinc-900/50 border border-zinc-800 rounded p-2">
        <TeX math="\Sigma = \frac{1}{T-1} X^\top X \quad (\text{annualized} \times 252)" />
      </div>

      {/* Stock picker */}
      <div>
        <div className="text-[10px] uppercase text-zinc-500 mb-1">
          Select stocks ({selected.length} chosen)
        </div>
        <input
          type="text"
          placeholder="Filter..."
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1 text-xs text-zinc-200 font-mono mb-1"
        />
        <div className="flex flex-wrap gap-1 max-h-24 overflow-y-auto">
          {filtered.slice(0, 40).map((sym) => (
            <button
              key={sym}
              onClick={() => toggleSymbol(sym)}
              className={`px-1.5 py-0.5 text-[10px] rounded border ${
                selected.includes(sym)
                  ? "border-blue-500 text-blue-400 bg-blue-500/10"
                  : "border-zinc-800 text-zinc-500 hover:text-zinc-300"
              }`}
            >
              {sym}
            </button>
          ))}
        </div>
      </div>

      {selected.length < 2 && (
        <p className="text-xs text-zinc-600">Select at least 2 stocks to compute the covariance matrix.</p>
      )}

      {priceQueries.isLoading && <p className="text-xs text-zinc-500">Loading prices...</p>}

      {cov && corr && vols && (
        <div className="grid grid-cols-2 gap-4">
          <div>
            <div className="text-[10px] uppercase text-zinc-500 mb-1">Covariance (annualized)</div>
            <Plot
              data={[
                {
                  z: cov,
                  x: selected,
                  y: selected,
                  type: "heatmap" as const,
                  colorscale: "RdYlGn",
                  reversescale: true,
                  text: cov.map((r) => r.map((v) => v.toFixed(4))),
                  texttemplate: "%{text}",
                  hoverinfo: "skip",
                },
              ]}
              layout={{ ...PL, height: 300 }}
              useResizeHandler
              className="w-full"
              config={{ displayModeBar: false }}
            />
          </div>
          <div>
            <div className="text-[10px] uppercase text-zinc-500 mb-1">Correlation</div>
            <Plot
              data={[
                {
                  z: corr,
                  x: selected,
                  y: selected,
                  type: "heatmap" as const,
                  colorscale: "RdBu",
                  zmid: 0,
                  text: corr.map((r) => r.map((v) => v.toFixed(2))),
                  texttemplate: "%{text}",
                  hoverinfo: "skip",
                },
              ]}
              layout={{ ...PL, height: 300 }}
              useResizeHandler
              className="w-full"
              config={{ displayModeBar: false }}
            />
          </div>
        </div>
      )}

      {vols && (
        <table className="text-xs font-mono w-full">
          <thead>
            <tr className="text-zinc-500 text-left">
              <th className="pr-3 pb-1">Stock</th>
              <th className="pr-3 pb-1">Ann. Vol</th>
            </tr>
          </thead>
          <tbody>
            {selected.map((s, i) => (
              <tr key={s} className="text-zinc-300 border-t border-zinc-800/50">
                <td className="pr-3 py-1">{s}</td>
                <td className="pr-3 py-1">{(vols[i] * 100).toFixed(2)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {varExplained && (
        <>
          <div className="text-[10px] uppercase text-zinc-500">Principal Components</div>
          <Plot
            data={[
              {
                x: varExplained.map((_, i) => `PC${i + 1}`),
                y: varExplained,
                type: "bar" as const,
                marker: { color: "#60a5fa" },
                name: "Var Explained %",
              },
            ]}
            layout={{
              ...PL,
              height: 220,
              yaxis: { title: { text: "%" } as Partial<Plotly.LayoutAxis["title"]>, gridcolor: "#27272a" },
            }}
            useResizeHandler
            className="w-full"
            config={{ displayModeBar: false }}
          />
        </>
      )}
    </div>
  );
}

/**
 * Rough eigenvalue estimation via power iteration on symmetric positive
 * semi-definite matrices. Returns sorted eigenvalues descending.
 */
function powerIteration(M: number[][]): number[] {
  const n = M.length;
  const eigs: number[] = [];
  let mat = M.map((r) => [...r]);

  for (let k = 0; k < n; k++) {
    let vec = Array.from({ length: n }, () => Math.random());
    for (let iter = 0; iter < 100; iter++) {
      const next = mat.map((row) => row.reduce((s, v, j) => s + v * vec[j], 0));
      const norm = Math.sqrt(next.reduce((s, v) => s + v * v, 0));
      if (norm < 1e-12) break;
      vec = next.map((v) => v / norm);
    }
    const Av = mat.map((row) => row.reduce((s, v, j) => s + v * vec[j], 0));
    const eigVal = vec.reduce((s, v, i) => s + v * Av[i], 0);
    eigs.push(Math.max(0, eigVal));

    mat = mat.map((row, i) =>
      row.map((v, j) => v - eigVal * vec[i] * vec[j]),
    );
  }

  return eigs.sort((a, b) => b - a);
}
