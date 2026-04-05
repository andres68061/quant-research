import { useMemo, useState } from "react";
import Plot from "react-plotly.js";

import TeX from "@/components/TeX.tsx";

const PL: Partial<Plotly.Layout> = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "transparent",
  font: { color: "#a1a1aa", family: "JetBrains Mono, monospace", size: 11 },
  margin: { t: 32, r: 16, b: 40, l: 52 },
};

export default function PortfolioVariance() {
  const [nAssets, setNAssets] = useState(3);
  const [weights, setWeights] = useState<number[]>([0.34, 0.33, 0.33]);
  const [covSeed, setCovSeed] = useState(1);

  const Sigma = useMemo(() => generateCov(nAssets, covSeed), [nAssets, covSeed]);

  const w = normalizeWeights(weights, nAssets);

  const sigmaW = Sigma.map((row) =>
    row.reduce((sum, val, j) => sum + val * w[j], 0),
  );
  const variance = w.reduce((sum, wi, i) => sum + wi * sigmaW[i], 0);
  const vol = Math.sqrt(Math.max(0, variance));

  const contrib = Sigma.map((row, i) =>
    row.map((s, j) => w[i] * w[j] * s),
  );

  const marginal = sigmaW;
  const totalContrib = w.map((wi, i) =>
    variance > 1e-12 ? (wi * marginal[i]) / variance : 0,
  );

  return (
    <div className="space-y-4 overflow-y-auto h-full">
      <h2 className="text-sm font-semibold text-zinc-200">
        Portfolio Variance: w<sup>T</sup> &Sigma; w
      </h2>

      <div className="bg-zinc-900/50 border border-zinc-800 rounded p-3">
        <TeX math="\sigma^2_p = \mathbf{w}^\top \Sigma \mathbf{w}" />
      </div>

      <div className="grid grid-cols-2 gap-4">
        {/* Left: controls */}
        <div className="space-y-3">
          <Field label="Assets">
            <select
              value={nAssets}
              onChange={(e) => {
                const n = +e.target.value;
                setNAssets(n);
                setWeights(Array(n).fill(+(1 / n).toFixed(4)));
              }}
              className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1 text-xs text-zinc-200 font-mono"
            >
              {[2, 3, 4, 5].map((n) => (
                <option key={n} value={n}>{n}</option>
              ))}
            </select>
          </Field>

          {w.map((_, i) => (
            <Field key={i} label={`w${i + 1}: ${w[i].toFixed(2)}`}>
              <input
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={weights[i] ?? 0}
                onChange={(e) => {
                  const next = [...weights];
                  next[i] = +e.target.value;
                  setWeights(next);
                }}
                className="w-full accent-blue-500"
              />
            </Field>
          ))}

          <button
            onClick={() => setCovSeed((s) => s + 1)}
            className="text-[10px] text-zinc-500 hover:text-zinc-300 underline"
          >
            Regenerate covariance matrix
          </button>

          <div className="bg-zinc-900 border border-zinc-800 rounded p-2 space-y-1">
            <div className="text-[10px] uppercase text-zinc-500">Results</div>
            <div className="text-xs font-mono text-emerald-400">
              Variance: {variance.toFixed(6)}
            </div>
            <div className="text-xs font-mono text-emerald-400">
              Volatility: {(vol * 100).toFixed(2)}%
            </div>
          </div>
        </div>

        {/* Right: heatmap */}
        <div>
          <div className="text-[10px] uppercase text-zinc-500 mb-1">
            Risk Contribution (w_i w_j sigma_ij)
          </div>
          <Plot
            data={[
              {
                z: contrib,
                x: labels(nAssets),
                y: labels(nAssets),
                type: "heatmap" as const,
                colorscale: "RdYlGn",
                reversescale: true,
                text: contrib.map((r) => r.map((v) => v.toFixed(4))),
                texttemplate: "%{text}",
                hoverinfo: "skip",
              },
            ]}
            layout={{ ...PL, height: 280 }}
            useResizeHandler
            className="w-full"
            config={{ displayModeBar: false }}
          />
        </div>
      </div>

      {/* Contribution table */}
      <table className="text-xs font-mono w-full">
        <thead>
          <tr className="text-zinc-500 text-left">
            <th className="pr-3 pb-1">Asset</th>
            <th className="pr-3 pb-1">Weight</th>
            <th className="pr-3 pb-1">Marginal</th>
            <th className="pr-3 pb-1">% Contrib</th>
          </tr>
        </thead>
        <tbody>
          {w.map((_, i) => (
            <tr key={i} className="text-zinc-300 border-t border-zinc-800/50">
              <td className="pr-3 py-1">Asset {i + 1}</td>
              <td className="pr-3 py-1">{(w[i] * 100).toFixed(1)}%</td>
              <td className="pr-3 py-1">{marginal[i].toFixed(4)}</td>
              <td className="pr-3 py-1">{(totalContrib[i] * 100).toFixed(1)}%</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ── Helpers ────────────────────────────────────────────────── */

function generateCov(n: number, seed: number): number[][] {
  const rng = mulberry32(seed);
  const A: number[][] = Array.from({ length: n }, () =>
    Array.from({ length: n }, () => rng() - 0.5),
  );
  const vols = Array.from({ length: n }, () => 0.15 + rng() * 0.2);

  const AAT = A.map((row, i) =>
    A.map((_, j) => row.reduce((s, _, k) => s + A[i][k] * A[j][k], 0)),
  );

  const diag = AAT.map((r, i) => Math.sqrt(r[i]));
  const corr = AAT.map((r, i) => r.map((v, j) => v / (diag[i] * diag[j])));
  return corr.map((r, i) => r.map((c, j) => c * vols[i] * vols[j]));
}

function mulberry32(seed: number) {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function normalizeWeights(raw: number[], n: number): number[] {
  const arr = raw.slice(0, n);
  while (arr.length < n) arr.push(0);
  const sum = arr.reduce((s, v) => s + v, 0);
  return sum > 0 ? arr.map((v) => v / sum) : arr.map(() => 1 / n);
}

function labels(n: number): string[] {
  return Array.from({ length: n }, (_, i) => `A${i + 1}`);
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
