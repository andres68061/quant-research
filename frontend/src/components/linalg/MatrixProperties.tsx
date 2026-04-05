import { useMemo, useState } from "react";
import Plot from "react-plotly.js";

import TeX from "@/components/TeX.tsx";

const PL: Partial<Plotly.Layout> = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "transparent",
  font: { color: "#a1a1aa", family: "JetBrains Mono, monospace", size: 11 },
  margin: { t: 32, r: 16, b: 40, l: 52 },
};

export default function MatrixProperties() {
  const [n, setN] = useState(3);
  const [values, setValues] = useState<number[]>([
    2, 1, 0, 1, 3, 1, 0, 1, 2,
  ]);

  const M = useMemo(() => {
    const m: number[][] = [];
    for (let i = 0; i < n; i++) m.push(values.slice(i * n, (i + 1) * n));
    return m;
  }, [values, n]);

  const det = useMemo(() => determinant(M), [M]);
  const trace = useMemo(() => M.reduce((s, r, i) => s + (r[i] ?? 0), 0), [M]);
  const isSymmetric = useMemo(
    () => M.every((r, i) => r.every((v, j) => Math.abs(v - M[j][i]) < 1e-9)),
    [M],
  );
  const inv = useMemo(() => (Math.abs(det) > 1e-10 ? invert(M) : null), [M, det]);
  const eigenvalues = useMemo(() => eigenApprox(M), [M]);

  const handleSizeChange = (newN: number) => {
    const newVals: number[] = [];
    for (let i = 0; i < newN; i++)
      for (let j = 0; j < newN; j++)
        newVals.push(i === j ? 1 : 0);
    setN(newN);
    setValues(newVals);
  };

  return (
    <div className="space-y-4 overflow-y-auto h-full">
      <h2 className="text-sm font-semibold text-zinc-200">Matrix Properties</h2>

      <div className="flex items-center gap-3">
        <span className="text-[10px] uppercase text-zinc-500">Size</span>
        {[2, 3, 4].map((s) => (
          <button
            key={s}
            onClick={() => handleSizeChange(s)}
            className={`px-2 py-0.5 text-[10px] rounded border ${
              n === s
                ? "border-blue-500 text-blue-400 bg-blue-500/10"
                : "border-zinc-800 text-zinc-500 hover:text-zinc-300"
            }`}
          >
            {s}x{s}
          </button>
        ))}
      </div>

      {/* Matrix input */}
      <div className="inline-grid gap-1" style={{ gridTemplateColumns: `repeat(${n}, 1fr)` }}>
        {values.slice(0, n * n).map((v, idx) => (
          <input
            key={idx}
            type="number"
            step={0.5}
            value={v}
            onChange={(e) => {
              const next = [...values];
              next[idx] = +e.target.value;
              setValues(next);
            }}
            className="w-16 bg-zinc-900 border border-zinc-800 rounded px-2 py-1 text-xs text-zinc-200 font-mono text-center"
          />
        ))}
      </div>

      {/* Properties */}
      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-3">
          <Prop label="Determinant" value={det.toFixed(4)} ok={Math.abs(det) > 1e-10} />
          <Prop label="Trace" value={trace.toFixed(4)} />
          <Prop label="Symmetric" value={isSymmetric ? "Yes" : "No"} ok={isSymmetric} />
          <Prop
            label="Invertible"
            value={Math.abs(det) > 1e-10 ? "Yes" : "Singular"}
            ok={Math.abs(det) > 1e-10}
          />
        </div>

        <div className="space-y-3">
          <div className="text-[10px] uppercase text-zinc-500">Eigenvalues (approx)</div>
          <div className="flex flex-wrap gap-2">
            {eigenvalues.map((e, i) => (
              <span
                key={i}
                className="bg-zinc-900 border border-zinc-800 rounded px-2 py-0.5 text-xs font-mono text-emerald-400"
              >
                {e.toFixed(3)}
              </span>
            ))}
          </div>

          <Plot
            data={[
              {
                x: eigenvalues.map((_, i) => `lambda${i + 1}`),
                y: eigenvalues,
                type: "bar" as const,
                marker: { color: "#60a5fa" },
              },
            ]}
            layout={{
              ...PL,
              height: 180,
              yaxis: { gridcolor: "#27272a" },
            }}
            useResizeHandler
            className="w-full"
            config={{ displayModeBar: false }}
          />
        </div>
      </div>

      {/* Transpose */}
      <div>
        <div className="text-[10px] uppercase text-zinc-500 mb-1">Transpose</div>
        <MatGrid mat={M.map((_, i) => M.map((r) => r[i]))} n={n} />
      </div>

      {/* Inverse */}
      {inv && (
        <div>
          <div className="text-[10px] uppercase text-zinc-500 mb-1">Inverse</div>
          <MatGrid mat={inv} n={n} />
        </div>
      )}

      {/* Formulas */}
      <div className="bg-zinc-900/50 border border-zinc-800 rounded p-3 space-y-2">
        <TeX math="\det(A),\; \mathrm{tr}(A) = \sum a_{ii},\; A A^{-1} = I" />
        <TeX math="A\mathbf{v} = \lambda \mathbf{v} \quad \text{(eigenvalue equation)}" />
      </div>
    </div>
  );
}

/* ── Small components ───────────────────────────────────────── */

function Prop({ label, value, ok }: { label: string; value: string; ok?: boolean }) {
  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded px-3 py-2">
      <div className="text-[9px] uppercase text-zinc-500">{label}</div>
      <div
        className={`text-sm font-mono ${
          ok === true ? "text-emerald-400" : ok === false ? "text-red-400" : "text-zinc-200"
        }`}
      >
        {value}
      </div>
    </div>
  );
}

function MatGrid({ mat, n }: { mat: number[][]; n: number }) {
  return (
    <div className="inline-grid gap-1 font-mono text-xs text-zinc-300" style={{ gridTemplateColumns: `repeat(${n}, 4rem)` }}>
      {mat.flat().map((v, i) => (
        <span key={i} className="bg-zinc-900 border border-zinc-800 rounded px-1 py-0.5 text-center">
          {v.toFixed(3)}
        </span>
      ))}
    </div>
  );
}

/* ── Math helpers ───────────────────────────────────────────── */

function determinant(M: number[][]): number {
  const n = M.length;
  if (n === 1) return M[0][0];
  if (n === 2) return M[0][0] * M[1][1] - M[0][1] * M[1][0];
  let d = 0;
  for (let j = 0; j < n; j++) {
    const sub = M.slice(1).map((r) => [...r.slice(0, j), ...r.slice(j + 1)]);
    d += (j % 2 === 0 ? 1 : -1) * M[0][j] * determinant(sub);
  }
  return d;
}

function invert(M: number[][]): number[][] | null {
  const n = M.length;
  const aug = M.map((r, i) => [...r, ...Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))]);

  for (let i = 0; i < n; i++) {
    let maxRow = i;
    for (let k = i + 1; k < n; k++) {
      if (Math.abs(aug[k][i]) > Math.abs(aug[maxRow][i])) maxRow = k;
    }
    [aug[i], aug[maxRow]] = [aug[maxRow], aug[i]];
    const pivot = aug[i][i];
    if (Math.abs(pivot) < 1e-12) return null;
    for (let j = 0; j < 2 * n; j++) aug[i][j] /= pivot;
    for (let k = 0; k < n; k++) {
      if (k === i) continue;
      const factor = aug[k][i];
      for (let j = 0; j < 2 * n; j++) aug[k][j] -= factor * aug[i][j];
    }
  }

  return aug.map((r) => r.slice(n));
}

function eigenApprox(M: number[][]): number[] {
  const n = M.length;
  const eigenvals: number[] = [];
  let mat = M.map((r) => [...r]);

  for (let k = 0; k < n; k++) {
    let vec = Array.from({ length: n }, () => Math.random());
    for (let iter = 0; iter < 80; iter++) {
      const next = mat.map((row) => row.reduce((s, v, j) => s + v * vec[j], 0));
      const norm = Math.sqrt(next.reduce((s, v) => s + v * v, 0));
      if (norm < 1e-14) break;
      vec = next.map((v) => v / norm);
    }
    const Av = mat.map((row) => row.reduce((s, v, j) => s + v * vec[j], 0));
    const eigVal = vec.reduce((s, v, i) => s + v * Av[i], 0);
    eigenvals.push(eigVal);
    mat = mat.map((row, i) => row.map((v, j) => v - eigVal * vec[i] * vec[j]));
  }

  return eigenvals.sort((a, b) => b - a);
}
