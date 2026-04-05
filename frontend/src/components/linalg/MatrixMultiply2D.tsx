import { useState } from "react";
import Plot from "react-plotly.js";

import TeX from "@/components/TeX.tsx";

const PL: Partial<Plotly.Layout> = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "transparent",
  font: { color: "#a1a1aa", family: "JetBrains Mono, monospace", size: 11 },
  margin: { t: 24, r: 16, b: 40, l: 40 },
};

export default function MatrixMultiply2D() {
  const [a, setA] = useState([2, 0, 0, 1]);
  const [v, setV] = useState([1, 1]);

  const [[a11, a12], [a21, a22]] = [
    [a[0], a[1]],
    [a[2], a[3]],
  ];
  const [v1, v2] = v;

  const rx = a11 * v1 + a12 * v2;
  const ry = a21 * v1 + a22 * v2;

  const e1t = [a11, a21];
  const e2t = [a12, a22];

  const axRange =
    Math.max(5, ...[v1, v2, rx, ry, ...e1t, ...e2t].map((x) => Math.abs(x) * 1.5));

  const presets: { label: string; mat: number[] }[] = [
    { label: "Rotation 45deg", mat: [0.707, -0.707, 0.707, 0.707] },
    { label: "Scale 2x", mat: [2, 0, 0, 2] },
    { label: "Shear", mat: [1, 0.5, 0, 1] },
    { label: "Reflect Y", mat: [-1, 0, 0, 1] },
  ];

  return (
    <div className="space-y-4 overflow-y-auto h-full">
      <h2 className="text-sm font-semibold text-zinc-200">2D Matrix Multiplication</h2>

      <div className="grid grid-cols-3 gap-4">
        {/* Matrix A */}
        <div>
          <div className="text-[10px] uppercase text-zinc-500 mb-1">Matrix A</div>
          <div className="grid grid-cols-2 gap-1">
            {a.map((val, i) => (
              <input
                key={i}
                type="number"
                step="0.1"
                value={val}
                onChange={(e) => {
                  const next = [...a];
                  next[i] = +e.target.value;
                  setA(next);
                }}
                className="bg-zinc-900 border border-zinc-800 rounded px-2 py-1 text-xs text-zinc-200 font-mono w-full"
              />
            ))}
          </div>
        </div>

        {/* Vector v */}
        <div>
          <div className="text-[10px] uppercase text-zinc-500 mb-1">Vector v</div>
          <div className="grid grid-cols-1 gap-1">
            {v.map((val, i) => (
              <input
                key={i}
                type="number"
                step="0.1"
                value={val}
                onChange={(e) => {
                  const next = [...v];
                  next[i] = +e.target.value;
                  setV(next);
                }}
                className="bg-zinc-900 border border-zinc-800 rounded px-2 py-1 text-xs text-zinc-200 font-mono w-full"
              />
            ))}
          </div>
        </div>

        {/* Result */}
        <div>
          <div className="text-[10px] uppercase text-zinc-500 mb-1">Result Av</div>
          <div className="font-mono text-xs text-emerald-400 space-y-1">
            <div>[{rx.toFixed(2)}]</div>
            <div>[{ry.toFixed(2)}]</div>
          </div>
        </div>
      </div>

      {/* Presets */}
      <div className="flex gap-2 flex-wrap">
        {presets.map((p) => (
          <button
            key={p.label}
            onClick={() => setA(p.mat)}
            className="px-2 py-1 text-[10px] bg-zinc-900 border border-zinc-800 rounded text-zinc-400 hover:text-zinc-200 transition-colors"
          >
            {p.label}
          </button>
        ))}
      </div>

      {/* Equation */}
      <div className="bg-zinc-900/50 border border-zinc-800 rounded p-3">
        <TeX
          math={`\\begin{bmatrix} ${a11} & ${a12} \\\\ ${a21} & ${a22} \\end{bmatrix} \\begin{bmatrix} ${v1} \\\\ ${v2} \\end{bmatrix} = \\begin{bmatrix} ${rx.toFixed(2)} \\\\ ${ry.toFixed(2)} \\end{bmatrix}`}
        />
      </div>

      {/* Plot */}
      <Plot
        data={[
          arrow(0, 0, v1, v2, "#60a5fa", `v = (${v1}, ${v2})`),
          arrow(0, 0, rx, ry, "#ef4444", `Av = (${rx.toFixed(1)}, ${ry.toFixed(1)})`),
          arrow(0, 0, e1t[0], e1t[1], "#22d3ee", "Ae1", true),
          arrow(0, 0, e2t[0], e2t[1], "#4ade80", "Ae2", true),
        ]}
        layout={{
          ...PL,
          xaxis: {
            range: [-axRange, axRange],
            zeroline: true,
            zerolinecolor: "#52525b",
            gridcolor: "#27272a",
          },
          yaxis: {
            range: [-axRange, axRange],
            scaleanchor: "x",
            zeroline: true,
            zerolinecolor: "#52525b",
            gridcolor: "#27272a",
          },
          showlegend: true,
          legend: { font: { size: 9 }, x: 0, y: 1 },
          height: 420,
        }}
        useResizeHandler
        className="w-full"
        config={{ displayModeBar: false }}
      />

      <p className="text-[11px] text-zinc-500">
        Blue = original vector, Red = transformed, Cyan/Green = transformed basis vectors.
      </p>
    </div>
  );
}

function arrow(
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  color: string,
  name: string,
  dashed = false,
): Plotly.Data {
  return {
    x: [x0, x1],
    y: [y0, y1],
    type: "scatter",
    mode: "lines+markers",
    line: { color, width: dashed ? 2 : 3, dash: dashed ? "dash" : "solid" },
    marker: { size: [0, 8] },
    name,
  } as Plotly.Data;
}
