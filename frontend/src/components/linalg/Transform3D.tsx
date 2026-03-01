import { useState } from "react";
import Plot from "react-plotly.js";

import TeX from "@/components/TeX.tsx";

type TransformType = "rotation-x" | "rotation-y" | "rotation-z" | "scale" | "shear" | "custom";

const PL: Partial<Plotly.Layout> = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "transparent",
  font: { color: "#a1a1aa", family: "JetBrains Mono, monospace", size: 11 },
  margin: { t: 8, r: 8, b: 8, l: 8 },
};

export default function Transform3D() {
  const [type, setType] = useState<TransformType>("rotation-z");
  const [angle, setAngle] = useState(45);
  const [scaleVec, setScaleVec] = useState([2, 1, 1]);
  const [shear, setShear] = useState(0.5);
  const [custom, setCustom] = useState([1, 0, 0, 0, 1, 0, 0, 0, 1]);

  const T = buildMatrix(type, angle, scaleVec, shear, custom);

  const cubeV = [
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
  ];
  const transformed = cubeV.map((v) => mulMat3(T, v));

  const edges = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
  ];

  const basis = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  const tBasis = basis.map((b) => mulMat3(T, b));
  const bColors = ["#ef4444", "#22c55e", "#3b82f6"];
  const bNames = ["X", "Y", "Z"];

  const traces: Plotly.Data[] = [];

  edges.forEach((e) => {
    traces.push(line3d(cubeV, e, "#52525b", true));
    traces.push(line3d(transformed, e, "#a855f7", false));
  });

  basis.forEach((b, i) => {
    traces.push(vec3d([0, 0, 0], b, bColors[i], `${bNames[i]} orig`, true));
    traces.push(vec3d([0, 0, 0], tBasis[i], bColors[i], `${bNames[i]} new`, false));
  });

  return (
    <div className="space-y-4 overflow-y-auto h-full">
      <h2 className="text-sm font-semibold text-zinc-200">3D Geometric Transformations</h2>

      <div className="flex gap-2 flex-wrap">
        {(["rotation-x", "rotation-y", "rotation-z", "scale", "shear", "custom"] as TransformType[]).map((t) => (
          <button
            key={t}
            onClick={() => setType(t)}
            className={`px-2 py-1 text-[10px] rounded border ${
              type === t
                ? "border-blue-500 text-blue-400 bg-blue-500/10"
                : "border-zinc-800 text-zinc-500 hover:text-zinc-300"
            }`}
          >
            {t.replace("-", " ")}
          </button>
        ))}
      </div>

      {type.startsWith("rotation") && (
        <Field label={`Angle: ${angle}deg`}>
          <input
            type="range"
            min={0}
            max={360}
            value={angle}
            onChange={(e) => setAngle(+e.target.value)}
            className="w-full accent-blue-500"
          />
        </Field>
      )}

      {type === "scale" && (
        <div className="grid grid-cols-3 gap-2">
          {["sx", "sy", "sz"].map((l, i) => (
            <Field key={l} label={l}>
              <input
                type="number"
                step={0.1}
                value={scaleVec[i]}
                onChange={(e) => {
                  const next = [...scaleVec];
                  next[i] = +e.target.value;
                  setScaleVec(next);
                }}
                className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1 text-xs text-zinc-200 font-mono"
              />
            </Field>
          ))}
        </div>
      )}

      {type === "shear" && (
        <Field label={`Shear: ${shear.toFixed(1)}`}>
          <input
            type="range"
            min={-1}
            max={1}
            step={0.1}
            value={shear}
            onChange={(e) => setShear(+e.target.value)}
            className="w-full accent-blue-500"
          />
        </Field>
      )}

      {type === "custom" && (
        <div className="grid grid-cols-3 gap-1">
          {custom.map((v, i) => (
            <input
              key={i}
              type="number"
              step={0.1}
              value={v}
              onChange={(e) => {
                const n = [...custom];
                n[i] = +e.target.value;
                setCustom(n);
              }}
              className="bg-zinc-900 border border-zinc-800 rounded px-2 py-1 text-xs text-zinc-200 font-mono"
            />
          ))}
        </div>
      )}

      {/* LaTeX */}
      <div className="bg-zinc-900/50 border border-zinc-800 rounded p-2">
        <TeX tex={matToTex(T)} />
      </div>

      <Plot
        data={traces}
        layout={{
          ...PL,
          scene: {
            xaxis: { range: [-3, 3], title: "X", gridcolor: "#27272a" },
            yaxis: { range: [-3, 3], title: "Y", gridcolor: "#27272a" },
            zaxis: { range: [-3, 3], title: "Z", gridcolor: "#27272a" },
            aspectmode: "cube",
          },
          showlegend: false,
          height: 480,
        }}
        useResizeHandler
        className="w-full"
        config={{ displayModeBar: false }}
      />

      <p className="text-[11px] text-zinc-500">
        Gray dashed = original cube, Purple = transformed. Solid/dashed arrows = transformed/original basis.
      </p>
    </div>
  );
}

/* ── Math helpers ───────────────────────────────────────────── */

function buildMatrix(
  type: TransformType,
  angleDeg: number,
  scaleVec: number[],
  shear: number,
  custom: number[],
): number[][] {
  const t = (angleDeg * Math.PI) / 180;
  const c = Math.cos(t);
  const s = Math.sin(t);

  switch (type) {
    case "rotation-x":
      return [[1, 0, 0], [0, c, -s], [0, s, c]];
    case "rotation-y":
      return [[c, 0, s], [0, 1, 0], [-s, 0, c]];
    case "rotation-z":
      return [[c, -s, 0], [s, c, 0], [0, 0, 1]];
    case "scale":
      return [[scaleVec[0], 0, 0], [0, scaleVec[1], 0], [0, 0, scaleVec[2]]];
    case "shear":
      return [[1, shear, 0], [0, 1, 0], [0, 0, 1]];
    case "custom":
      return [custom.slice(0, 3), custom.slice(3, 6), custom.slice(6, 9)];
  }
}

function mulMat3(M: number[][], v: number[]): number[] {
  return M.map((row) => row[0] * v[0] + row[1] * v[1] + row[2] * v[2]);
}

function matToTex(M: number[][]): string {
  const rows = M.map((r) => r.map((v) => v.toFixed(2)).join(" & ")).join(" \\\\ ");
  return `T = \\begin{bmatrix} ${rows} \\end{bmatrix}`;
}

function line3d(verts: number[][], edge: number[], color: string, dashed: boolean): Plotly.Data {
  return {
    x: edge.map((i) => verts[i][0]),
    y: edge.map((i) => verts[i][1]),
    z: edge.map((i) => verts[i][2]),
    type: "scatter3d",
    mode: "lines",
    line: { color, width: dashed ? 2 : 3, dash: dashed ? "dot" : "solid" },
    showlegend: false,
    hoverinfo: "skip",
  } as Plotly.Data;
}

function vec3d(
  from: number[],
  to: number[],
  color: string,
  name: string,
  dashed: boolean,
): Plotly.Data {
  return {
    x: [from[0], to[0]],
    y: [from[1], to[1]],
    z: [from[2], to[2]],
    type: "scatter3d",
    mode: "lines+markers",
    line: { color, width: dashed ? 2 : 4, dash: dashed ? "dash" : "solid" },
    marker: { size: [0, 4] },
    name,
    showlegend: false,
  } as Plotly.Data;
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
