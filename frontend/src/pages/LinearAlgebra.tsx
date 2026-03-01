import { useState } from "react";

import AppLayout from "@/components/layout/AppLayout.tsx";
import LeftSidebar from "@/components/layout/LeftSidebar.tsx";
import CovarianceMatrix from "@/components/linalg/CovarianceMatrix.tsx";
import FactorModels from "@/components/linalg/FactorModels.tsx";
import MatrixMultiply2D from "@/components/linalg/MatrixMultiply2D.tsx";
import MatrixProperties from "@/components/linalg/MatrixProperties.tsx";
import PortfolioVariance from "@/components/linalg/PortfolioVariance.tsx";
import Transform3D from "@/components/linalg/Transform3D.tsx";

const MODES = [
  "2D Matrix Multiply",
  "3D Transformations",
  "Portfolio Variance",
  "Covariance Matrix",
  "Factor Models",
  "Matrix Properties",
] as const;

type Mode = (typeof MODES)[number];

export default function LinearAlgebra() {
  const [mode, setMode] = useState<Mode>("2D Matrix Multiply");

  return (
    <AppLayout
      left={
        <LeftSidebar>
          <div>
            <label className="block text-[10px] uppercase tracking-wider text-zinc-500 mb-1">
              Topic
            </label>
            <select
              value={mode}
              onChange={(e) => setMode(e.target.value as Mode)}
              className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5 text-xs text-zinc-200 font-mono"
            >
              {MODES.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>

          <div className="border-t border-zinc-800 pt-3 mt-1 text-[10px] text-zinc-500 leading-relaxed">
            Interactive linear algebra visualizations with quant finance applications.
          </div>
        </LeftSidebar>
      }
    >
      <ModeView mode={mode} />
    </AppLayout>
  );
}

function ModeView({ mode }: { mode: Mode }) {
  switch (mode) {
    case "2D Matrix Multiply":
      return <MatrixMultiply2D />;
    case "3D Transformations":
      return <Transform3D />;
    case "Portfolio Variance":
      return <PortfolioVariance />;
    case "Covariance Matrix":
      return <CovarianceMatrix />;
    case "Factor Models":
      return <FactorModels />;
    case "Matrix Properties":
      return <MatrixProperties />;
  }
}
