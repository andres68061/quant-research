import { useMutation, useQuery } from "@tanstack/react-query";
import { useState } from "react";

import KPICard from "@/components/cards/KPICard.tsx";
import ConfusionMatrix from "@/components/charts/ConfusionMatrix.tsx";
import FeatureImportance from "@/components/charts/FeatureImportance.tsx";
import FoldAccuracy from "@/components/charts/FoldAccuracy.tsx";
import RunButton from "@/components/controls/RunButton.tsx";
import AppLayout from "@/components/layout/AppLayout.tsx";
import BottomPanel from "@/components/layout/BottomPanel.tsx";
import LeftSidebar from "@/components/layout/LeftSidebar.tsx";
import RightSidebar from "@/components/layout/RightSidebar.tsx";
import FoldTable from "@/components/tables/FoldTable.tsx";
import { api } from "@/lib/api.ts";
import type { MLStrategyResponse } from "@/lib/types.ts";
import { cn, fmtPct } from "@/lib/utils.ts";

type Tab = "diagnostics" | "folds";

export default function MLAlphaReplay() {
  const [symbol, setSymbol] = useState("GLD");
  const [modelType, setModelType] = useState("xgboost");
  const [trainDays, setTrainDays] = useState(126);
  const [testDays, setTestDays] = useState(5);
  const [maxSplits, setMaxSplits] = useState(30);
  const [activeTab, setActiveTab] = useState<Tab>("diagnostics");
  const [symbolFilter, setSymbolFilter] = useState("");

  const assetsQuery = useQuery({ queryKey: ["assets"], queryFn: api.listAssets });
  const assets = assetsQuery.data?.assets ?? [];
  const filteredAssets = symbolFilter
    ? assets.filter((a) => a.symbol.includes(symbolFilter.toUpperCase()))
    : assets;

  const mlStrategy = useMutation({ mutationFn: api.runMLStrategy });

  const result: MLStrategyResponse | undefined = mlStrategy.data;
  const wf = result?.walkforward;
  const fi = result?.feature_importance;
  const cm = wf?.confusion_matrix;

  const handleRun = () => {
    mlStrategy.mutate({
      symbol,
      model_type: modelType,
      initial_train_days: trainDays,
      test_days: testDays,
      max_splits: maxSplits,
    });
  };

  return (
    <AppLayout
      left={
        <LeftSidebar>
          <Field label="Symbol">
            <input
              type="text"
              placeholder="Search..."
              value={symbolFilter}
              onChange={(e) => setSymbolFilter(e.target.value)}
              className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5 text-xs text-zinc-200 font-mono mb-1"
            />
            <select
              value={symbol}
              onChange={(e) => setSymbol(e.target.value)}
              size={6}
              className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-0.5 text-xs text-zinc-200 font-mono"
            >
              {filteredAssets.map((a) => (
                <option key={a.symbol} value={a.symbol}>
                  {a.symbol} ({a.type})
                </option>
              ))}
            </select>
          </Field>

          <Field label="Model">
            <select
              value={modelType}
              onChange={(e) => setModelType(e.target.value)}
              className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5 text-xs text-zinc-200 font-mono"
            >
              <option value="xgboost">XGBoost</option>
              <option value="random_forest">Random Forest</option>
              <option value="logistic">Logistic Regression</option>
              <option value="lstm">LSTM</option>
            </select>
          </Field>

          <Field label="Initial Train Days">
            <input
              type="number"
              value={trainDays}
              onChange={(e) => setTrainDays(Number(e.target.value))}
              min={30}
              className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5 text-xs text-zinc-200 font-mono tabular-nums"
            />
          </Field>

          <Field label="Test Days">
            <input
              type="number"
              value={testDays}
              onChange={(e) => setTestDays(Number(e.target.value))}
              min={1}
              className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5 text-xs text-zinc-200 font-mono tabular-nums"
            />
          </Field>

          <Field label="Max Splits">
            <input
              type="number"
              value={maxSplits}
              onChange={(e) => setMaxSplits(Number(e.target.value))}
              min={5}
              className="w-full bg-zinc-900 border border-zinc-800 rounded px-2 py-1.5 text-xs text-zinc-200 font-mono tabular-nums"
            />
          </Field>

          <RunButton onClick={handleRun} loading={mlStrategy.isPending} label="Run ML Strategy" />

          {mlStrategy.isError && (
            <div className="text-[10px] text-red-400 bg-red-950/50 border border-red-900 rounded px-2 py-1">
              {(mlStrategy.error as Error).message}
            </div>
          )}
        </LeftSidebar>
      }
      right={
        <RightSidebar>
          {wf ? (
            <>
              <MetricCards wf={result!.walkforward} meta={result!.metadata} />
            </>
          ) : (
            <div className="text-xs text-zinc-600">Run a strategy to see metrics</div>
          )}
        </RightSidebar>
      }
      bottom={
        wf ? (
          <BottomPanel>
            <FoldTable folds={wf.folds} />
          </BottomPanel>
        ) : undefined
      }
    >
      {wf ? (
        <div className="flex flex-col h-full">
          <TabBar active={activeTab} onChange={setActiveTab} />
          <div className="flex-1 min-h-0 overflow-y-auto">
            {activeTab === "diagnostics" && (
              <div className="grid grid-cols-2 gap-4">
                {fi && fi.length > 0 && <FeatureImportance data={fi} height={380} />}
                {cm && <ConfusionMatrix data={cm} height={380} />}
                {(!fi || fi.length === 0) && !cm && (
                  <div className="col-span-2 text-xs text-zinc-600 text-center py-8">
                    No diagnostics available for this model type
                  </div>
                )}
              </div>
            )}
            {activeTab === "folds" && (
              <FoldAccuracy folds={wf.folds} overallAccuracy={wf.overall_accuracy} height={360} />
            )}
          </div>
        </div>
      ) : (
        <EmptyState />
      )}
    </AppLayout>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="text-[10px] uppercase tracking-wider text-zinc-500 mb-1 block">
        {label}
      </label>
      {children}
    </div>
  );
}

function TabBar({ active, onChange }: { active: Tab; onChange: (t: Tab) => void }) {
  const tabs: { id: Tab; label: string }[] = [
    { id: "diagnostics", label: "Diagnostics" },
    { id: "folds", label: "Fold Accuracy" },
  ];

  return (
    <div className="flex gap-1 mb-4 border-b border-zinc-800 pb-2">
      {tabs.map((t) => (
        <button
          key={t.id}
          onClick={() => onChange(t.id)}
          className={cn(
            "px-3 py-1 text-xs font-medium rounded-t transition-colors cursor-pointer",
            active === t.id
              ? "bg-zinc-800 text-zinc-200 border-b-2 border-blue-500"
              : "text-zinc-500 hover:text-zinc-300",
          )}
        >
          {t.label}
        </button>
      ))}
    </div>
  );
}

function EmptyState() {
  return (
    <div className="flex items-center justify-center h-full">
      <div className="text-center">
        <div className="text-zinc-600 text-sm">Configure and run an ML strategy</div>
        <div className="text-zinc-700 text-xs mt-1">
          Walk-forward validation results will appear here
        </div>
      </div>
    </div>
  );
}

function MetricCards({
  wf,
  meta,
}: {
  wf: MLStrategyResponse["walkforward"];
  meta: MLStrategyResponse["metadata"];
}) {
  const items: { label: string; value: string; accent?: "positive" | "negative" }[] = [
    {
      label: "Model",
      value: wf.model_type,
    },
    {
      label: "Accuracy",
      value: fmtPct(wf.overall_accuracy),
      accent: wf.overall_accuracy > 0.5 ? "positive" : "negative",
    },
    {
      label: "Precision",
      value: fmtPct(wf.overall_precision ?? 0),
    },
    {
      label: "Recall",
      value: fmtPct(wf.overall_recall ?? 0),
    },
    {
      label: "F1 Score",
      value: fmtPct(wf.overall_f1 ?? 0),
    },
    {
      label: "ROC AUC",
      value: wf.overall_roc_auc != null ? fmtPct(wf.overall_roc_auc) : "N/A",
    },
    {
      label: "Folds",
      value: String(wf.n_splits),
    },
  ];

  return (
    <>
      {meta.symbol && (
        <div className="text-[10px] font-mono text-zinc-500 bg-zinc-900 border border-zinc-800 rounded px-2 py-1 mb-1">
          {meta.symbol} | {meta.final_rows ?? "?"} rows | {meta.total_features ?? "?"} features
        </div>
      )}
      <div className="flex flex-col gap-2">
        {items.map((item) => (
          <KPICard key={item.label} {...item} />
        ))}
      </div>
    </>
  );
}
