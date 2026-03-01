import { motion } from "framer-motion";

import type { PerformanceMetrics } from "@/lib/types.ts";
import { fmtPct, fmtRatio } from "@/lib/utils.ts";

import KPICard from "./KPICard.tsx";

interface Props {
  metrics: PerformanceMetrics;
}

const container = {
  hidden: {},
  show: { transition: { staggerChildren: 0.06 } },
};

const card = {
  hidden: { opacity: 0, x: 8 },
  show: { opacity: 1, x: 0, transition: { duration: 0.2, ease: "easeOut" } },
};

export default function KPIPanel({ metrics }: Props) {
  const items: { label: string; value: string; accent?: "positive" | "negative" }[] = [
    {
      label: "Sharpe",
      value: fmtRatio(metrics.sharpe_ratio),
      accent: metrics.sharpe_ratio >= 0 ? "positive" : "negative",
    },
    {
      label: "Sortino",
      value: fmtRatio(metrics.sortino_ratio),
      accent: metrics.sortino_ratio >= 0 ? "positive" : "negative",
    },
    {
      label: "Annual Return",
      value: fmtPct(metrics.annualized_return),
      accent: metrics.annualized_return >= 0 ? "positive" : "negative",
    },
    {
      label: "Max Drawdown",
      value: fmtPct(metrics.max_drawdown),
      accent: "negative",
    },
    {
      label: "Volatility",
      value: fmtPct(metrics.annualized_volatility),
    },
    {
      label: "Calmar",
      value: fmtRatio(metrics.calmar_ratio),
      accent: metrics.calmar_ratio >= 0 ? "positive" : "negative",
    },
  ];

  return (
    <motion.div
      className="flex flex-col gap-2"
      variants={container}
      initial="hidden"
      animate="show"
      key={JSON.stringify(metrics)}
    >
      {items.map((item) => (
        <motion.div key={item.label} variants={card}>
          <KPICard {...item} />
        </motion.div>
      ))}
    </motion.div>
  );
}
