import { useSyncExternalStore } from "react";

import {
  getTestCount,
  resetTestCount,
  sidakTBar,
  subscribeTestCount,
} from "@/lib/testCounter.ts";

const TOOLTIP =
  "The Sidak-corrected bar for a family of N tests at family-wise alpha 0.05.\n\n" +
  "Caveats: a formal correction applies to a pre-specified family, not to a click " +
  "count; and overlapping backtests are correlated, which makes Sidak conservative. " +
  "Treat it as a reminder of how much you have searched, not as a p-value.";

/**
 * Shows how many backtests have been run this session and the bar that implies.
 *
 * See lib/testCounter.ts for the reasoning and the caveats.
 */
export default function TestCounter() {
  const count = useSyncExternalStore(subscribeTestCount, getTestCount, () => 0);

  if (count === 0) return null;

  return (
    <div className="text-[10px] text-zinc-500 leading-relaxed" title={TOOLTIP}>
      <span className="font-mono tabular-nums text-zinc-400">{count}</span> test
      {count === 1 ? "" : "s"} this session · significance bar{" "}
      <span className="font-mono tabular-nums text-amber-400">
        |t| ≥ {sidakTBar(count).toFixed(2)}
      </span>
      <button
        onClick={resetTestCount}
        className="ml-2 text-zinc-600 hover:text-zinc-400 cursor-pointer"
      >
        reset
      </button>
    </div>
  );
}
