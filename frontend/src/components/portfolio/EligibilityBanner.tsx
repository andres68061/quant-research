/**
 * Compact banner reporting joint-history coverage for a selection.
 *
 * Shows ``Joint history: N overlapping days (need M).`` with a colour cue.
 * If the selection has fewer than 2 symbols, renders nothing.
 */
import type { PortfolioJointHistoryResponse } from "@/lib/types.ts";
import { cn } from "@/lib/utils.ts";

interface Props {
  selectedCount: number;
  joint: PortfolioJointHistoryResponse | undefined;
  isError?: boolean;
  /** Hint text for the not-eligible state. */
  hint?: string;
}

const DEFAULT_HINT = "Shorten the basket or move the start date earlier.";

export default function EligibilityBanner({
  selectedCount,
  joint,
  isError,
  hint = DEFAULT_HINT,
}: Props) {
  if (selectedCount < 2) return null;

  if (isError) {
    return (
      <div className="text-[10px] text-red-400 bg-red-950/50 border border-red-900 rounded px-2 py-1">
        Could not check joint history
      </div>
    );
  }

  if (!joint) return null;

  return (
    <div
      className={cn(
        "text-[10px] rounded px-2 py-1 border",
        joint.eligible
          ? "text-zinc-500 border-zinc-800 bg-zinc-900/40"
          : "text-amber-400 border-amber-900/60 bg-amber-950/30",
      )}
    >
      Joint history: {joint.joint_rows} overlapping days (need {joint.min_required}).
      {!joint.eligible && ` ${hint}`}
    </div>
  );
}
