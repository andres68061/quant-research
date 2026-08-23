import { useState } from "react";
import { useQuery } from "@tanstack/react-query";

import { api } from "@/lib/api.ts";
import { cn } from "@/lib/utils.ts";
import type { WatchdogCheck, WatchdogStatus } from "@/lib/types.ts";

const POLL_MS = 5 * 60 * 1000;

const TONE: Record<WatchdogStatus["status"], { bar: string; dot: string; label: string }> = {
  ok: { bar: "hidden", dot: "bg-emerald-400", label: "Data OK" },
  warning: {
    bar: "border-amber-500/40 bg-amber-500/10 text-amber-200",
    dot: "bg-amber-400",
    label: "Data warning",
  },
  error: {
    bar: "border-red-500/50 bg-red-500/10 text-red-200",
    dot: "bg-red-400",
    label: "Data problem",
  },
  unknown: {
    bar: "border-zinc-700 bg-zinc-900 text-zinc-400",
    dot: "bg-zinc-500",
    label: "Monitor not running",
  },
};

/**
 * Surfaces the scheduled watchdog verdict wherever the user already is.
 *
 * A silent banner when everything is fine; a persistent one when it is not.
 * This is the channel that replaces "read the log file" — the same verdict also
 * fires a desktop notification at the time it happens.
 */
export default function WatchdogBanner() {
  const [expanded, setExpanded] = useState(false);
  const { data } = useQuery({
    queryKey: ["watchdog"],
    queryFn: api.getWatchdogStatus,
    refetchInterval: POLL_MS,
    staleTime: POLL_MS,
  });

  if (!data || data.status === "ok") return null;

  const tone = TONE[data.status] ?? TONE.unknown;
  const failing = data.checks.filter((c) => c.status !== "ok");

  return (
    <div className={cn("border-b px-4 py-1.5 text-xs shrink-0", tone.bar)}>
      <button
        onClick={() => setExpanded((v) => !v)}
        className="flex items-center gap-2 w-full text-left cursor-pointer"
      >
        <span className={cn("w-1.5 h-1.5 rounded-full shrink-0", tone.dot)} />
        <span className="font-medium">{tone.label}</span>
        <span className="text-zinc-400 truncate">{data.summary}</span>
        {failing.length > 0 && (
          <span className="ml-auto text-zinc-500 shrink-0">
            {expanded ? "hide" : `${failing.length} check${failing.length === 1 ? "" : "s"}`}
          </span>
        )}
      </button>

      {expanded && failing.length > 0 && (
        <ul className="mt-2 space-y-1 border-t border-zinc-800 pt-2">
          {failing.map((check: WatchdogCheck) => (
            <li key={check.name} className="flex gap-2">
              <span
                className={cn(
                  "font-mono text-[10px] shrink-0 w-44 truncate",
                  check.status === "error" ? "text-red-400" : "text-amber-400",
                )}
              >
                {check.name}
              </span>
              <span className="text-zinc-400">{check.detail}</span>
            </li>
          ))}
          {data.generated_at && (
            <li className="text-[10px] text-zinc-600 pt-1">Last checked {data.generated_at}</li>
          )}
        </ul>
      )}
    </div>
  );
}
