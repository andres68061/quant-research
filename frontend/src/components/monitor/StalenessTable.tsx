import type { StalenessBoardRow, StaleStatus } from "@/lib/types.ts";
import { cn } from "@/lib/utils.ts";

interface Props {
  rows: StalenessBoardRow[];
  asOf: string;
  snapshots?: Record<string, string>;
  onSelect?: (id: string) => void;
}

const STATUS_CLASS: Record<StaleStatus, string> = {
  fresh: "text-emerald-400",
  late: "text-amber-400",
  stale: "text-red-400",
  empty: "text-zinc-500",
};

/** Every monitored series with its last observation and freshness verdict, worst first. */
export default function StalenessTable({ rows, asOf, snapshots, onSelect }: Props) {
  const problems = rows.filter((r) => r.status !== "fresh");
  const frozen = Object.entries(snapshots ?? {});
  return (
    <div className="text-[11px]">
      <div className="flex items-baseline justify-between mb-2">
        <span className="text-[10px] uppercase tracking-wider text-zinc-500">
          Freshness · as of {asOf}
          {frozen.map(([source, date]) => (
            <span key={source} className="ml-2 text-amber-400 normal-case tracking-normal">
              {source.toUpperCase()} frozen at snapshot {date}
            </span>
          ))}
        </span>
        <span className={cn("font-mono", problems.length ? "text-amber-400" : "text-emerald-400")}>
          {problems.length ? `${problems.length} need attention` : `all ${rows.length} fresh`}
        </span>
      </div>
      <div className="overflow-x-auto max-h-56 overflow-y-auto">
        <table className="w-full font-mono tabular-nums">
          <thead className="text-[9px] uppercase tracking-wider text-zinc-500 sticky top-0 bg-zinc-900">
            <tr>
              <th className="text-left px-2 py-1">series</th>
              <th className="text-left px-2 py-1">group</th>
              <th className="text-left px-2 py-1">cadence</th>
              <th className="text-right px-2 py-1">last</th>
              <th className="text-right px-2 py-1">age d</th>
              <th className="text-right px-2 py-1">late after</th>
              <th className="text-right px-2 py-1">status</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr
                key={r.series_id}
                className={cn(
                  "border-t border-zinc-800/60",
                  onSelect && "cursor-pointer hover:bg-zinc-800/40",
                )}
                onClick={() => onSelect?.(r.series_id)}
              >
                <td className="px-2 py-0.5 text-zinc-300">
                  {r.series_id} <span className="text-zinc-600">{r.name}</span>
                </td>
                <td className="px-2 py-0.5 text-zinc-500">{r.group}</td>
                <td className="px-2 py-0.5 text-zinc-500">{r.frequency}</td>
                <td className="px-2 py-0.5 text-right text-zinc-400">{r.last_date ?? "–"}</td>
                <td className="px-2 py-0.5 text-right text-zinc-400">
                  {r.days_since_last ?? "–"}
                  {r.snapshot_as_of ? <span className="text-zinc-600"> ❄</span> : null}
                </td>
                <td className="px-2 py-0.5 text-right text-zinc-600">{r.expected_max_gap_days}</td>
                <td className={cn("px-2 py-0.5 text-right", STATUS_CLASS[r.status])}>{r.status}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
