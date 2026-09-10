import type { Caveat } from "@/lib/types.ts";
import { cn } from "@/lib/utils.ts";

interface Props {
  methodology: Record<string, string>;
  caveats: Caveat[];
}

/** Methodology + registry caveats for any surface that shows a research number. */
export default function DisclosureBlock({ methodology, caveats }: Props) {
  return (
    <details className="text-[11px] text-zinc-500 border border-zinc-800 rounded p-2">
      <summary className="cursor-pointer text-zinc-400">
        Methodology &amp; {caveats.length} disclosed caveats
      </summary>
      <dl className="mt-2 space-y-1.5">
        {Object.entries(methodology).map(([key, detail]) => (
          <div key={key} className="flex gap-2">
            <dt className="w-32 shrink-0 uppercase tracking-wider text-[9px] text-zinc-600 pt-0.5">
              {key.replace(/_/g, " ")}
            </dt>
            <dd className="text-zinc-400">{detail}</dd>
          </div>
        ))}
      </dl>
      <div className="mt-3 space-y-2 border-t border-zinc-800 pt-2">
        {caveats.map((caveat) => (
          <div key={caveat.id}>
            <div className="flex items-center gap-1.5">
              <span
                className={cn(
                  "px-1 py-0.5 rounded text-[9px] uppercase",
                  caveat.severity === "high"
                    ? "text-red-400 bg-red-950/40"
                    : caveat.severity === "medium"
                      ? "text-amber-400 bg-amber-950/30"
                      : "text-zinc-400 bg-zinc-800/60",
                )}
              >
                {caveat.severity}
              </span>
              <span className="text-zinc-300">{caveat.title}</span>
            </div>
            <p className="text-zinc-500 mt-0.5 leading-relaxed">{caveat.detail}</p>
          </div>
        ))}
      </div>
    </details>
  );
}
