import { cn } from "@/lib/utils.ts";

interface Props {
  label: string;
  value: string;
  accent?: "positive" | "negative" | "neutral";
}

export default function KPICard({ label, value, accent }: Props) {
  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded px-3 py-2">
      <div className="text-[10px] uppercase tracking-wider text-zinc-500 mb-1">{label}</div>
      <div
        className={cn(
          "text-xl font-mono tabular-nums leading-tight",
          accent === "positive" && "text-emerald-400",
          accent === "negative" && "text-red-400",
          accent === "neutral" && "text-blue-400",
          !accent && "text-zinc-100",
        )}
      >
        {value}
      </div>
    </div>
  );
}
