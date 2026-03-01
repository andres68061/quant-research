import { cn } from "@/lib/utils.ts";

interface Props {
  signal: "long" | "short" | "flat" | string;
}

export default function SignalBadge({ signal }: Props) {
  const normalized = signal.toLowerCase();
  return (
    <span
      className={cn(
        "inline-flex items-center px-2 py-0.5 rounded text-[10px] font-mono uppercase tracking-wider",
        normalized === "long" && "bg-emerald-950 text-emerald-400 border border-emerald-800",
        normalized === "short" && "bg-red-950 text-red-400 border border-red-800",
        normalized !== "long" &&
          normalized !== "short" &&
          "bg-zinc-900 text-zinc-500 border border-zinc-800",
      )}
    >
      {signal}
    </span>
  );
}
