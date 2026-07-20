import { cn } from "@/lib/utils.ts";

interface Props {
  signal: "long" | "short" | "flat" | "long_short" | string;
}

export default function SignalBadge({ signal }: Props) {
  const normalized = signal.toLowerCase();
  const isLongShort = normalized === "long_short" || normalized === "l/s";
  const label = isLongShort ? "L/S" : signal;
  return (
    <span
      className={cn(
        "inline-flex items-center px-2 py-0.5 rounded text-[10px] font-mono uppercase tracking-wider",
        normalized === "long" && "bg-emerald-950 text-emerald-400 border border-emerald-800",
        normalized === "short" && "bg-red-950 text-red-400 border border-red-800",
        isLongShort && "bg-blue-950 text-blue-400 border border-blue-800",
        normalized !== "long" &&
          normalized !== "short" &&
          !isLongShort &&
          "bg-zinc-900 text-zinc-500 border border-zinc-800",
      )}
    >
      {label}
    </span>
  );
}
