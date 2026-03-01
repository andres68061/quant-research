import { useEffect, useRef, useState } from "react";

import { cn } from "@/lib/utils.ts";

interface Props {
  onClick: () => void;
  loading?: boolean;
  label?: string;
  disabled?: boolean;
}

export default function RunButton({
  onClick,
  loading = false,
  label = "Run",
  disabled = false,
}: Props) {
  const [elapsed, setElapsed] = useState(0);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (loading) {
      setElapsed(0);
      intervalRef.current = setInterval(() => setElapsed((s) => s + 1), 1000);
    } else if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [loading]);

  const fmtTime = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return m > 0 ? `${m}m ${sec}s` : `${sec}s`;
  };

  return (
    <div className="space-y-1.5">
      <button
        onClick={onClick}
        disabled={loading || disabled}
        className={cn(
          "w-full py-2 px-4 rounded text-xs font-semibold uppercase tracking-wider transition-all",
          loading
            ? "bg-zinc-800 text-zinc-500 cursor-wait"
            : disabled
              ? "bg-zinc-900 text-zinc-600 cursor-not-allowed"
              : "bg-blue-600 hover:bg-blue-500 text-white cursor-pointer",
        )}
      >
        {loading ? (
          <span className="flex items-center justify-center gap-2">
            <span className="w-3 h-3 border-2 border-zinc-500 border-t-transparent rounded-full animate-spin" />
            Running...
          </span>
        ) : (
          label
        )}
      </button>

      {loading && (
        <div className="space-y-1">
          <div className="h-1 w-full bg-zinc-900 rounded overflow-hidden">
            <div className="h-full bg-blue-500/60 rounded animate-pulse" style={{ width: "100%" }} />
          </div>
          <div className="text-[10px] font-mono text-zinc-600 text-center tabular-nums">
            {fmtTime(elapsed)}
          </div>
        </div>
      )}
    </div>
  );
}
