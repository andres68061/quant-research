import { useEffect, useRef } from "react";

import { cn } from "@/lib/utils.ts";
import { useReplayStore } from "@/stores/replayStore.ts";

const speeds = [1, 5, 20];

export default function ReplayControls() {
  const { frameIndex, playing, speed, totalFrames, setFrame, play, pause, setSpeed, advance } =
    useReplayStore();
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (playing && totalFrames > 0) {
      intervalRef.current = setInterval(advance, 1000 / speed);
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [playing, speed, totalFrames, advance]);

  if (totalFrames === 0) return null;

  return (
    <div className="flex items-center gap-3 bg-zinc-900 border border-zinc-800 rounded px-3 py-2">
      <button
        onClick={playing ? pause : play}
        className="text-xs font-mono text-zinc-300 hover:text-white transition-colors w-10 cursor-pointer"
      >
        {playing ? "⏸" : "▶"}
      </button>

      <input
        type="range"
        min={0}
        max={totalFrames - 1}
        value={frameIndex}
        onChange={(e) => setFrame(Number(e.target.value))}
        className="flex-1 accent-blue-500 h-1 cursor-pointer"
      />

      <span className="text-[10px] font-mono text-zinc-500 tabular-nums w-16 text-right">
        {frameIndex + 1}/{totalFrames}
      </span>

      <div className="flex gap-1">
        {speeds.map((s) => (
          <button
            key={s}
            onClick={() => setSpeed(s)}
            className={cn(
              "text-[10px] font-mono px-1.5 py-0.5 rounded transition-colors cursor-pointer",
              speed === s
                ? "bg-zinc-700 text-zinc-200"
                : "text-zinc-600 hover:text-zinc-400",
            )}
          >
            {s}x
          </button>
        ))}
      </div>
    </div>
  );
}
