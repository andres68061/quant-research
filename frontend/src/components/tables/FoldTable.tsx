import type { FoldResult } from "@/lib/types.ts";

interface Props {
  folds: FoldResult[];
}

export default function FoldTable({ folds }: Props) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs font-mono">
        <thead>
          <tr className="text-[10px] uppercase tracking-wider text-zinc-500 border-b border-zinc-800">
            <th className="text-left py-2 px-2">Fold</th>
            <th className="text-left py-2 px-2">Train</th>
            <th className="text-left py-2 px-2">Test</th>
            <th className="text-right py-2 px-2">Train Size</th>
            <th className="text-right py-2 px-2">Test Size</th>
            <th className="text-right py-2 px-2">Accuracy</th>
          </tr>
        </thead>
        <tbody>
          {folds.map((f) => (
            <tr key={f.fold} className="border-b border-zinc-900 hover:bg-zinc-900/50 transition-colors">
              <td className="py-1.5 px-2 text-zinc-400">{f.fold}</td>
              <td className="py-1.5 px-2 text-zinc-500">
                {f.train_start} — {f.train_end}
              </td>
              <td className="py-1.5 px-2 text-zinc-500">
                {f.test_start} — {f.test_end}
              </td>
              <td className="py-1.5 px-2 text-right tabular-nums">{f.train_size}</td>
              <td className="py-1.5 px-2 text-right tabular-nums">{f.test_size}</td>
              <td className="py-1.5 px-2 text-right tabular-nums text-zinc-200">
                {(f.accuracy * 100).toFixed(1)}%
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
