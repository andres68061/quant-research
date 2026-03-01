import type { ReactNode } from "react";

interface Props {
  children: ReactNode;
}

export default function RightSidebar({ children }: Props) {
  return (
    <aside className="w-72 shrink-0 border-l border-zinc-800 bg-zinc-950 p-4 overflow-y-auto">
      <h3 className="text-[10px] uppercase tracking-wider text-zinc-500 mb-4 font-semibold">
        Metrics
      </h3>
      <div className="flex flex-col gap-3">{children}</div>
    </aside>
  );
}
