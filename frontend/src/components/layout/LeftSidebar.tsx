import type { ReactNode } from "react";

interface Props {
  children: ReactNode;
}

export default function LeftSidebar({ children }: Props) {
  return (
    <aside className="w-64 shrink-0 border-r border-zinc-800 bg-zinc-950 p-4 overflow-y-auto">
      <h3 className="text-[10px] uppercase tracking-wider text-zinc-500 mb-4 font-semibold">
        Controls
      </h3>
      <div className="flex flex-col gap-4">{children}</div>
    </aside>
  );
}
