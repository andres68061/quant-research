import type { ReactNode } from "react";

interface Props {
  children: ReactNode;
}

export default function BottomPanel({ children }: Props) {
  return (
    <div className="border-t border-zinc-800 bg-zinc-950 p-4 overflow-x-auto shrink-0 max-h-72 overflow-y-auto">
      {children}
    </div>
  );
}
