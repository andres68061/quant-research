import type { ReactNode } from "react";

import TopBar from "./TopBar.tsx";

interface Props {
  left?: ReactNode;
  right?: ReactNode;
  bottom?: ReactNode;
  children: ReactNode;
}

export default function AppLayout({ left, right, bottom, children }: Props) {
  return (
    <div className="flex flex-col h-screen overflow-hidden">
      <TopBar />
      <div className="flex flex-1 min-h-0">
        {left}
        <main className="flex-1 flex flex-col min-w-0 overflow-hidden">
          <div className="flex-1 overflow-y-auto p-4">{children}</div>
          {bottom}
        </main>
        {right}
      </div>
    </div>
  );
}
