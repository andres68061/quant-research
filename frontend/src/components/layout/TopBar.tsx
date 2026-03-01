import { useState } from "react";
import { NavLink, useLocation } from "react-router-dom";

import { cn } from "@/lib/utils.ts";

interface NavGroup {
  label: string;
  items: { to: string; label: string }[];
}

const NAV_GROUPS: NavGroup[] = [
  {
    label: "Strategies",
    items: [
      { to: "/", label: "Portfolio" },
      { to: "/ml-alpha", label: "ML Alpha" },
      { to: "/momentum", label: "Momentum" },
      { to: "/replay", label: "Replay" },
    ],
  },
  {
    label: "Analytics",
    items: [
      { to: "/etf-optimizer", label: "ETF Optimizer" },
      { to: "/metals", label: "Metals" },
      { to: "/economic", label: "Economic" },
      { to: "/sectors", label: "Sectors" },
      { to: "/excluded-stocks", label: "Excluded" },
    ],
  },
  {
    label: "Reference",
    items: [
      { to: "/methodology", label: "Methodology" },
      { to: "/sharpe-limitations", label: "Sharpe Limits" },
      { to: "/linear-algebra", label: "Linear Algebra" },
    ],
  },
];

export default function TopBar() {
  const location = useLocation();

  return (
    <header className="flex items-center h-11 border-b border-zinc-800 bg-zinc-950 px-4 shrink-0">
      <span className="text-sm font-semibold text-zinc-300 mr-6 tracking-tight">
        Quant Analytics
      </span>

      <nav className="flex gap-3">
        {NAV_GROUPS.map((group) => (
          <NavDropdown key={group.label} group={group} pathname={location.pathname} />
        ))}
      </nav>

      <div className="ml-auto flex items-center gap-3">
        <span className="text-[10px] font-mono text-zinc-600 bg-zinc-900 px-2 py-0.5 rounded">
          LOCAL
        </span>
        <a
          href="http://localhost:8000/docs"
          target="_blank"
          rel="noopener noreferrer"
          className="text-[10px] text-zinc-600 hover:text-zinc-400 transition-colors"
        >
          API Docs
        </a>
      </div>
    </header>
  );
}

function NavDropdown({ group, pathname }: { group: NavGroup; pathname: string }) {
  const [open, setOpen] = useState(false);
  const isGroupActive = group.items.some(
    (item) =>
      item.to === pathname || (item.to !== "/" && pathname.startsWith(item.to)),
  );

  if (group.items.length === 1) {
    const item = group.items[0];
    return (
      <NavLink
        to={item.to}
        className={({ isActive }) =>
          cn(
            "px-2.5 py-1.5 text-xs font-medium rounded transition-colors",
            isActive
              ? "bg-zinc-800 text-zinc-100"
              : "text-zinc-500 hover:text-zinc-300 hover:bg-zinc-900",
          )
        }
      >
        {item.label}
      </NavLink>
    );
  }

  return (
    <div
      className="relative"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
    >
      <button
        className={cn(
          "px-2.5 py-1.5 text-xs font-medium rounded transition-colors cursor-pointer",
          isGroupActive
            ? "bg-zinc-800 text-zinc-100"
            : "text-zinc-500 hover:text-zinc-300 hover:bg-zinc-900",
        )}
      >
        {group.label}
        <span className="ml-1 text-zinc-600">&#9662;</span>
      </button>

      {open && (
        <div className="absolute top-full left-0 mt-0.5 bg-zinc-900 border border-zinc-800 rounded shadow-lg z-50 min-w-[140px] py-1">
          {group.items.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              onClick={() => setOpen(false)}
              className={({ isActive }) =>
                cn(
                  "block px-3 py-1.5 text-xs transition-colors",
                  isActive
                    ? "bg-zinc-800 text-zinc-100"
                    : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50",
                )
              }
            >
              {item.label}
            </NavLink>
          ))}
        </div>
      )}
    </div>
  );
}
