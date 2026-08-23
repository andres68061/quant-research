import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";

import AppLayout from "@/components/layout/AppLayout.tsx";
import { api } from "@/lib/api.ts";
import { cn } from "@/lib/utils.ts";
import type { GlossaryRegistryTerm } from "@/lib/types.ts";

/**
 * The single definition registry, rendered.
 *
 * Definitions come from the API rather than being written here, so this page,
 * the data-health page and any research note cannot drift apart on what a term
 * means. Cross-references are clickable, which is what makes it usable as a
 * reference rather than a wall.
 */
export default function Glossary() {
  const { data, isLoading } = useQuery({ queryKey: ["glossary"], queryFn: api.getGlossary });
  const [filter, setFilter] = useState("");
  const [category, setCategory] = useState<string | null>(null);
  const [focused, setFocused] = useState<string | null>(null);

  const terms = useMemo(() => {
    const all = data?.terms ?? [];
    const needle = filter.trim().toLowerCase();
    return all.filter(
      (t) =>
        (!category || t.category === category) &&
        (!needle ||
          t.term.toLowerCase().includes(needle) ||
          t.definition.toLowerCase().includes(needle)),
    );
  }, [data, filter, category]);

  const grouped = useMemo(() => {
    const map = new Map<string, GlossaryRegistryTerm[]>();
    for (const t of terms) {
      if (!map.has(t.category)) map.set(t.category, []);
      map.get(t.category)!.push(t);
    }
    return map;
  }, [terms]);

  return (
    <AppLayout>
      <div className="max-w-3xl mx-auto py-6 px-4 space-y-5">
        <header>
          <h1 className="text-lg font-semibold text-zinc-200 tracking-tight">Glossary</h1>
          <p className="text-xs text-zinc-500 mt-1">
            {data?.terms.length ?? 0} terms, defined so each one stands alone. This is the
            single registry — the data-health page and research notes read from it rather
            than restating definitions.
          </p>
        </header>

        <div className="flex gap-2 flex-wrap items-center">
          <input
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder="Filter terms…"
            className="flex-1 min-w-[180px] bg-zinc-900 border border-zinc-800 rounded px-3 py-1.5 text-xs text-zinc-200 focus:outline-none focus:border-zinc-600"
          />
          <button
            onClick={() => setCategory(null)}
            className={cn(
              "px-2 py-1 text-[11px] rounded border cursor-pointer",
              category === null
                ? "border-blue-500/50 bg-blue-500/10 text-blue-200"
                : "border-zinc-800 text-zinc-500 hover:text-zinc-300",
            )}
          >
            all
          </button>
          {(data?.categories ?? []).map((c) => (
            <button
              key={c.id}
              onClick={() => setCategory(c.id)}
              className={cn(
                "px-2 py-1 text-[11px] rounded border cursor-pointer",
                category === c.id
                  ? "border-blue-500/50 bg-blue-500/10 text-blue-200"
                  : "border-zinc-800 text-zinc-500 hover:text-zinc-300",
              )}
            >
              {c.id}
            </button>
          ))}
        </div>

        {isLoading && <div className="text-xs text-zinc-500">Loading…</div>}

        {(data?.categories ?? [])
          .filter((c) => grouped.has(c.id))
          .map((c) => (
            <section key={c.id}>
              <h2 className="text-sm font-medium text-zinc-300 border-b border-zinc-800 pb-1 mb-3">
                {c.label}
              </h2>
              <dl className="space-y-3">
                {(grouped.get(c.id) ?? []).map((entry) => (
                  <div
                    key={entry.term}
                    id={slug(entry.term)}
                    className={cn(
                      "scroll-mt-16 rounded px-2 py-1.5 -mx-2 transition-colors",
                      focused === entry.term && "bg-zinc-900",
                    )}
                  >
                    <dt className="text-xs text-zinc-100 font-mono">{entry.term}</dt>
                    <dd className="text-[11px] text-zinc-400 leading-relaxed mt-1">
                      {entry.definition}
                    </dd>
                    {entry.see_also.length > 0 && (
                      <dd className="mt-1 flex gap-1.5 flex-wrap items-baseline">
                        <span className="text-[10px] text-zinc-600">see also</span>
                        {entry.see_also.map((ref) => (
                          <a
                            key={ref}
                            href={`#${slug(ref)}`}
                            onClick={() => {
                              setCategory(null);
                              setFocused(ref);
                            }}
                            className="text-[10px] text-blue-400/80 hover:text-blue-300 font-mono cursor-pointer"
                          >
                            {ref}
                          </a>
                        ))}
                      </dd>
                    )}
                  </div>
                ))}
              </dl>
            </section>
          ))}

        {!isLoading && terms.length === 0 && (
          <div className="text-xs text-zinc-500">No terms match that filter.</div>
        )}
      </div>
    </AppLayout>
  );
}

function slug(term: string): string {
  return term
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
}
