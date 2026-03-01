import katex from "katex";
import { useMemo } from "react";

import { cn } from "@/lib/utils.ts";

interface Props {
  math: string;
  display?: boolean;
  className?: string;
}

export default function TeX({ math, display = false, className }: Props) {
  const html = useMemo(
    () => katex.renderToString(math, { displayMode: display, throwOnError: false }),
    [math, display],
  );

  return display ? (
    <div
      className={cn("overflow-x-auto py-2", className)}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  ) : (
    <span className={className} dangerouslySetInnerHTML={{ __html: html }} />
  );
}
