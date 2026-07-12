export function fmtPct(v: number, decimals = 2): string {
  return `${(v * 100).toFixed(decimals)}%`;
}

export function fmtRatio(v: number, decimals = 2): string {
  return v.toFixed(decimals);
}

export function fmtInt(v: number): string {
  return Math.round(v).toLocaleString();
}

export function cn(...classes: (string | false | undefined | null)[]): string {
  return classes.filter(Boolean).join(" ");
}
