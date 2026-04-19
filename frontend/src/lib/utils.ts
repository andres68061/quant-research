export function cn(...values: Array<string | false | null | undefined>): string {
  return values.filter(Boolean).join(" ");
}

export function fmtPct(value: number, digits = 2): string {
  return `${(value * 100).toFixed(digits)}%`;
}

export function fmtRatio(value: number, digits = 2): string {
  return value.toFixed(digits);
}

export function fmtInt(value: number): string {
  return Math.round(value).toLocaleString();
}
