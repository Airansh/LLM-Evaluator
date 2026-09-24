// Missing values render as "N/A" - never as 0 - so absent data can't be mistaken for a score.

export const NA = "N/A";

export function score(value: number | null | undefined): string {
  return value == null ? NA : value.toFixed(1);
}

export function pp(value: number | null | undefined): string {
  if (value == null) return NA;
  const sign = value > 0 ? "+" : value < 0 ? "−" : "±";
  return `${sign}${Math.abs(value).toFixed(1)} pp`;
}

export function ms(value: number | null | undefined): string {
  if (value == null) return NA;
  return value >= 1000 ? `${(value / 1000).toFixed(2)} s` : `${Math.round(value)} ms`;
}

export function usd(value: number | null | undefined): string {
  if (value == null) return "unavailable";
  if (value === 0) return "$0";
  if (value < 0.01) return `$${value.toPrecision(2)}`;
  return `$${value.toFixed(value < 1 ? 4 : 2)}`;
}

export function tokens(value: number | null | undefined): string {
  return value == null ? NA : value.toLocaleString("en-US");
}

export function dateTime(iso: string | null | undefined): string {
  if (!iso) return NA;
  // The backend stores UTC; SQLite drops the offset, so treat offset-less strings as UTC.
  const normalized = /[zZ]|[+-]\d\d:\d\d$/.test(iso) ? iso : `${iso}Z`;
  return new Date(normalized).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

export function modelLabel(provider: string, modelId: string): string {
  return `${provider}/${modelId}`;
}
