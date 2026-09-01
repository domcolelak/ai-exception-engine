/** Shared formatting helpers. Operations people read hours, not seconds. */

export function duration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) return "0 min";
  const minutes = seconds / 60;
  if (minutes < 60) return `${minutes.toFixed(0)} min`;
  const hours = minutes / 60;
  if (hours < 48) return `${hours.toFixed(1)} h`;
  return `${(hours / 24).toFixed(1)} d`;
}

export function hours(value: number): string {
  return `${value.toFixed(1)} h`;
}

export function percent(value: number, digits = 0): string {
  return `${(value * 100).toFixed(digits)} %`;
}

export function euro(value: number): string {
  return new Intl.NumberFormat("en-GB", {
    style: "currency",
    currency: "EUR",
    maximumFractionDigits: 0,
  }).format(value);
}

export function count(value: number): string {
  return new Intl.NumberFormat("en-GB").format(value);
}

export function dateTime(value: string | null): string {
  if (!value) return "never";
  return new Date(value).toLocaleString("en-GB", {
    dateStyle: "medium",
    timeStyle: "short",
  });
}

export const severityOrder = ["critical", "high", "medium", "low"] as const;

export function severityClass(severity: string): string {
  return `badge badge-${severity}`;
}
