// Server time stamps are UTC. The UI shows them in the viewer's time zone.

export interface TimeFormatOptions {
  seconds?: boolean;
  /** Reference moment for "today"; tests set it. */
  now?: Date;
  /** IANA zone; the browser zone when not set. Tests set it. */
  timeZone?: string;
}

const ZONE_SUFFIX = /(Z|[+-]\d\d:?\d\d)$/i;

function parseServerTime(raw: string): Date | null {
  const value = raw.trim().replace(" ", "T");
  // A value without a zone is UTC: the server never sends local time.
  const date = new Date(ZONE_SUFFIX.test(value) ? value : `${value}Z`);
  return Number.isNaN(date.getTime()) ? null : date;
}

function parts(date: Date, timeZone: string | undefined, seconds: boolean): Record<string, string> {
  const format = new Intl.DateTimeFormat("ru-RU", {
    timeZone,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    ...(seconds ? { second: "2-digit" } : {}),
    hourCycle: "h23",
  });
  return Object.fromEntries(format.formatToParts(date).map((part) => [part.type, part.value]));
}

/** "19:09" today, "28.10 19:09" on another day, "31.12.2025 17:00" in another year. */
export function formatLocalTime(raw: string | null | undefined, options: TimeFormatOptions = {}): string {
  if (!raw) return "";
  const date = parseServerTime(raw);
  if (date === null) return "";
  const seconds = options.seconds ?? false;
  const value = parts(date, options.timeZone, seconds);
  const today = parts(options.now ?? new Date(), options.timeZone, false);
  const time = seconds ? `${value.hour}:${value.minute}:${value.second}` : `${value.hour}:${value.minute}`;
  if (value.year !== today.year) return `${value.day}.${value.month}.${value.year} ${time}`;
  if (value.month !== today.month || value.day !== today.day) return `${value.day}.${value.month} ${time}`;
  return time;
}
