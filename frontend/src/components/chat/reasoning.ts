const labels: Record<string, string> = {
  auto: "Авто",
  none: "Без рассуждений",
  minimal: "Минимум",
  low: "Низкий",
  medium: "Средний",
  high: "Высокий",
  xhigh: "Максимум",
  max: "Максимум",
};

export function reasoningLabel(effort: string): string {
  return labels[effort] ?? effort;
}

/** Russian plural form for ``count``: one (1, 21), few (2–4, 22–24) or many. */
export function plural(count: number, one: string, few: string, many: string): string {
  const tens = count % 100;
  const units = count % 10;
  if (tens >= 11 && tens <= 14) return many;
  if (units === 1) return one;
  if (units >= 2 && units <= 4) return few;
  return many;
}

const MIN_SPEED_SECONDS = 0.2;

function number(value: number): string {
  return value.toLocaleString("ru-RU");
}

/**
 * Parts of the reasoning label: characters, tokens and tokens per second.
 * An estimated token count gets "≈". The speed needs a measurable span of reasoning: on a
 * shorter span the arrival of a single fragment changes it too much.
 */
export function reasoningStatsParts(
  chars: number,
  stats: { tokens: number | null; exact: boolean; seconds: number | null } | null | undefined,
): string[] {
  const parts = [`${number(chars)} ${plural(chars, "символ", "символа", "символов")}`];
  const tokens = stats?.tokens;
  if (tokens == null || tokens <= 0) return parts;
  parts.push(`${stats?.exact ? "" : "≈"}${number(tokens)} ${plural(tokens, "токен", "токена", "токенов")}`);
  const seconds = stats?.seconds;
  if (seconds != null && seconds >= MIN_SPEED_SECONDS) {
    const speed = tokens / seconds;
    parts.push(`${speed < 10 ? number(Math.round(speed * 10) / 10) : number(Math.round(speed))} ток/с`);
  }
  return parts;
}
