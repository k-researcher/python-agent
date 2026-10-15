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
