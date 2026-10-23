export const SESSION_MODES = ["dev", "ask", "arch", "pm", "qa", "devops", "cybersec", "analyst"] as const;
export type SessionMode = (typeof SESSION_MODES)[number];
export const MODE_LABELS: Record<SessionMode, string> = {
  dev: "Разработчик", ask: "Консультант", arch: "Архитектор", pm: "Руководитель проекта",
  qa: "Тестировщик", devops: "DevOps-инженер", cybersec: "Безопасность", analyst: "Аналитик",
};

export function modeLabel(mode: string | null | undefined): string {
  // A list check, not `in`: `in` also finds inherited names such as "toString".
  if (mode && (SESSION_MODES as readonly string[]).includes(mode)) {
    return MODE_LABELS[mode as SessionMode];
  }
  return mode || MODE_LABELS.dev;
}
