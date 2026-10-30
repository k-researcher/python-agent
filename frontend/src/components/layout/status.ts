// Status dot color mapping (UI spec §6):
// running — blue pulsing, awaiting_confirmation — yellow, completed — green,
// error — red, everything else — grey.

const STATUS_CLASS: Record<string, string> = {
  running: "status-dot--running",
  awaiting_confirmation: "status-dot--awaiting",
  completed: "status-dot--completed",
  error: "status-dot--error",
};

const STATUS_LABEL: Record<string, string> = {
  idle: "Готова",
  pending: "В очереди",
  starting: "Запуск",
  running: "Выполняется",
  awaiting_confirmation: "Ждёт подтверждения",
  completed: "Завершена",
  error: "Ошибка",
  stopped: "Остановлена",
  interrupted: "Прервана",
};

const AUDIT_LABEL: Record<string, string> = {
  started: "Идёт",
  completed: "Выполнено",
  error: "Ошибка",
  cancelled: "Отменено",
};

// Colour of a status pill; an unknown status stays grey.
const STATUS_TONE: Record<string, string> = {
  running: "var(--info)",
  started: "var(--info)",
  awaiting_confirmation: "var(--warning)",
  completed: "var(--success)",
  error: "var(--danger)",
};

export function statusDotClass(status: string): string {
  return STATUS_CLASS[status] ?? "status-dot--idle";
}

export function sessionStatusLabel(status: string): string {
  return Object.hasOwn(STATUS_LABEL, status) ? STATUS_LABEL[status] : "Неизвестный статус";
}

/** Label of an outbound request status; an unknown status is shown as it is. */
export function auditStatusLabel(status: string): string {
  return Object.hasOwn(AUDIT_LABEL, status) ? AUDIT_LABEL[status] : status;
}

/** Inline style that gives a status pill its colour. */
export function statusTone(status: string): Record<string, string> {
  return Object.hasOwn(STATUS_TONE, status) ? { "--status-color": STATUS_TONE[status] } : {};
}
