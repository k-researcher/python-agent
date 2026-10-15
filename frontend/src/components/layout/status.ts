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

export function statusDotClass(status: string): string {
  return STATUS_CLASS[status] ?? "status-dot--idle";
}

export function sessionStatusLabel(status: string): string {
  return Object.hasOwn(STATUS_LABEL, status) ? STATUS_LABEL[status] : "Неизвестный статус";
}
