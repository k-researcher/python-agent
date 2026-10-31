import { onScopeDispose, ref, watch, type Ref } from "vue";
import { api } from "../api/client";
import type { MonitorRow, ServerEvent } from "../api/types";

/** Load only while visible. Ignore replies from a closed tab or a previous login. */
export function useMonitor(enabled: Readonly<Ref<boolean>>) {
  const rows = ref<MonitorRow[]>([]);
  const loading = ref(false);
  const error = ref("");
  const stopping = ref<string[]>([]);
  let generation = 0;
  let flight: Promise<void> | null = null;
  let controller: AbortController | undefined;
  let dirty = false;
  let timer: ReturnType<typeof setInterval> | undefined;
  let debounce: ReturnType<typeof setTimeout> | undefined;

  function reset(): void {
    generation += 1;
    controller?.abort();
    controller = undefined;
    clearInterval(timer);
    clearTimeout(debounce);
    timer = debounce = undefined;
    flight = null;
    dirty = false;
    rows.value = [];
    error.value = "";
    loading.value = false;
    stopping.value = [];
  }

  function refresh(): Promise<void> {
    if (!enabled.value) return Promise.resolve();
    if (flight) {
      dirty = true;
      return flight;
    }
    const token = generation;
    const requestController = new AbortController();
    controller = requestController;
    loading.value = true;
    flight = (async () => {
      try {
        do {
          dirty = false;
          const result = await api.monitor(requestController.signal);
          if (token !== generation || !enabled.value) return;
          if (!dirty) {
            rows.value = result;
            error.value = "";
          }
        } while (dirty && enabled.value && token === generation);
      } catch {
        if (token === generation && enabled.value) error.value = "Не удалось загрузить список агентов.";
      } finally {
        if (token === generation) {
          loading.value = false;
          flight = null;
          controller = undefined;
        }
      }
    })();
    return flight;
  }

  function onEvent(event: ServerEvent): void {
    if (!enabled.value || !(event.type === "session.status" || event.type === "session.error"
      || event.type.startsWith("approval."))) return;
    if (debounce !== undefined) return;
    debounce = setTimeout(() => {
      debounce = undefined;
      void refresh();
    }, 150);
  }

  async function stop(id: string): Promise<void> {
    if (!enabled.value || stopping.value.includes(id)) return;
    const token = generation;
    stopping.value = [...stopping.value, id];
    try {
      await api.stop(id);
      if (token === generation && enabled.value) await refresh();
    } catch {
      if (token === generation && enabled.value) error.value = "Не удалось остановить агента. Повторите попытку.";
    } finally {
      if (token === generation) stopping.value = stopping.value.filter(value => value !== id);
    }
  }

  watch(enabled, (active) => {
    reset();
    if (active) {
      void refresh();
      timer = setInterval(() => { void refresh(); }, 5000);
    }
  }, { immediate: true });
  onScopeDispose(reset);
  return { rows, loading, error, stopping, refresh, onEvent, stop };
}
