<script setup lang="ts">
// Model settings: connections to model servers, the models the agent may use and the
// model for each role. Changes are stored on the server on top of the models file.

import { computed, onMounted, reactive, ref } from "vue";
import { ArrowLeft, Plus, Search } from "lucide-vue-next";
import { useRouter } from "vue-router";
import { ApiError } from "../api/client";
import {
  modelSettings,
  type DiscoveredModel,
  type ModelView,
  type ModelsView,
  type ProviderView,
} from "../api/modelSettings";
import {
  MIN_CONTEXT,
  PROVIDER_PRESETS,
  addedAs,
  canChat,
  formatTokens,
  isValidId,
  keyStatus,
  originLabel,
  suggestModel,
} from "../components/settings/modelForms";
import { useAppStore } from "../stores/app";

const router = useRouter();
const appStore = useAppStore();

const view = ref<ModelsView | null>(null);
const loadError = ref("");
const actionError = ref("");
const busy = ref("");

interface Discovery {
  loading: boolean;
  message: string;
  found: DiscoveredModel[];
}
const discoveries = reactive<Record<string, Discovery>>({});

const presetKey = ref(PROVIDER_PRESETS[0].key);
const preset = computed(() => PROVIDER_PRESETS.find((item) => item.key === presetKey.value) ?? PROVIDER_PRESETS[0]);
const form = reactive({ id: PROVIDER_PRESETS[0].id, baseUrl: PROVIDER_PRESETS[0].baseUrl, apiKey: "" });
const showForm = ref(false);

// Edited context and answer limits, by model ID, until saved.
const limits = reactive<Record<string, { context_window: number; max_tokens: number }>>({});

const activeModels = computed(() => view.value?.models.filter((item) => !item.disabled) ?? []);

function message(error: unknown): string {
  return error instanceof ApiError ? error.message : "Сервер недоступен";
}

async function apply(task: string, action: () => Promise<ModelsView>): Promise<boolean> {
  busy.value = task;
  actionError.value = "";
  try {
    setView(await action());
    // The composer reads its model list from the public config.
    await appStore.loadConfig().catch(() => undefined);
    return true;
  } catch (error) {
    actionError.value = `Не удалось сохранить: ${message(error)}`;
    return false;
  } finally {
    busy.value = "";
  }
}

function setView(next: ModelsView): void {
  view.value = next;
  for (const item of next.models) {
    limits[item.id] = { context_window: item.context_window, max_tokens: item.max_tokens };
  }
}

async function load(): Promise<void> {
  loadError.value = "";
  try {
    setView(await modelSettings.get());
  } catch (error) {
    loadError.value = `Настройки моделей не загрузились: ${message(error)}`;
  }
}

function choosePreset(key: string): void {
  presetKey.value = key;
  form.id = preset.value.id;
  form.baseUrl = preset.value.baseUrl;
  form.apiKey = "";
}

const formError = computed(() => {
  if (!isValidId(form.id)) return "Имя: латиница, цифры и . _ : / @ -";
  if (view.value?.providers.some((item) => item.id === form.id)) return "Подключение с таким именем уже есть";
  try {
    const url = new URL(form.baseUrl);
    if (url.protocol !== "http:" && url.protocol !== "https:") return "Адрес должен начинаться с http:// или https://";
  } catch {
    return "Укажите адрес сервера, например http://127.0.0.1:1234/v1";
  }
  return "";
});

async function addProvider(): Promise<void> {
  if (formError.value) return;
  const id = form.id;
  const saved = await apply("provider:new", () =>
    modelSettings.putProvider(id, {
      kind: "openai",
      base_url: form.baseUrl.trim(),
      ...(form.apiKey ? { api_key: form.apiKey } : {}),
    }),
  );
  if (!saved) return;
  form.apiKey = "";
  showForm.value = false;
  const provider = view.value?.providers.find((item) => item.id === id);
  if (provider) await discover(provider);
}

async function discover(provider: ProviderView): Promise<void> {
  const state: Discovery = { loading: true, message: "", found: [] };
  discoveries[provider.id] = state;
  try {
    const result = await modelSettings.testProvider(provider.id);
    const found = result.details ?? result.models.map((id) => ({ id, context_window: null, kind: null }));
    discoveries[provider.id] = {
      loading: false,
      found,
      message: result.ok
        ? found.length ? "" : "Сервер не вернул ни одной модели"
        : "Сервер не отвечает. Проверьте, что он запущен и адрес верный.",
    };
  } catch (error) {
    discoveries[provider.id] = { loading: false, found: [], message: message(error) };
  }
}

async function addModel(provider: ProviderView, found: DiscoveredModel): Promise<void> {
  if (!view.value) return;
  const { id, patch } = suggestModel(provider, found, view.value.models);
  await apply(`model:${id}`, () => modelSettings.putModel(id, patch));
}

function limitsChanged(item: ModelView): boolean {
  const edited = limits[item.id];
  return !!edited && (edited.context_window !== item.context_window || edited.max_tokens !== item.max_tokens);
}

function limitsError(item: ModelView): string {
  const edited = limits[item.id];
  if (!edited) return "";
  if (!Number.isInteger(edited.context_window) || edited.context_window < MIN_CONTEXT) {
    return `Контекст не меньше ${MIN_CONTEXT}`;
  }
  if (!Number.isInteger(edited.max_tokens) || edited.max_tokens < 1 || edited.max_tokens >= edited.context_window) {
    return "Ответ должен быть меньше контекста";
  }
  return "";
}

async function saveLimits(item: ModelView): Promise<void> {
  if (limitsError(item)) return;
  await apply(`model:${item.id}`, () => modelSettings.putModel(item.id, { ...limits[item.id] }));
}

function toggleProvider(item: ProviderView): Promise<boolean> {
  return apply(`provider:${item.id}`, () => modelSettings.putProvider(item.id, { disabled: !item.disabled }));
}

function toggleModel(item: ModelView): Promise<boolean> {
  return apply(`model:${item.id}`, () => modelSettings.putModel(item.id, { disabled: !item.disabled }));
}

function removeProvider(item: ProviderView): Promise<boolean> {
  delete discoveries[item.id];
  return apply(`provider:${item.id}`, () => modelSettings.removeProvider(item.id));
}

function removeModel(item: ModelView): Promise<boolean> {
  return apply(`model:${item.id}`, () => modelSettings.removeModel(item.id));
}

function setDefault(value: string): Promise<boolean> {
  return apply("routing", () => modelSettings.putRouting({ default: value }));
}

function setLight(value: string): Promise<boolean> {
  return apply("routing", () => modelSettings.putRouting({ light: value || null }));
}

function toggleFallback(id: string): Promise<boolean> {
  const current = view.value?.routing.fallback ?? [];
  const fallback = current.includes(id) ? current.filter((item) => item !== id) : [...current, id];
  return apply("routing", () => modelSettings.putRouting({ fallback }));
}

function back(): void {
  if (window.history.length > 1) router.back();
  else void router.push({ name: "home" });
}

onMounted(load);
</script>

<template>
  <section class="settings" data-testid="models-settings">
    <header class="settings__header">
      <button type="button" class="ghost" data-testid="settings-back" @click="back">
        <ArrowLeft :size="16" aria-hidden="true" />
        <span>Назад</span>
      </button>
      <h1>Модели</h1>
    </header>

    <p v-if="loadError" class="notice notice--error" role="alert">
      {{ loadError }}
      <button type="button" class="link" @click="load">Повторить</button>
    </p>
    <p v-if="actionError" class="notice notice--error" role="alert" data-testid="settings-error">{{ actionError }}</p>
    <ul v-if="view?.warnings.length" class="notice notice--warning">
      <li v-for="warning in view.warnings" :key="warning">{{ warning }}</li>
    </ul>

    <template v-if="view">
      <section class="block" aria-labelledby="providers-title">
        <div class="block__head">
          <h2 id="providers-title">Подключения</h2>
          <button type="button" class="ghost" data-testid="provider-add-open" @click="showForm = !showForm">
            <Plus :size="15" aria-hidden="true" />
            <span>Подключить сервер</span>
          </button>
        </div>
        <p class="hint">Сервер с OpenAI-совместимым API: облачный шлюз или локальный, например LM Studio.</p>

        <form v-if="showForm" class="card form" data-testid="provider-form" @submit.prevent="addProvider">
          <div class="presets" role="radiogroup" aria-label="Тип сервера">
            <button
              v-for="item in PROVIDER_PRESETS"
              :key="item.key"
              type="button"
              role="radio"
              :aria-checked="presetKey === item.key"
              class="preset"
              :data-preset="item.key"
              @click="choosePreset(item.key)"
            >{{ item.label }}</button>
          </div>
          <label>
            <span>Имя</span>
            <input v-model.trim="form.id" data-testid="provider-id" autocomplete="off" />
          </label>
          <label>
            <span>Адрес API</span>
            <input v-model.trim="form.baseUrl" class="mono" data-testid="provider-url" autocomplete="off" />
          </label>
          <label v-if="preset.needsKey">
            <span>Ключ API</span>
            <input v-model="form.apiKey" type="password" data-testid="provider-key" autocomplete="off" />
          </label>
          <p v-if="preset.key === 'lmstudio'" class="hint">
            Запустите сервер в LM Studio (вкладка Developer или <code>lms server start</code>). Длина контекста
            модели при загрузке должна быть не меньше той, что указана здесь.
          </p>
          <p v-if="formError" class="field-error" data-testid="provider-form-error">{{ formError }}</p>
          <div class="actions">
            <button type="submit" class="primary" :disabled="!!formError || busy === 'provider:new'">
              Подключить и найти модели
            </button>
            <button type="button" class="ghost" @click="showForm = false">Отмена</button>
          </div>
        </form>

        <article
          v-for="item in view.providers"
          :key="item.id"
          class="card"
          :class="{ 'card--off': item.disabled }"
          data-testid="provider-card"
          :data-provider="item.id"
        >
          <div class="card__row">
            <div class="card__main">
              <strong>{{ item.id }}</strong>
              <span class="mono muted">{{ item.base_url }}</span>
              <span class="muted small">{{ keyStatus(item) }} · {{ originLabel(item.origin) }}</span>
            </div>
            <div class="card__actions">
              <button
                type="button"
                class="ghost"
                :disabled="item.disabled || discoveries[item.id]?.loading"
                data-testid="provider-discover"
                @click="discover(item)"
              >
                <Search :size="15" aria-hidden="true" />
                <span>{{ discoveries[item.id]?.loading ? "Ищу…" : "Найти модели" }}</span>
              </button>
              <button type="button" class="ghost" :disabled="!!busy" @click="toggleProvider(item)">
                {{ item.disabled ? "Включить" : "Отключить" }}
              </button>
              <button
                v-if="item.origin === 'ui'"
                type="button"
                class="ghost danger"
                :disabled="!!busy"
                data-testid="provider-remove"
                @click="removeProvider(item)"
              >Удалить</button>
            </div>
          </div>

          <div v-if="discoveries[item.id] && !discoveries[item.id].loading" class="found" data-testid="found-models">
            <p v-if="discoveries[item.id].message" class="muted small">{{ discoveries[item.id].message }}</p>
            <ul v-else>
              <li v-for="found in discoveries[item.id].found" :key="found.id" class="found__row" :data-found="found.id">
                <span class="mono">{{ found.id }}</span>
                <span class="muted small">
                  {{ canChat(found) ? `контекст ${formatTokens(found.context_window)}` : "модель эмбеддингов" }}
                </span>
                <span v-if="addedAs(view.models, item.id, found.id)" class="muted small found__state">
                  Добавлена как {{ addedAs(view.models, item.id, found.id) }}
                </span>
                <button
                  v-else-if="canChat(found)"
                  type="button"
                  class="ghost found__state"
                  :disabled="!!busy"
                  data-testid="found-add"
                  @click="addModel(item, found)"
                >
                  <Plus :size="14" aria-hidden="true" />
                  <span>Добавить</span>
                </button>
              </li>
            </ul>
          </div>
        </article>
      </section>

      <section class="block" aria-labelledby="models-title">
        <h2 id="models-title">Модели агента</h2>
        <p v-if="!view.models.length" class="muted">Моделей пока нет. Подключите сервер и найдите модели.</p>
        <article
          v-for="item in view.models"
          :key="item.id"
          class="card"
          :class="{ 'card--off': item.disabled }"
          data-testid="model-card"
          :data-model="item.id"
        >
          <div class="card__row">
            <div class="card__main">
              <strong>{{ item.id }}</strong>
              <span class="muted small">
                <span class="mono">{{ item.model }}</span> · {{ item.provider }} · {{ originLabel(item.origin) }}
              </span>
            </div>
            <div class="card__actions">
              <button type="button" class="ghost" :disabled="!!busy" @click="toggleModel(item)">
                {{ item.disabled ? "Включить" : "Отключить" }}
              </button>
              <button
                v-if="item.origin === 'ui'"
                type="button"
                class="ghost danger"
                :disabled="!!busy"
                data-testid="model-remove"
                @click="removeModel(item)"
              >Удалить</button>
            </div>
          </div>
          <div v-if="limits[item.id]" class="limits">
            <label>
              <span>Контекст, токенов</span>
              <input v-model.number="limits[item.id].context_window" type="number" min="4096" step="1024" data-testid="model-context" />
            </label>
            <label>
              <span>Максимум ответа</span>
              <input v-model.number="limits[item.id].max_tokens" type="number" min="1" step="256" data-testid="model-max-tokens" />
            </label>
            <button
              v-if="limitsChanged(item)"
              type="button"
              class="primary"
              :disabled="!!limitsError(item) || !!busy"
              data-testid="model-save"
              @click="saveLimits(item)"
            >Сохранить</button>
            <span v-if="limitsChanged(item) && limitsError(item)" class="field-error">{{ limitsError(item) }}</span>
          </div>
        </article>
      </section>

      <section v-if="activeModels.length" class="block" aria-labelledby="roles-title">
        <h2 id="roles-title">Роли</h2>
        <div class="card roles">
          <label>
            <span>Основная модель</span>
            <select
              :value="view.routing.default"
              :disabled="!!busy"
              data-testid="routing-default"
              @change="setDefault(($event.target as HTMLSelectElement).value)"
            >
              <option v-for="item in activeModels" :key="item.id" :value="item.id">{{ item.id }}</option>
            </select>
          </label>
          <label>
            <span>Служебная модель (названия, сжатие контекста)</span>
            <select
              :value="view.routing.light ?? ''"
              :disabled="!!busy"
              data-testid="routing-light"
              @change="setLight(($event.target as HTMLSelectElement).value)"
            >
              <option value="">Как основная</option>
              <option v-for="item in activeModels" :key="item.id" :value="item.id">{{ item.id }}</option>
            </select>
          </label>
          <fieldset>
            <legend>Запасные модели при сбоях</legend>
            <label v-for="item in activeModels" :key="item.id" class="check">
              <input
                type="checkbox"
                :checked="view.routing.fallback.includes(item.id)"
                :disabled="!!busy || item.id === view.routing.default"
                :data-fallback="item.id"
                @change="toggleFallback(item.id)"
              />
              <span>{{ item.id }}</span>
            </label>
          </fieldset>
        </div>
      </section>
    </template>
    <p v-else-if="!loadError" class="muted">Загрузка…</p>
  </section>
</template>

<style scoped>
.settings {
  max-width: 860px;
  margin: 0 auto;
  padding: 16px 20px 48px;
  overflow-y: auto;
  height: 100%;
  box-sizing: border-box;
}

.settings__header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
}

h1 {
  margin: 0;
  font-size: 20px;
}

h2 {
  margin: 0;
  font-size: 15px;
}

.block {
  margin-top: 20px;
}

.block__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.hint {
  margin: 4px 0 10px;
  color: var(--text-muted);
  font-size: 13px;
}

.card {
  margin-top: 8px;
  padding: 12px 14px;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--surface);
}

.card--off {
  opacity: 0.6;
}

.card__row {
  display: flex;
  flex-wrap: wrap;
  align-items: flex-start;
  justify-content: space-between;
  gap: 8px 12px;
}

.card__main {
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
  overflow-wrap: anywhere;
}

.card__actions {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.small {
  font-size: 12px;
}

button.ghost,
button.primary,
.preset {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 5px 10px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: transparent;
  color: var(--text);
  font-size: 13px;
  cursor: pointer;
}

button.primary {
  background: var(--accent);
  border-color: var(--accent);
  color: var(--accent-contrast);
}

button:disabled {
  opacity: 0.5;
  cursor: default;
}

button.danger {
  color: var(--danger);
}

button.link {
  border: 0;
  background: none;
  color: var(--link);
  text-decoration: underline;
  cursor: pointer;
}

.preset[aria-checked="true"] {
  border-color: var(--accent);
  color: var(--accent);
  background: color-mix(in srgb, var(--accent) 10%, transparent);
}

.form,
.roles {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.presets {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

label {
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 13px;
  color: var(--text-muted);
}

input,
select {
  padding: 6px 8px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--bg);
  color: var(--text);
  font: inherit;
  font-size: 14px;
}

input.mono {
  font-family: var(--font-mono);
}

.actions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.field-error {
  margin: 0;
  color: var(--danger);
  font-size: 13px;
}

.notice {
  margin: 8px 0;
  padding: 8px 12px;
  border-radius: var(--radius-sm);
  font-size: 13px;
}

.notice--error {
  border: 1px solid color-mix(in srgb, var(--danger) 45%, transparent);
  background: color-mix(in srgb, var(--danger) 8%, transparent);
  color: var(--danger);
}

.notice--warning {
  padding-left: 28px;
  border: 1px solid color-mix(in srgb, var(--warning) 45%, transparent);
  background: color-mix(in srgb, var(--warning) 8%, transparent);
  color: var(--warning);
}

.found {
  margin-top: 10px;
  padding-top: 8px;
  border-top: 1px solid var(--border);
}

.found ul {
  margin: 0;
  padding: 0;
  list-style: none;
}

.found__row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 4px 12px;
  padding: 4px 0;
  overflow-wrap: anywhere;
}

.found__state {
  margin-left: auto;
}

.limits {
  display: flex;
  flex-wrap: wrap;
  align-items: flex-end;
  gap: 10px;
  margin-top: 10px;
}

.limits input {
  width: 9rem;
}

fieldset {
  margin: 0;
  padding: 0;
  border: 0;
}

legend {
  margin-bottom: 4px;
  font-size: 13px;
  color: var(--text-muted);
}

.check {
  flex-direction: row;
  align-items: center;
  gap: 8px;
  color: var(--text);
}
</style>
