<script setup lang="ts">
// Model settings: connections to model servers, the models the agent may use and the
// model for each role. Changes are stored on the server on top of the models file.

import { computed, onBeforeUnmount, onMounted, reactive, ref, watch } from "vue";
import { CircleAlert, Info, Plus, Search, Trash2, TriangleAlert, X } from "lucide-vue-next";
import { ApiError } from "../api/client";
import {
  modelSettings,
  type DiscoveredModel,
  type KeySource,
  type LocalServer,
  type ModelPatch,
  type ModelView,
  type ModelsView,
  type ProbeResult,
  type ProviderView,
} from "../api/modelSettings";
import {
  MIN_CONTEXT,
  PROVIDER_PRESETS,
  SERVER_NAMES,
  addedAs,
  canChat,
  describeFound,
  isValidId,
  keyStatus,
  originLabel,
  probeText,
  providerIdFor,
  rolesOf,
  suggestModel,
} from "../components/settings/modelForms";
import SettingsDialog from "../components/settings/SettingsDialog.vue";
import SettingsHeader from "../components/settings/SettingsHeader.vue";
import ToggleSwitch from "../components/settings/ToggleSwitch.vue";
import { useAppStore } from "../stores/app";
import { SESSION_MODES, modeLabel } from "../modes";

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

// One-token checks: of found models by "provider model", and of agent models by model ID.
type Check = ProbeResult | "checking";
const foundChecks = reactive<Record<string, Check>>({});
const modelChecks = reactive<Record<string, Check>>({});
const foundKey = (provider: string, model: string) => `${provider} ${model}`;
let probeVersion = 0;

function invalidateChecks(): void {
  probeVersion += 1;
  for (const key of Object.keys(modelChecks)) delete modelChecks[key];
  for (const key of Object.keys(foundChecks)) delete foundChecks[key];
  for (const [id, discovery] of Object.entries(discoveries)) {
    if (discovery.loading) delete discoveries[id];
  }
}

const localServers = ref<LocalServer[] | null>(null);
const scanning = ref(false);

function foundCheck(provider: ProviderView, found: DiscoveredModel): Check | undefined {
  return foundChecks[foundKey(provider.id, found.id)];
}

function answered(check: Check | undefined): check is ProbeResult {
  return !!check && check !== "checking" && check.ok;
}

function failed(check: Check | undefined): check is ProbeResult {
  return !!check && check !== "checking" && !check.ok;
}

async function checkFound(provider: ProviderView, found: DiscoveredModel): Promise<void> {
  const key = foundKey(provider.id, found.id);
  const version = probeVersion;
  const discovery = discoveries[provider.id];
  foundChecks[key] = "checking";
  let result: ProbeResult;
  try {
    result = await modelSettings.probeFound(provider.id, found.id);
  } catch (error) {
    result = { ok: false, status: null, seconds: null, message: message(error) };
  }
  if (alive && version === probeVersion && discovery === discoveries[provider.id]) foundChecks[key] = result;
}

async function checkModel(item: ModelView): Promise<void> {
  const version = probeVersion;
  modelChecks[item.id] = "checking";
  let result: ProbeResult;
  try {
    result = await modelSettings.probeModel(item.id);
  } catch (error) {
    result = { ok: false, status: null, seconds: null, message: message(error) };
  }
  if (!alive || version !== probeVersion) return;
  modelChecks[item.id] = result;
  // The composer hides a model that did not answer.
  await appStore.loadConfig().catch(() => undefined);
}

async function scan(): Promise<void> {
  scanning.value = true;
  try {
    localServers.value = await modelSettings.scanLocal();
  } catch {
    localServers.value = [];
  } finally {
    scanning.value = false;
  }
}

async function connectLocal(server: LocalServer): Promise<void> {
  if (!view.value) return;
  const id = providerIdFor(server.server, view.value.providers);
  const saved = await apply(`provider:${id}`, () =>
    modelSettings.putProvider(id, { kind: "openai", base_url: server.base_url, key_source: "none" }, saveOptions(true)),
  );
  if (!saved) return;
  server.provider = id;
  const provider = view.value?.providers.find((item) => item.id === id);
  if (provider) await discover(provider);
}

const presetKey = ref(PROVIDER_PRESETS[0].key);
const preset = computed(() => PROVIDER_PRESETS.find((item) => item.key === presetKey.value) ?? PROVIDER_PRESETS[0]);
const form = reactive({ id: PROVIDER_PRESETS[0].id, baseUrl: PROVIDER_PRESETS[0].baseUrl, apiKey: "" });
const showForm = ref(false);

function openForm(): void {
  choosePreset(presetKey.value);
  actionError.value = "";
  showForm.value = true;
}

function closeForm(): void {
  if (busy.value !== "provider:new") showForm.value = false;
}

// Provider address being edited, by provider ID.
const addresses = reactive<Record<string, string>>({});

function addressError(value: string): string {
  try {
    const url = new URL(value);
    return url.protocol === "http:" || url.protocol === "https:" ? "" : "Адрес должен начинаться с http:// или https://";
  } catch {
    return "Укажите адрес, например http://127.0.0.1:1234/v1";
  }
}

async function saveAddress(item: ProviderView): Promise<void> {
  const value = addresses[item.id]?.trim() ?? "";
  if (addressError(value)) return;
  const saved = await apply(`provider:${item.id}`, () => modelSettings.putProvider(item.id, { base_url: value }, saveOptions()));
  if (!saved) return;
  delete addresses[item.id];
  delete discoveries[item.id];
  for (const model of view.value?.models ?? []) if (model.provider === item.id) delete modelChecks[model.id];
}

// Edited limits and price, by model ID, until saved. "unset" is shown only while the
// model has no price yet.
type PriceMode = "free" | "paid" | "unset";
interface LimitsEdit { context_window: number; max_tokens: number; price: PriceMode; input: number; output: number }
const limits = reactive<Record<string, LimitsEdit>>({});

function editOf(item: ModelView): LimitsEdit {
  const pricing = item.pricing;
  const price: PriceMode = !pricing ? "unset" : pricing.input_per_mtok || pricing.output_per_mtok ? "paid" : "free";
  return {
    context_window: item.context_window,
    max_tokens: item.max_tokens,
    price,
    input: pricing?.input_per_mtok ?? 0,
    output: pricing?.output_per_mtok ?? 0,
  };
}

const priceKey = (edit: LimitsEdit) => (edit.price === "paid" ? `paid ${edit.input} ${edit.output}` : edit.price);

function priceLabel(item: ModelView): string {
  const edit = editOf(item);
  if (edit.price === "unset") return "Цена не указана";
  if (edit.price === "free") return "Без оплаты";
  return `Платная: вход $${edit.input}, выход $${edit.output} за 1M токенов`;
}

// The session money ceiling from the server; a model without a price cannot start under it.
const costLimit = computed(() => appStore.config?.session_cost_limit ?? null);

const activeModels = computed(() => view.value?.models.filter((item) => !item.disabled && !item.provider_disabled) ?? []);
const roleNames = computed(() => [...new Set([...SESSION_MODES, ...Object.keys(view.value?.routing.roles ?? {})])]);
const keys = reactive<Record<string, { source: KeySource; value: string; env: string }>>({});

function saveOptions(create = false) {
  return { ...(view.value?.checksum ? { checksum: view.value.checksum } : {}), ...(create ? { create: true } : {}) };
}

function editKey(item: ProviderView): void {
  keys[item.id] = {
    source: item.key_source ?? (item.api_key_env ? "env" : item.api_key_set ? "ui" : "none"),
    value: "", env: item.key_env ?? item.api_key_env ?? "",
  };
}

function keyError(item: ProviderView): string {
  const key = keys[item.id];
  if (!key) return "";
  if (key.source === "env" && !/^[A-Za-z_][A-Za-z0-9_]*$/.test(key.env)) return "Укажите имя переменной окружения";
  if (key.source === "ui" && !key.value.trim() && !(item.key_source === "ui" && item.api_key_set)) return "Введите новый ключ";
  return "";
}

async function saveKey(item: ProviderView): Promise<void> {
  const key = keys[item.id];
  if (!key || keyError(item)) return;
  const patch = key.source === "env" ? { key_source: key.source, api_key_env: key.env }
    : key.source === "ui" ? { key_source: key.source, ...(key.value ? { api_key: key.value } : {}) }
      : { key_source: key.source };
  if (await apply(`provider:${item.id}`, () => modelSettings.putProvider(item.id, patch, saveOptions()))) delete keys[item.id];
}


function message(error: unknown): string {
  return error instanceof ApiError ? error.message : "Сервер недоступен";
}

async function apply(task: string, action: () => Promise<ModelsView>): Promise<boolean> {
  if (busy.value) return false;
  busy.value = task;
  actionError.value = "";
  try {
    loadVersion += 1;
    const result = await action();
    invalidateChecks();
    setView(result, true);
    // The composer reads its model list from the public config.
    await appStore.loadConfig().catch(() => undefined);
    return true;
  } catch (error) {
    actionError.value = `Не удалось сохранить: ${message(error)}`;
    if (error instanceof ApiError && error.status === 409) await load();
    return false;
  } finally {
    busy.value = "";
    if (appStore.modelsChecksum && appStore.modelsChecksum !== view.value?.checksum) void load();
  }
}

function setView(next: ModelsView, preserveEdits = true): void {
  const previous = view.value;
  if (previous && previous.checksum !== next.checksum) invalidateChecks();
  for (const item of next.providers) {
    if (JSON.stringify(previous?.providers.find(provider => provider.id === item.id)) !== JSON.stringify(item)) delete discoveries[item.id];
  }
  for (const item of next.models) {
    const old = previous?.models.find(model => model.id === item.id);
    const edited = limits[item.id];
    const fresh = editOf(item);
    const before = old && editOf(old);
    const keep = preserveEdits && before && edited;
    limits[item.id] = {
      context_window: keep && edited.context_window !== before.context_window ? edited.context_window : fresh.context_window,
      max_tokens: keep && edited.max_tokens !== before.max_tokens ? edited.max_tokens : fresh.max_tokens,
      ...(keep && priceKey(edited) !== priceKey(before)
        ? { price: edited.price, input: edited.input, output: edited.output }
        : { price: fresh.price, input: fresh.input, output: fresh.output }),
    };
    if (old?.model !== item.model || old?.provider !== item.provider
      || previous?.providers.find(provider => provider.id === old?.provider)?.base_url
        !== next.providers.find(provider => provider.id === item.provider)?.base_url) delete modelChecks[item.id];
  }
  view.value = next;
}

let alive = true;
let loadVersion = 0;
let loadFlight: Promise<void> | null = null;
let reload = false;

function load(): Promise<void> {
  if (!alive) return Promise.resolve();
  if (loadFlight) { reload = true; return loadFlight; }
  loadFlight = (async () => {
    do {
      reload = false;
      const version = loadVersion;
      loadError.value = "";
      try {
        const result = await modelSettings.get();
        if (alive && version === loadVersion && !reload) setView(result);
      } catch (error) {
        if (alive && version === loadVersion) loadError.value = `Настройки моделей не загрузились: ${message(error)}`;
      }
    } while (reload && alive);
  })().finally(() => { loadFlight = null; });
  return loadFlight;
}

watch(() => appStore.modelsChecksum, checksum => {
  if (!checksum || checksum === view.value?.checksum || busy.value) return;
  if (Object.keys(keys).length || Object.keys(addresses).length || view.value?.models.some(limitsChanged)) {
    actionError.value = "Настройки изменены в другом окне. При сохранении будет проверена их версия.";
  } else void load();
});
onBeforeUnmount(() => { alive = false; loadVersion += 1; });

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
      key_source: form.apiKey ? "ui" : "none",
      ...(form.apiKey ? { api_key: form.apiKey } : {}),
    }, saveOptions(true)),
  );
  if (!saved) return;
  form.apiKey = "";
  showForm.value = false;
  const provider = view.value?.providers.find((item) => item.id === id);
  if (provider) await discover(provider);
}

async function discover(provider: ProviderView): Promise<void> {
  const version = probeVersion;
  for (const key of Object.keys(foundChecks)) {
    if (key.startsWith(`${provider.id} `)) delete foundChecks[key];
  }
  const state = reactive<Discovery>({ loading: true, message: "", found: [] });
  discoveries[provider.id] = state;
  try {
    const result = await modelSettings.testProvider(provider.id);
    if (!alive || version !== probeVersion || discoveries[provider.id] !== state) return;
    const found = result.details ?? result.models.map((id) => ({ id, context_window: null, kind: null }));
    discoveries[provider.id] = {
      loading: false,
      found,
      message: result.ok
        ? found.length ? "" : "Сервер не вернул ни одной модели"
        : "Сервер не отвечает. Проверьте, что он запущен и адрес верный.",
    };
  } catch (error) {
    if (alive && version === probeVersion && discoveries[provider.id] === state) discoveries[provider.id] = { loading: false, found: [], message: message(error) };
  }
}

async function addModel(provider: ProviderView, found: DiscoveredModel): Promise<void> {
  if (!view.value) return;
  const { id, patch } = suggestModel(provider, found, view.value.models);
  await apply(`model:${id}`, () => modelSettings.putModel(id, patch, saveOptions(true)));
}

function priceChanged(item: ModelView): boolean {
  const edited = limits[item.id];
  return !!edited && priceKey(edited) !== priceKey(editOf(item));
}

function limitsChanged(item: ModelView): boolean {
  const edited = limits[item.id];
  return !!edited && (edited.context_window !== item.context_window || edited.max_tokens !== item.max_tokens || priceChanged(item));
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
  if (edited.price === "paid" && ![edited.input, edited.output].every((value) => Number.isFinite(value) && value >= 0)) {
    return "Цена — число от 0";
  }
  return "";
}

async function saveLimits(item: ModelView): Promise<void> {
  if (limitsError(item)) return;
  const edited = limits[item.id];
  const patch: ModelPatch = { context_window: edited.context_window, max_tokens: edited.max_tokens };
  if (priceChanged(item) && edited.price !== "unset") {
    patch.pricing = edited.price === "free"
      ? { input_per_mtok: 0, output_per_mtok: 0 }
      : { input_per_mtok: edited.input, output_per_mtok: edited.output };
  }
  await apply(`model:${item.id}`, () => modelSettings.putModel(item.id, patch, saveOptions()));
}

function toggleProvider(item: ProviderView): Promise<boolean> {
  return apply(`provider:${item.id}`, () => modelSettings.putProvider(item.id, { disabled: !item.disabled }, saveOptions()));
}

function toggleModel(item: ModelView): Promise<boolean> {
  return apply(`model:${item.id}`, () => modelSettings.putModel(item.id, { disabled: !item.disabled }, saveOptions()));
}

// A removal waits for the confirmation dialog; the dialog stays open while it runs or fails.
interface Removal { title: string; text: string; action: () => Promise<boolean> }
const removal = ref<Removal | null>(null);
const dialogOpen = computed(() => showForm.value || !!removal.value);

function removeProvider(item: ProviderView): void {
  const linked = (view.value?.models ?? []).filter((model) => model.provider === item.id).length;
  actionError.value = "";
  removal.value = {
    title: `Удалить подключение «${item.id}»?`,
    text: linked
      ? `Вместе с ним удалятся его модели: ${linked}. Сам сервер моделей не меняется.`
      : "Сам сервер моделей не меняется.",
    action: async () => {
      const saved = await apply(`provider:${item.id}`, () => modelSettings.removeProvider(item.id, saveOptions()));
      if (saved) {
        delete discoveries[item.id];
        delete keys[item.id];
        delete addresses[item.id];
      }
      return saved;
    },
  };
}

function removeModel(item: ModelView): void {
  actionError.value = "";
  removal.value = {
    title: `Удалить модель «${item.id}»?`,
    text: "Модель пропадёт из списка моделей агента. Сервер и файлы модели не меняются.",
    action: () => apply(`model:${item.id}`, () => modelSettings.removeModel(item.id, saveOptions())),
  };
}

async function confirmRemoval(): Promise<void> {
  const current = removal.value;
  if (current && (await current.action())) removal.value = null;
}

function closeRemoval(): void {
  if (!busy.value) removal.value = null;
}

function setDefault(value: string): Promise<boolean> {
  return apply("routing", () => modelSettings.putRouting({ default: value }, saveOptions()));
}

function setLight(value: string): Promise<boolean> {
  return apply("routing", () => modelSettings.putRouting({ light: value || null }, saveOptions()));
}

function toggleFallback(id: string): Promise<boolean> {
  const current = view.value?.routing.fallback ?? [];
  const fallback = current.includes(id) ? current.filter((item) => item !== id) : [...current, id];
  return apply("routing", () => modelSettings.putRouting({ fallback }, saveOptions()));
}

function setRole(role: string, value: string): Promise<boolean> {
  const roles = { ...view.value?.routing.roles };
  if (value) roles[role] = value;
  else delete roles[role];
  return apply("routing", () => modelSettings.putRouting({ roles }, saveOptions()));
}

function resetProvider(id: string): Promise<boolean> {
  return apply(`provider:${id}`, () => modelSettings.resetProvider(id, saveOptions()));
}

function resetModel(id: string): Promise<boolean> {
  return apply(`model:${id}`, () => modelSettings.resetModel(id, saveOptions()));
}

onMounted(() => { void load(); void scan(); });

</script>

<template>
  <section class="ui-page" data-testid="models-settings">
    <SettingsHeader title="Модели" />

    <p v-if="loadError" class="ui-notice ui-notice--error" role="alert">
      <CircleAlert :size="16" aria-hidden="true" />
      <span class="ui-notice__text">{{ loadError }}</span>
      <button type="button" class="ui-link" @click="load">Повторить</button>
    </p>
    <p v-if="actionError && !dialogOpen" class="ui-notice ui-notice--error" role="alert" data-testid="settings-error">
      <CircleAlert :size="16" aria-hidden="true" />
      <span class="ui-notice__text">{{ actionError }}</span>
      <button type="button" class="ui-btn ui-btn--icon" aria-label="Скрыть сообщение" @click="actionError = ''">
        <X :size="15" aria-hidden="true" />
      </button>
    </p>
    <div v-if="view?.warnings.length" class="ui-notice ui-notice--warning">
      <TriangleAlert :size="16" aria-hidden="true" />
      <ul class="ui-notice__text">
        <li v-for="warning in view.warnings" :key="warning">{{ warning }}</li>
      </ul>
    </div>

    <template v-if="view">
      <section class="ui-section" aria-labelledby="providers-title">
        <div class="ui-section__head">
          <h2 id="providers-title" class="ui-section__title">Подключения</h2>
          <div class="ui-section__actions">
            <button type="button" class="ui-btn ui-btn--filled" data-testid="provider-add-open" @click="openForm">
              <Plus :size="16" aria-hidden="true" />
              <span>Подключить сервер</span>
            </button>
          </div>
        </div>
        <p class="ui-hint">Сервер с OpenAI-совместимым API: облачный шлюз или локальный, например LM Studio.</p>

        <div class="ui-card local" data-testid="local-servers">
          <div class="ui-card__row">
            <strong class="ui-card__title">На этом компьютере</strong>
            <button type="button" class="ui-btn ui-btn--text" :disabled="scanning" data-testid="local-scan" @click="scan">
              <Search :size="16" aria-hidden="true" />
              <span>{{ scanning ? "Ищу…" : "Искать снова" }}</span>
            </button>
          </div>
          <p v-if="localServers && !localServers.length" class="ui-card__meta local__empty">
            Запущенных серверов моделей не найдено. Запустите сервер в LM Studio (<code>lms server start</code>)
            или Ollama и нажмите «Искать снова».
          </p>
          <ul v-else-if="localServers" class="local__list">
            <li v-for="server in localServers" :key="server.base_url" class="local__row" :data-local="server.base_url">
              <span>{{ SERVER_NAMES[server.server] }}</span>
              <span class="mono ui-card__meta">{{ server.base_url }}</span>
              <span class="ui-card__meta">моделей: {{ server.models }}</span>
              <span v-if="server.provider" class="ui-card__meta local__action">Подключён как {{ server.provider }}</span>
              <button
                v-else
                type="button"
                class="ui-btn ui-btn--tonal local__action"
                :disabled="!!busy"
                data-testid="local-connect"
                @click="connectLocal(server)"
              >
                <Plus :size="16" aria-hidden="true" />
                <span>Подключить</span>
              </button>
            </li>
          </ul>
        </div>

        <article
          v-for="item in view.providers"
          :key="item.id"
          class="ui-card"
          :class="{ 'ui-card--off': item.disabled }"
          data-testid="provider-card"
          :data-provider="item.id"
        >
          <div class="ui-card__row">
            <div class="ui-card__main">
              <strong class="ui-card__title">{{ item.id }}</strong>
              <span class="mono ui-card__meta">{{ item.base_url }}</span>
              <span class="ui-card__meta">{{ keyStatus(item) }} · {{ originLabel(item.origin, item.has_base) }}</span>
              <span v-if="item.key_source" class="ui-card__meta" data-testid="provider-key-origin">
                Источник ключа: {{ item.key_source === 'ui' ? 'сохранён в приложении' : item.key_source === 'env' ? 'окружение' : 'без ключа' }}
              </span>
              <button
                v-if="item.has_base && item.origin === 'ui'"
                type="button"
                class="ui-link ui-small card__link"
                :disabled="!!busy"
                data-testid="provider-reset"
                @click="resetProvider(item.id)"
              >Сбросить изменения</button>
            </div>
            <div class="ui-card__actions">
              <button
                type="button"
                class="ui-btn ui-btn--outlined"
                :disabled="item.disabled || discoveries[item.id]?.loading"
                data-testid="provider-discover"
                @click="discover(item)"
              >
                <Search :size="16" aria-hidden="true" />
                <span>{{ discoveries[item.id]?.loading ? "Ищу…" : "Найти модели" }}</span>
              </button>
              <button
                type="button"
                class="ui-btn ui-btn--text"
                :disabled="!!busy"
                data-testid="provider-edit-url"
                @click="addresses[item.id] = item.base_url"
              >Адрес</button>
              <button
                type="button"
                class="ui-btn ui-btn--text"
                :disabled="!!busy"
                data-testid="provider-edit-key"
                @click="editKey(item)"
              >Ключ</button>
              <button
                type="button"
                class="ui-btn ui-btn--icon ui-btn--danger"
                :aria-label="`Удалить подключение ${item.id}`"
                title="Удалить"
                :disabled="!!busy"
                data-testid="provider-remove"
                @click="removeProvider(item)"
              >
                <Trash2 :size="16" aria-hidden="true" />
              </button>
              <ToggleSwitch
                :model-value="!item.disabled"
                :label="`Подключение ${item.id} включено`"
                :disabled="!!busy"
                data-testid="provider-toggle"
                @update:model-value="toggleProvider(item)"
              />
            </div>
          </div>

          <form
            v-if="addresses[item.id] !== undefined"
            class="inline-form"
            data-testid="provider-url-form"
            @submit.prevent="saveAddress(item)"
          >
            <label class="ui-field inline-form__wide">
              <span class="ui-field__label">Адрес API</span>
              <input
                v-model.trim="addresses[item.id]"
                class="ui-input mono"
                autocomplete="off"
                spellcheck="false"
                :aria-invalid="!!addressError(addresses[item.id])"
                data-testid="provider-url-edit"
              />
              <small v-if="addressError(addresses[item.id])" class="ui-field__help ui-field__help--error">
                {{ addressError(addresses[item.id]) }}
              </small>
            </label>
            <div class="inline-form__actions">
              <button type="button" class="ui-btn ui-btn--text" @click="delete addresses[item.id]">Отмена</button>
              <button type="submit" class="ui-btn ui-btn--filled" :disabled="!!addressError(addresses[item.id]) || !!busy">
                Сохранить
              </button>
            </div>
          </form>

          <form v-if="keys[item.id]" class="inline-form" data-testid="provider-key-form" @submit.prevent="saveKey(item)">
            <label class="ui-field">
              <span class="ui-field__label">Источник ключа</span>
              <select v-model="keys[item.id].source" class="ui-input" data-testid="provider-key-source">
                <option value="ui">Сохранить в приложении</option>
                <option value="env">Из окружения</option>
                <option value="none">Без ключа</option>
              </select>
            </label>
            <label v-if="keys[item.id].source === 'ui'" class="ui-field inline-form__wide">
              <span class="ui-field__label">Новый ключ</span>
              <input
                v-model="keys[item.id].value"
                class="ui-input"
                type="password"
                autocomplete="new-password"
                data-testid="provider-key-value"
              />
              <small v-if="item.key_source === 'ui' && item.api_key_set" class="ui-field__help">
                Оставьте пустым, чтобы сохранить текущий ключ.
              </small>
            </label>
            <label v-if="keys[item.id].source === 'env'" class="ui-field inline-form__wide">
              <span class="ui-field__label">Переменная окружения</span>
              <input v-model.trim="keys[item.id].env" class="ui-input mono" autocomplete="off" data-testid="provider-key-env" />
            </label>
            <p v-if="keys[item.id].source === 'none'" class="ui-field__help inline-form__wide">
              Сохранённый ключ будет очищен. Ключ из окружения использоваться не будет.
            </p>
            <p v-if="keyError(item)" class="ui-field__help ui-field__help--error inline-form__wide">{{ keyError(item) }}</p>
            <div class="inline-form__actions">
              <button type="button" class="ui-btn ui-btn--text" @click="delete keys[item.id]">Отмена</button>
              <button type="submit" class="ui-btn ui-btn--filled" :disabled="!!busy || !!keyError(item)">Сохранить ключ</button>
            </div>
          </form>

          <div v-if="discoveries[item.id] && !discoveries[item.id].loading" class="found" data-testid="found-models">
            <p v-if="discoveries[item.id].message" class="ui-card__meta">{{ discoveries[item.id].message }}</p>
            <ul v-else>
              <li v-for="found in discoveries[item.id].found" :key="found.id" class="found__item" :data-found="found.id">
                <div class="found__info">
                  <span class="mono found__name">{{ found.id }}</span>
                  <span class="ui-card__meta">{{ describeFound(found) }}</span>
                  <span v-if="found.state" class="ui-card__meta">
                    {{ found.state === "loaded" ? "загружена в память" : "не загружена" }}
                  </span>
                  <span v-if="foundCheck(item, found) === 'checking'" class="ui-card__meta">Проверяю…</span>
                  <span v-else-if="answered(foundCheck(item, found))" class="ui-small ui-ok" data-testid="found-check">
                    {{ probeText(foundCheck(item, found) as ProbeResult) }}
                  </span>
                  <span v-else-if="failed(foundCheck(item, found))" class="ui-small ui-bad" data-testid="found-check">
                    {{ probeText(foundCheck(item, found) as ProbeResult) }}
                  </span>
                </div>
                <div v-if="addedAs(view.models, item.id, found.id)" class="found__actions ui-card__meta">
                  Добавлена как {{ addedAs(view.models, item.id, found.id) }}
                </div>
                <div v-else-if="canChat(found)" class="found__actions">
                  <button
                    type="button"
                    class="ui-btn ui-btn--text"
                    title="Пробный запрос генерации; может расходовать подписку"
                    data-testid="found-probe"
                    :disabled="foundCheck(item, found) === 'checking'"
                    @click="checkFound(item, found)"
                  >Проверить ответ</button>
                  <button
                    type="button"
                    class="ui-btn ui-btn--tonal"
                    :disabled="!!busy"
                    data-testid="found-add"
                    @click="addModel(item, found)"
                  >
                    <Plus :size="16" aria-hidden="true" />
                    <span>Добавить</span>
                  </button>
                </div>
              </li>
            </ul>
          </div>
        </article>
      </section>

      <section class="ui-section" aria-labelledby="models-title">
        <h2 id="models-title" class="ui-section__title">Модели агента</h2>
        <p class="ui-hint">Модели, которые агент может выбрать для сессии. Выключенная модель не предлагается.</p>
        <p v-if="!view.models.length" class="ui-empty">Моделей пока нет. Подключите сервер и найдите модели.</p>
        <article
          v-for="item in view.models"
          :key="item.id"
          class="ui-card"
          :class="{ 'ui-card--off': item.disabled || item.provider_disabled }"
          data-testid="model-card"
          :data-model="item.id"
        >
          <div class="ui-card__row">
            <div class="ui-card__main">
              <strong class="ui-card__title">{{ item.id }}</strong>
              <span class="ui-card__meta">
                <span class="mono">{{ item.model }}</span> · {{ item.provider }} · {{ originLabel(item.origin, item.has_base) }}
              </span>
              <span v-if="item.provider_disabled" class="ui-card__meta" data-testid="model-provider-off">Сервер отключён</span>
              <span class="ui-card__meta" data-testid="model-price">{{ priceLabel(item) }}</span>
              <span v-if="!item.pricing && costLimit" class="ui-small ui-bad" data-testid="model-price-warning">
                У сессий есть лимит стоимости ${{ costLimit }}, поэтому модель без цены не запустится.
                Выберите «Без оплаты» или укажите цену.
              </span>
              <span v-if="rolesOf(view.routing, item.id).length" class="ui-small" data-testid="model-roles">
                Роль: {{ rolesOf(view.routing, item.id).join(", ") }}
              </span>
              <span v-if="modelChecks[item.id] === 'checking'" class="ui-card__meta">Проверяю…</span>
              <span
                v-else-if="modelChecks[item.id]"
                class="ui-small"
                :class="answered(modelChecks[item.id]) ? 'ui-ok' : 'ui-bad'"
                data-testid="model-check"
              >{{ probeText(modelChecks[item.id] as ProbeResult) }}{{
                failed(modelChecks[item.id]) ? " · не предлагается для новых сессий" : ""
              }}</span>
              <button
                v-if="item.has_base && item.origin === 'ui'"
                type="button"
                class="ui-link ui-small card__link"
                :disabled="!!busy"
                data-testid="model-reset"
                @click="resetModel(item.id)"
              >Сбросить изменения</button>
            </div>
            <div class="ui-card__actions">
              <button
                type="button"
                class="ui-btn ui-btn--outlined"
                :disabled="item.disabled || item.provider_disabled || modelChecks[item.id] === 'checking'"
                title="Пробный запрос генерации; может расходовать подписку"
                data-testid="model-probe"
                @click="checkModel(item)"
              >Проверить ответ</button>
              <button
                type="button"
                class="ui-btn ui-btn--icon ui-btn--danger"
                :aria-label="`Удалить модель ${item.id}`"
                title="Удалить"
                :disabled="!!busy"
                data-testid="model-remove"
                @click="removeModel(item)"
              >
                <Trash2 :size="16" aria-hidden="true" />
              </button>
              <ToggleSwitch
                :model-value="!item.disabled"
                :label="`Модель ${item.id} включена`"
                :disabled="!!busy"
                data-testid="model-toggle"
                @update:model-value="toggleModel(item)"
              />
            </div>
          </div>
          <div v-if="limits[item.id]" class="limits">
            <label class="ui-field">
              <span class="ui-field__label">Контекст, токенов</span>
              <input
                v-model.number="limits[item.id].context_window"
                class="ui-input"
                type="number"
                min="4096"
                step="1024"
                data-testid="model-context"
              />
            </label>
            <label class="ui-field">
              <span class="ui-field__label">Максимум ответа</span>
              <input
                v-model.number="limits[item.id].max_tokens"
                class="ui-input"
                type="number"
                min="1"
                step="256"
                data-testid="model-max-tokens"
              />
            </label>
            <label class="ui-field">
              <span class="ui-field__label">Оплата</span>
              <select v-model="limits[item.id].price" class="ui-input" data-testid="model-price-mode">
                <option v-if="!item.pricing" value="unset" disabled>Не указана</option>
                <option value="free">Без оплаты</option>
                <option value="paid">Платная</option>
              </select>
            </label>
            <template v-if="limits[item.id].price === 'paid'">
              <label class="ui-field">
                <span class="ui-field__label">Вход, $ за 1M</span>
                <input
                  v-model.number="limits[item.id].input"
                  class="ui-input"
                  type="number"
                  min="0"
                  step="0.01"
                  data-testid="model-price-input"
                />
              </label>
              <label class="ui-field">
                <span class="ui-field__label">Выход, $ за 1M</span>
                <input
                  v-model.number="limits[item.id].output"
                  class="ui-input"
                  type="number"
                  min="0"
                  step="0.01"
                  data-testid="model-price-output"
                />
              </label>
            </template>
            <button
              v-if="limitsChanged(item)"
              type="button"
              class="ui-btn ui-btn--filled"
              :disabled="!!limitsError(item) || !!busy"
              data-testid="model-save"
              @click="saveLimits(item)"
            >Сохранить</button>
            <span v-if="limitsChanged(item) && limitsError(item)" class="ui-field__help ui-field__help--error limits__error">
              {{ limitsError(item) }}
            </span>
          </div>
        </article>
      </section>

      <p v-if="!activeModels.length" class="ui-notice" data-testid="models-empty">
        <Info :size="16" aria-hidden="true" />
        <span class="ui-notice__text">
          Нет включённых моделей. Подключите сервер и добавьте модель или включите существующую.
        </span>
      </p>

      <section v-if="activeModels.length" class="ui-section" aria-labelledby="roles-title">
        <h2 id="roles-title" class="ui-section__title">Роли</h2>
        <p class="ui-hint">Какая модель отвечает в каждом режиме. Изменение сохраняется сразу.</p>
        <div class="ui-card roles">
          <label class="ui-field">
            <span class="ui-field__label">Основная модель</span>
            <select
              class="ui-input"
              :value="view.routing.default"
              :disabled="!!busy"
              data-testid="routing-default"
              @change="setDefault(($event.target as HTMLSelectElement).value)"
            >
              <option v-for="item in activeModels" :key="item.id" :value="item.id">{{ item.id }}</option>
            </select>
          </label>
          <label class="ui-field">
            <span class="ui-field__label">Служебная модель (названия, сжатие контекста)</span>
            <select
              class="ui-input"
              :value="view.routing.light ?? ''"
              :disabled="!!busy"
              data-testid="routing-light"
              @change="setLight(($event.target as HTMLSelectElement).value)"
            >
              <option value="">Как основная</option>
              <option v-for="item in activeModels" :key="item.id" :value="item.id">{{ item.id }}</option>
            </select>
          </label>
          <label v-for="role in roleNames" :key="role" class="ui-field">
            <span class="ui-field__label">{{ modeLabel(role) }}</span>
            <select
              class="ui-input"
              :value="view.routing.roles[role] ?? ''"
              :disabled="!!busy"
              :data-routing-role="role"
              @change="setRole(role, ($event.target as HTMLSelectElement).value)"
            >
              <option value="">Как основная</option>
              <option v-for="item in activeModels" :key="item.id" :value="item.id">{{ item.id }}</option>
            </select>
          </label>
          <fieldset class="fallback">
            <legend class="ui-field__label">Запасные модели при сбоях</legend>
            <label v-for="item in activeModels" :key="item.id" class="ui-choice">
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

      <section
        v-if="view.deleted?.providers.length || view.deleted?.models.length"
        class="ui-section"
        aria-labelledby="deleted-title"
        data-testid="deleted-settings"
      >
        <h2 id="deleted-title" class="ui-section__title">Удалённые настройки из файла</h2>
        <p class="ui-hint">Эти настройки заданы в файле моделей и скрыты здесь. Их можно вернуть.</p>
        <div class="ui-card">
          <div v-for="id in view.deleted?.providers" :key="`provider:${id}`" class="found__item">
            <div class="found__info deleted__info">
              <span class="found__name">Сервер {{ id }}</span>
              <span class="ui-card__meta" :data-remove-hint-provider="id">
                Чтобы убрать совсем, удалите блок <code>{{ id }}</code> из раздела <code>providers</code> в config/models.yaml.
              </span>
            </div>
            <div class="found__actions">
              <button
                type="button"
                class="ui-btn ui-btn--outlined"
                :disabled="!!busy"
                :data-restore-provider="id"
                @click="resetProvider(id)"
              >Восстановить</button>
            </div>
          </div>
          <div v-for="id in view.deleted?.models" :key="`model:${id}`" class="found__item">
            <div class="found__info deleted__info">
              <span class="found__name">Модель {{ id }}</span>
              <span class="ui-card__meta" :data-remove-hint-model="id">
                Чтобы убрать совсем, удалите блок <code>{{ id }}</code> из раздела <code>models</code> в config/models.yaml.
              </span>
            </div>
            <div class="found__actions">
              <button
                type="button"
                class="ui-btn ui-btn--outlined"
                :disabled="!!busy"
                :data-restore-model="id"
                @click="resetModel(id)"
              >Восстановить</button>
            </div>
          </div>
        </div>
      </section>
    </template>
    <p v-else-if="!loadError" class="ui-hint">Загрузка…</p>

    <SettingsDialog :open="showForm" title="Подключить сервер" @close="closeForm">
      <form id="provider-form" class="ui-form" novalidate data-testid="provider-form" @submit.prevent="addProvider">
        <div class="ui-tabs presets" role="radiogroup" aria-label="Тип сервера">
          <button
            v-for="item in PROVIDER_PRESETS"
            :key="item.key"
            type="button"
            role="radio"
            :aria-checked="presetKey === item.key"
            :data-preset="item.key"
            @click="choosePreset(item.key)"
          >{{ item.label }}</button>
        </div>
        <label class="ui-field">
          <span class="ui-field__label">Имя</span>
          <input v-model.trim="form.id" class="ui-input" data-autofocus data-testid="provider-id" autocomplete="off" />
        </label>
        <label class="ui-field">
          <span class="ui-field__label">Адрес API</span>
          <input
            v-model.trim="form.baseUrl"
            class="ui-input mono"
            data-testid="provider-url"
            autocomplete="off"
            spellcheck="false"
          />
        </label>
        <label v-if="preset.needsKey" class="ui-field">
          <span class="ui-field__label">Ключ API</span>
          <input v-model="form.apiKey" class="ui-input" type="password" data-testid="provider-key" autocomplete="off" />
        </label>
        <p v-if="preset.key === 'lmstudio'" class="ui-field__help">
          <span>
            Запустите сервер в LM Studio (вкладка Developer или <code>lms server start</code>). Порт сервера
            виден там же, по умолчанию 1234. Длина контекста модели при загрузке должна быть не меньше той,
            что указана здесь.
          </span>
        </p>
        <p v-if="formError" class="ui-field__help ui-field__help--error" data-testid="provider-form-error">{{ formError }}</p>
        <p v-if="actionError" class="ui-notice ui-notice--error" role="alert">
          <CircleAlert :size="16" aria-hidden="true" />
          <span class="ui-notice__text">{{ actionError }}</span>
        </p>
      </form>
      <template #actions>
        <button type="button" class="ui-btn ui-btn--text" :disabled="busy === 'provider:new'" @click="closeForm">
          Отмена
        </button>
        <button
          type="submit"
          form="provider-form"
          class="ui-btn ui-btn--filled"
          :disabled="!!formError || busy === 'provider:new'"
          data-testid="provider-save"
        >
          {{ busy === "provider:new" ? "Подключаю…" : "Подключить и найти модели" }}
        </button>
      </template>
    </SettingsDialog>

    <SettingsDialog :open="!!removal" :title="removal?.title ?? ''" alert @close="closeRemoval">
      <p>{{ removal?.text }}</p>
      <p v-if="actionError" class="ui-notice ui-notice--error" role="alert">
        <CircleAlert :size="16" aria-hidden="true" />
        <span class="ui-notice__text">{{ actionError }}</span>
      </p>
      <template #actions>
        <button type="button" class="ui-btn ui-btn--text" data-autofocus :disabled="!!busy" @click="closeRemoval">
          Отмена
        </button>
        <button
          type="button"
          class="ui-btn ui-btn--filled ui-btn--danger"
          :disabled="!!busy"
          data-testid="remove-confirm"
          @click="confirmRemoval"
        >{{ busy ? "Удаляю…" : "Удалить" }}</button>
      </template>
    </SettingsDialog>
  </section>
</template>

<style scoped>
.ui-card {
  margin-top: 8px;
}

/* A switched off card dims its details too, but keeps the buttons readable. */
.ui-card--off .limits {
  opacity: 0.6;
}

.ui-card--off .ui-card__main > .card__link {
  opacity: 1;
}

/* An optional action sits under the label it belongs to, so every card has the same buttons. */
.card__link {
  align-self: flex-start;
  margin-top: 2px;
}

.local {
  container-type: inline-size;
}

.local__empty {
  margin-top: 6px;
}

.local__list {
  margin-top: 8px;
}

/* A found server: name, address and count in columns, the action at the end. */
.local__row {
  display: grid;
  grid-template-columns: 8.5rem minmax(0, 1fr) 7rem auto;
  align-items: center;
  gap: 4px 12px;
  padding: 4px 0;
  overflow-wrap: anywhere;
}

.local__action {
  justify-self: end;
}

/* Narrow card: name and action on the first line, address and count under them. */
@container (max-width: 520px) {
  .local__row {
    grid-template-columns: minmax(0, 1fr) auto;
  }

  .local__row > :nth-child(2),
  .local__row > :nth-child(3) {
    grid-column: 1;
  }

  .local__action {
    grid-column: 2;
    grid-row: 1;
  }
}

.found {
  margin-top: 12px;
  padding-top: 8px;
  border-top: 1px solid var(--border);
}

.found__item {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 6px 12px;
  padding: 6px 0;
}

.found__item + .found__item {
  border-top: 1px solid var(--border);
}

.found__info {
  display: flex;
  flex: 1 1 20rem;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 2px 12px;
  min-width: 0;
}

.found__name {
  overflow-wrap: anywhere;
}

/* The hint goes under the name, so every restore button keeps the same column. */
.deleted__info {
  flex-direction: column;
  align-items: flex-start;
}

.found__actions {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: flex-end;
  gap: 6px;
  margin-left: auto;
}

/* An edit form inside a card: fields in a row, the buttons at the end. */
.inline-form {
  display: flex;
  flex-wrap: wrap;
  align-items: flex-end;
  gap: 12px;
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid var(--border);
}

.inline-form__wide {
  flex: 1 1 18rem;
  min-width: 0;
}

.inline-form__actions {
  display: flex;
  gap: 8px;
  margin-left: auto;
}

.limits {
  display: flex;
  flex-wrap: wrap;
  align-items: flex-end;
  gap: 12px;
  margin-top: 12px;
}

.limits .ui-input {
  width: 9rem;
}

.limits__error {
  align-self: center;
}

.roles {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.roles .ui-input {
  max-width: 420px;
}

.fallback {
  display: flex;
  flex-direction: column;
  margin: 0;
  padding: 0;
  border: 0;
}

.fallback legend {
  margin-bottom: 4px;
}

.presets {
  align-self: flex-start;
}

code {
  font-size: 0.92em;
}
</style>
