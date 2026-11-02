<script setup lang="ts">
// Model settings: connections to model servers, the models the agent may use and the
// model for each role. Changes are stored on the server on top of the models file.

import { computed, onBeforeUnmount, onMounted, reactive, ref, watch } from "vue";
import { ArrowLeft, Plus, Search } from "lucide-vue-next";
import { useRouter } from "vue-router";
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
import { useAppStore } from "../stores/app";
import { SESSION_MODES, modeLabel } from "../modes";

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

async function removeProvider(item: ProviderView): Promise<boolean> {
  const linked = (view.value?.models ?? []).filter((model) => model.provider === item.id).length;
  const question = linked
    ? `Удалить подключение «${item.id}»? Вместе с ним удалятся его модели: ${linked}.`
    : `Удалить подключение «${item.id}»?`;
  if (!window.confirm(question)) return false;
  const saved = await apply(`provider:${item.id}`, () => modelSettings.removeProvider(item.id, saveOptions()));
  if (saved) {
    delete discoveries[item.id];
    delete keys[item.id];
    delete addresses[item.id];
  }
  return saved;
}

function removeModel(item: ModelView): Promise<boolean> {
  if (!window.confirm(`Удалить модель «${item.id}»?`)) return Promise.resolve(false);
  return apply(`model:${item.id}`, () => modelSettings.removeModel(item.id, saveOptions()));
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

function back(): void {
  if (window.history.length > 1) router.back();
  else void router.push({ name: "home" });
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
  <section class="settings" data-testid="models-settings">
    <header class="settings__header">
      <button type="button" class="ghost" data-testid="settings-back" @click="back">
        <ArrowLeft :size="16" aria-hidden="true" />
        <span>Назад</span>
      </button>
      <h1>Модели</h1>
      <nav class="settings__nav" aria-label="Разделы настроек">
        <RouterLink :to="{ name: 'settings-models' }" aria-current="page">Модели</RouterLink>
        <RouterLink :to="{ name: 'settings-speech' }" data-testid="settings-speech-link">Голос</RouterLink>
      </nav>
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

        <div class="card local" data-testid="local-servers">
          <div class="card__row">
            <strong>На этом компьютере</strong>
            <button type="button" class="ghost" :disabled="scanning" data-testid="local-scan" @click="scan">
              <Search :size="15" aria-hidden="true" />
              <span>{{ scanning ? "Ищу…" : "Искать снова" }}</span>
            </button>
          </div>
          <p v-if="localServers && !localServers.length" class="muted small">
            Запущенных серверов моделей не найдено. Запустите сервер в LM Studio (<code>lms server start</code>)
            или Ollama и нажмите «Искать снова».
          </p>
          <ul v-else-if="localServers" class="local__list">
            <li v-for="server in localServers" :key="server.base_url" class="found__row" :data-local="server.base_url">
              <span>{{ SERVER_NAMES[server.server] }}</span>
              <span class="mono muted small">{{ server.base_url }}</span>
              <span class="muted small">моделей: {{ server.models }}</span>
              <span v-if="server.provider" class="muted small found__state">Подключён как {{ server.provider }}</span>
              <button
                v-else
                type="button"
                class="ghost found__state"
                :disabled="!!busy"
                data-testid="local-connect"
                @click="connectLocal(server)"
              >
                <Plus :size="14" aria-hidden="true" />
                <span>Подключить</span>
              </button>
            </li>
          </ul>
        </div>

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
            Запустите сервер в LM Studio (вкладка Developer или <code>lms server start</code>). Порт сервера
            виден там же, по умолчанию 1234. Длина контекста модели при загрузке должна быть не меньше той,
            что указана здесь.
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
              <span class="muted small">{{ keyStatus(item) }} · {{ originLabel(item.origin, item.has_base) }}</span>
              <span v-if="item.key_source" class="muted small" data-testid="provider-key-origin">
                Источник ключа: {{ item.key_source === 'ui' ? 'сохранён в приложении' : item.key_source === 'env' ? 'окружение' : 'без ключа' }}
              </span>
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
              <button
                type="button"
                class="ghost"
                :disabled="!!busy"
                data-testid="provider-edit-url"
                @click="addresses[item.id] = item.base_url"
              >Изменить адрес</button>
              <button type="button" class="ghost" :disabled="!!busy" data-testid="provider-edit-key" @click="editKey(item)">Изменить ключ</button>
              <button type="button" class="ghost" :disabled="!!busy" data-testid="provider-toggle" @click="toggleProvider(item)">
                {{ item.disabled ? "Включить" : "Отключить" }}
              </button>
              <button v-if="item.has_base && item.origin === 'ui'" type="button" class="ghost" :disabled="!!busy" data-testid="provider-reset" @click="resetProvider(item.id)">Сбросить изменения</button>
              <button
                type="button"
                class="ghost danger"
                :disabled="!!busy"
                data-testid="provider-remove"
                @click="removeProvider(item)"
              >Удалить</button>
            </div>
          </div>

          <form
            v-if="addresses[item.id] !== undefined"
            class="address"
            data-testid="provider-url-form"
            @submit.prevent="saveAddress(item)"
          >
            <input v-model.trim="addresses[item.id]" class="mono" aria-label="Адрес API" data-testid="provider-url-edit" />
            <button type="submit" class="primary" :disabled="!!addressError(addresses[item.id]) || !!busy">Сохранить</button>
            <button type="button" class="ghost" @click="delete addresses[item.id]">Отмена</button>
            <span v-if="addressError(addresses[item.id])" class="field-error">{{ addressError(addresses[item.id]) }}</span>
          </form>

          <form v-if="keys[item.id]" class="key-editor" data-testid="provider-key-form" @submit.prevent="saveKey(item)">
            <label>
              <span>Источник ключа</span>
              <select v-model="keys[item.id].source" data-testid="provider-key-source">
                <option value="ui">Сохранить в приложении</option>
                <option value="env">Из окружения</option>
                <option value="none">Без ключа</option>
              </select>
            </label>
            <label v-if="keys[item.id].source === 'ui'">
              <span>Новый ключ</span>
              <input v-model="keys[item.id].value" type="password" autocomplete="new-password" data-testid="provider-key-value" />
              <span v-if="item.key_source === 'ui' && item.api_key_set" class="hint">Оставьте пустым, чтобы сохранить текущий ключ.</span>
            </label>
            <label v-if="keys[item.id].source === 'env'">
              <span>Переменная окружения</span>
              <input v-model.trim="keys[item.id].env" autocomplete="off" data-testid="provider-key-env" />
            </label>
            <p v-if="keys[item.id].source === 'none'" class="hint">Сохранённый ключ будет очищен. Ключ из окружения использоваться не будет.</p>
            <p v-if="keyError(item)" class="field-error">{{ keyError(item) }}</p>
            <div class="actions">
              <button type="submit" class="primary" :disabled="!!busy || !!keyError(item)">Сохранить ключ</button>
              <button type="button" class="ghost" @click="delete keys[item.id]">Отмена</button>
            </div>
          </form>

          <div v-if="discoveries[item.id] && !discoveries[item.id].loading" class="found" data-testid="found-models">
            <p v-if="discoveries[item.id].message" class="muted small">{{ discoveries[item.id].message }}</p>
            <template v-else>
              <ul>
                <li v-for="found in discoveries[item.id].found" :key="found.id" class="found__item" :data-found="found.id">
                  <div class="found__info">
                    <span class="mono found__name">{{ found.id }}</span>
                    <span class="muted small">{{ describeFound(found) }}</span>
                    <span v-if="found.state" class="muted small">
                      {{ found.state === "loaded" ? "загружена в память" : "не загружена" }}
                    </span>
                    <span v-if="foundCheck(item, found) === 'checking'" class="muted small">Проверяю…</span>
                    <span v-else-if="answered(foundCheck(item, found))" class="small ok" data-testid="found-check">
                      {{ probeText(foundCheck(item, found) as ProbeResult) }}
                    </span>
                    <span v-else-if="failed(foundCheck(item, found))" class="small bad" data-testid="found-check">
                      {{ probeText(foundCheck(item, found) as ProbeResult) }}
                    </span>
                  </div>
                  <div v-if="addedAs(view.models, item.id, found.id)" class="found__actions muted small">
                    Добавлена как {{ addedAs(view.models, item.id, found.id) }}
                  </div>
                  <div v-else-if="canChat(found)" class="found__actions">
                    <button
                      type="button"
                      class="ghost"
                      title="Пробный запрос генерации; может расходовать подписку"
                      data-testid="found-probe"
                      @click="checkFound(item, found)"
                      :disabled="foundCheck(item, found) === 'checking'"
                    >Проверить ответ</button>
                    <button
                      type="button"
                      class="ghost"
                      :disabled="!!busy"
                      data-testid="found-add"
                      @click="addModel(item, found)"
                    >
                      <Plus :size="14" aria-hidden="true" />
                      <span>Добавить</span>
                    </button>
                  </div>
                </li>
              </ul>
            </template>
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
          :class="{ 'card--off': item.disabled || item.provider_disabled }"
          data-testid="model-card"
          :data-model="item.id"
        >
          <div class="card__row">
            <div class="card__main">
              <strong>{{ item.id }}</strong>
              <span class="muted small">
                <span class="mono">{{ item.model }}</span> · {{ item.provider }} · {{ originLabel(item.origin, item.has_base) }}
              </span>
              <span v-if="item.provider_disabled" class="small muted" data-testid="model-provider-off">Сервер отключён</span>
              <span class="small muted" data-testid="model-price">{{ priceLabel(item) }}</span>
              <span v-if="!item.pricing && costLimit" class="small bad" data-testid="model-price-warning">
                У сессий есть лимит стоимости ${{ costLimit }}, поэтому модель без цены не запустится.
                Выберите «Без оплаты» или укажите цену.
              </span>
              <span v-if="rolesOf(view.routing, item.id).length" class="small" data-testid="model-roles">
                Роль: {{ rolesOf(view.routing, item.id).join(", ") }}
              </span>
              <span v-if="modelChecks[item.id] === 'checking'" class="muted small">Проверяю…</span>
              <span
                v-else-if="modelChecks[item.id]"
                class="small"
                :class="answered(modelChecks[item.id]) ? 'ok' : 'bad'"
                data-testid="model-check"
              >{{ probeText(modelChecks[item.id] as ProbeResult) }}{{
                failed(modelChecks[item.id]) ? " · не предлагается для новых сессий" : ""
              }}</span>
            </div>
            <div class="card__actions">
              <button
                type="button"
                class="ghost"
                :disabled="item.disabled || item.provider_disabled || modelChecks[item.id] === 'checking'"
                title="Пробный запрос генерации; может расходовать подписку"
                data-testid="model-probe"
                @click="checkModel(item)"
              >Проверить ответ</button>
              <button
                type="button"
                class="ghost"
                :disabled="!!busy"
                data-testid="model-toggle"
                @click="toggleModel(item)"
              >
                {{ item.disabled ? "Включить" : "Отключить" }}
              </button>
              <button v-if="item.has_base && item.origin === 'ui'" type="button" class="ghost" :disabled="!!busy" data-testid="model-reset" @click="resetModel(item.id)">Сбросить изменения</button>
              <button
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
            <label>
              <span>Оплата</span>
              <select v-model="limits[item.id].price" data-testid="model-price-mode">
                <option v-if="!item.pricing" value="unset" disabled>Не указана</option>
                <option value="free">Без оплаты</option>
                <option value="paid">Платная</option>
              </select>
            </label>
            <template v-if="limits[item.id].price === 'paid'">
              <label>
                <span>Вход, $ за 1M</span>
                <input v-model.number="limits[item.id].input" type="number" min="0" step="0.01" data-testid="model-price-input" />
              </label>
              <label>
                <span>Выход, $ за 1M</span>
                <input v-model.number="limits[item.id].output" type="number" min="0" step="0.01" data-testid="model-price-output" />
              </label>
            </template>
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

      <p v-if="!activeModels.length" class="notice" data-testid="models-empty">Нет включённых моделей. Подключите сервер и добавьте модель или включите существующую.</p>

      <section v-if="view.deleted?.providers.length || view.deleted?.models.length" class="block" data-testid="deleted-settings">
        <h2>Удалённые настройки из файла</h2>
        <p class="hint">Они не используются. Восстановление явно вернёт файловые настройки.</p>
        <div v-for="id in view.deleted?.providers" :key="`provider:${id}`" class="found__row">
          <span>Сервер {{ id }}</span>
          <button type="button" class="ghost" :disabled="!!busy" :data-restore-provider="id" @click="resetProvider(id)">Восстановить</button>
        </div>
        <div v-for="id in view.deleted?.models" :key="`model:${id}`" class="found__row">
          <span>Модель {{ id }}</span>
          <button type="button" class="ghost" :disabled="!!busy" :data-restore-model="id" @click="resetModel(id)">Восстановить</button>
        </div>
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
          <label v-for="role in roleNames" :key="role">
            <span>{{ modeLabel(role) }}</span>
            <select :value="view.routing.roles[role] ?? ''" :disabled="!!busy" :data-routing-role="role" @change="setRole(role, ($event.target as HTMLSelectElement).value)">
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
  flex-wrap: wrap;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
}

.settings__nav {
  display: flex;
  gap: 4px;
  margin-left: auto;
}

.settings__nav a {
  padding: 4px 10px;
  border-radius: var(--radius-sm);
  color: var(--text-muted);
  font-size: 13px;
  text-decoration: none;
}

.settings__nav a[aria-current="page"] {
  background: var(--bg-subtle);
  color: var(--text);
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

/* Dim the details of a switched off item, but keep its buttons readable. */
.card--off .card__main,
.card--off .limits {
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
  border-color: color-mix(in srgb, var(--danger) 50%, var(--border));
  color: var(--danger);
}

button.danger:not(:disabled):hover {
  background: color-mix(in srgb, var(--danger) 10%, transparent);
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
.key-editor,
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

.ok {
  color: var(--success);
}

.bad {
  color: var(--danger);
}

.local {
  margin-bottom: 4px;
}

.local__list {
  margin: 8px 0 0;
  padding: 0;
  list-style: none;
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

/* A found model: details on the left, actions in one group on the right. */
.found__item {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 6px 12px;
  padding: 6px 0;
}

.found__item + .found__item {
  border-top: 1px dashed var(--border);
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

.found__actions {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 6px;
  margin-left: auto;
}

.address {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  margin-top: 10px;
}

.address input {
  flex: 1 1 18rem;
  min-width: 0;
}

.key-editor { margin-top: 12px; }

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

.limits input,
.limits select {
  box-sizing: border-box;
  height: 34px;
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
