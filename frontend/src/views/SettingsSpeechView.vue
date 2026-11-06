<script setup lang="ts">
// Speech recognition settings: local server profiles in the order they are tried.

import { computed, onMounted, reactive, ref } from "vue";
import { useRouter } from "vue-router";
import { ArrowLeft, Plus } from "lucide-vue-next";
import { speechSettings, type SpeechCheck, type SpeechProfile } from "../api/speech";

const router = useRouter();

const profiles = ref<SpeechProfile[] | null>(null);
const loadError = ref("");
const actionError = ref("");
const busy = ref(false);
const showForm = ref(false);
const checks = reactive<Record<string, SpeechCheck | "checking">>({});
const form = reactive({ id: "local", baseUrl: "http://127.0.0.1:8178/v1", language: "", model: "" });

const LANGUAGES = [
  { value: "", label: "Определять автоматически" },
  { value: "ru", label: "Русский" },
  { value: "en", label: "Английский" },
];

// The same rules as the server: a short ID and a server on this computer only.
const PROFILE_ID = /^[A-Za-z0-9._-]{1,100}$/;
const LOCAL_HOSTS = new Set(["localhost", "127.0.0.1", "[::1]"]);

function isLocalAddress(url: string): boolean {
  try {
    const parsed = new URL(url);
    return (parsed.protocol === "http:" || parsed.protocol === "https:") && LOCAL_HOSTS.has(parsed.hostname);
  } catch {
    return false;
  }
}

const formError = computed(() => {
  if (!PROFILE_ID.test(form.id)) {
    return "ID: латинские буквы, цифры, точка, дефис или подчёркивание";
  }
  if (profiles.value?.some((item) => item.id === form.id)) return "Профиль с таким ID уже есть";
  if (!isLocalAddress(form.baseUrl)) return "Адрес должен указывать на этот компьютер (127.0.0.1 или localhost)";
  return "";
});

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function roleOf(item: SpeechProfile): string {
  if (item.disabled) return "Отключён";
  const enabled = (profiles.value ?? []).filter((profile) => !profile.disabled);
  const position = enabled.findIndex((profile) => profile.id === item.id);
  return position === 0 ? "Основной" : `Запасной ${position}`;
}

function languageLabel(value: string): string {
  return LANGUAGES.find((item) => item.value === value)?.label ?? value;
}

function checkText(result: SpeechCheck | "checking" | undefined): string {
  if (result === undefined) return "";
  if (result === "checking") return "Проверяю…";
  if (result.ok) return `Отвечает${result.seconds === null ? "" : ` за ${result.seconds.toLocaleString("ru-RU")} с`}`;
  return result.status === null ? "Сервер недоступен" : `Не отвечает: HTTP ${result.status}`;
}

async function load(): Promise<void> {
  loadError.value = "";
  try {
    profiles.value = (await speechSettings.get()).profiles;
  } catch (error) {
    loadError.value = `Не удалось загрузить настройки: ${message(error)}`;
  }
}

async function run(action: () => Promise<{ profiles: SpeechProfile[] }>): Promise<boolean> {
  busy.value = true;
  actionError.value = "";
  try {
    profiles.value = (await action()).profiles;
    return true;
  } catch (error) {
    actionError.value = `Не удалось сохранить: ${message(error)}`;
    return false;
  } finally {
    busy.value = false;
  }
}

async function add(): Promise<void> {
  if (formError.value || busy.value) return;
  const saved = await run(() =>
    speechSettings.putProfile(form.id, { base_url: form.baseUrl, language: form.language, model: form.model }),
  );
  if (saved) showForm.value = false;
}

async function check(item: SpeechProfile): Promise<void> {
  checks[item.id] = "checking";
  try {
    checks[item.id] = await speechSettings.testProfile(item.id);
  } catch (error) {
    checks[item.id] = { ok: false, status: null, seconds: null, message: message(error) };
  }
}

function remove(item: SpeechProfile): void {
  if (!window.confirm(`Удалить профиль распознавания «${item.id}»?`)) return;
  delete checks[item.id];
  void run(() => speechSettings.removeProfile(item.id));
}

function back(): void {
  if (window.history.length > 1) router.back();
  else void router.push({ name: "home" });
}

onMounted(load);
</script>

<template>
  <section class="settings" data-testid="speech-settings">
    <header class="settings__header">
      <button type="button" class="ghost" data-testid="settings-back" @click="back">
        <ArrowLeft :size="16" aria-hidden="true" />
        <span>Назад</span>
      </button>
      <h1>Голос</h1>
      <nav class="settings__nav" aria-label="Разделы настроек">
        <RouterLink :to="{ name: 'settings-models' }">Модели</RouterLink>
        <RouterLink :to="{ name: 'settings-speech' }" aria-current="page">Голос</RouterLink>
        <RouterLink :to="{ name: 'settings-skills' }">Навыки</RouterLink>
      </nav>
    </header>

    <p v-if="loadError" class="notice notice--error" role="alert">
      {{ loadError }}
      <button type="button" class="link" @click="load">Повторить</button>
    </p>
    <p v-if="actionError" class="notice notice--error" role="alert" data-testid="speech-error">{{ actionError }}</p>

    <template v-if="profiles">
      <section class="block" aria-labelledby="speech-title">
        <div class="block__head">
          <h2 id="speech-title">Распознавание речи</h2>
          <button type="button" class="ghost" data-testid="speech-add-open" @click="showForm = !showForm">
            <Plus :size="15" aria-hidden="true" />
            <span>Добавить сервер</span>
          </button>
        </div>
        <p class="hint">
          Речь распознаёт сервер на этом компьютере, запись не уходит наружу и не сохраняется. Нужен
          OpenAI-совместимый сервер распознавания. Первый включённый профиль основной; если он не отвечает,
          используется следующий.
        </p>

        <form v-if="showForm" class="card form" data-testid="speech-form" @submit.prevent="add">
          <label>
            <span>ID профиля</span>
            <input v-model.trim="form.id" class="mono" data-testid="speech-id" />
          </label>
          <label>
            <span>Адрес API</span>
            <input v-model.trim="form.baseUrl" class="mono" data-testid="speech-url" />
          </label>
          <label>
            <span>Язык речи</span>
            <select v-model="form.language" data-testid="speech-language">
              <option v-for="item in LANGUAGES" :key="item.value" :value="item.value">{{ item.label }}</option>
            </select>
          </label>
          <label>
            <span>Модель на сервере (если сервер её просит)</span>
            <input v-model.trim="form.model" class="mono" data-testid="speech-model" />
          </label>
          <p v-if="formError" class="field-error" data-testid="speech-form-error">{{ formError }}</p>
          <div class="actions">
            <button type="submit" class="primary" :disabled="!!formError || busy" data-testid="speech-save">Сохранить</button>
            <button type="button" class="ghost" @click="showForm = false">Отмена</button>
          </div>
        </form>

        <p v-if="!profiles.length && !showForm" class="muted" data-testid="speech-empty">
          Серверов распознавания нет. Добавьте локальный сервер, и в поле ввода заработает микрофон.
        </p>

        <article
          v-for="item in profiles"
          :key="item.id"
          class="card"
          :class="{ 'card--off': item.disabled }"
          :data-profile="item.id"
        >
          <div class="card__row">
            <div class="card__main">
              <strong>{{ item.id }}</strong>
              <span class="mono muted small">{{ item.base_url }}</span>
              <span class="muted small">
                {{ roleOf(item) }} · {{ languageLabel(item.language) }}{{ item.model ? ` · модель ${item.model}` : "" }}
              </span>
              <button
                v-if="roleOf(item) !== 'Основной' && !item.disabled"
                type="button"
                class="link small card__link"
                :disabled="busy"
                data-testid="speech-primary"
                @click="run(() => speechSettings.makePrimary(item.id))"
              >Сделать основным</button>
              <span
                v-if="checks[item.id]"
                class="small"
                :class="{ ok: checks[item.id] !== 'checking' && (checks[item.id] as SpeechCheck).ok, bad: checks[item.id] !== 'checking' && !(checks[item.id] as SpeechCheck).ok }"
                data-testid="speech-check"
              >{{ checkText(checks[item.id]) }}</span>
            </div>
            <div class="card__actions">
              <button type="button" class="ghost" :disabled="checks[item.id] === 'checking'" data-testid="speech-test" @click="check(item)">Проверить</button>
              <button
                type="button"
                class="ghost toggle"
                :disabled="busy"
                data-testid="speech-toggle"
                @click="run(() => speechSettings.putProfile(item.id, { disabled: !item.disabled }))"
              >{{ item.disabled ? "Включить" : "Отключить" }}</button>
              <button type="button" class="ghost danger" :disabled="busy" data-testid="speech-remove" @click="remove(item)">Удалить</button>
            </div>
          </div>
        </article>
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
  flex-wrap: wrap;
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
.card--off .card__main > :not(.card__link) {
  opacity: 0.6;
}

/* An optional action sits under the label it belongs to, so every card has the same buttons. */
.card__link {
  align-self: flex-start;
  min-height: 0;
  height: auto;
  padding: 0;
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

/* Buttons keep to the right edge; a wrapped group stays there too. */
.card__actions {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 6px;
  margin-left: auto;
}

/* "Включить" and "Отключить" take the same width, so the next buttons do not move. */
.toggle {
  min-width: 7.2em;
  justify-content: center;
}

.small {
  font-size: 12px;
}

button.ghost,
button.primary {
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

.form {
  display: flex;
  flex-direction: column;
  gap: 10px;
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

.ok {
  color: var(--success);
}

.bad {
  color: var(--danger);
}
</style>
