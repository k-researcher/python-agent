<script setup lang="ts">
// Speech recognition settings: local server profiles in the order they are tried.

import { computed, onMounted, reactive, ref } from "vue";
import { CircleAlert, Plus, Trash2 } from "lucide-vue-next";
import { speechSettings, type SpeechCheck, type SpeechProfile } from "../api/speech";
import SettingsDialog from "../components/settings/SettingsDialog.vue";
import SettingsHeader from "../components/settings/SettingsHeader.vue";
import ToggleSwitch from "../components/settings/ToggleSwitch.vue";

const profiles = ref<SpeechProfile[] | null>(null);
const loadError = ref("");
const actionError = ref("");
const busy = ref(false);
const showForm = ref(false);
type CheckState = SpeechCheck | "checking" | "restarting";
const checks = reactive<Record<string, CheckState>>({});
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

function checkText(result: CheckState | undefined): string {
  if (result === undefined) return "";
  if (result === "checking") return "Проверяю…";
  if (result === "restarting") return "Перезапускаю…";
  // A restart reports its own failure: the command failed or the server did not come back.
  if (!result.ok && result.restarted !== undefined && result.message) return result.message;
  if (result.ok) {
    const time = result.seconds === null ? "" : ` за ${result.seconds.toLocaleString("ru-RU")} с`;
    return `${result.restarted ? "Перезапущен, отвечает" : "Отвечает"}${time}`;
  }
  const failure = result.status === null ? "сервер недоступен" : `не отвечает: HTTP ${result.status}`;
  return result.restarted ? `Перезапущен, но ${failure}` : failure[0].toUpperCase() + failure.slice(1);
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

function pending(id: string): boolean {
  return checks[id] === "checking" || checks[id] === "restarting";
}

function finished(id: string): SpeechCheck | null {
  const result = checks[id];
  return result === undefined || result === "checking" || result === "restarting" ? null : result;
}

// The server runs the command from its own configuration and then checks the profile.
async function restart(item: SpeechProfile): Promise<void> {
  checks[item.id] = "restarting";
  try {
    checks[item.id] = await speechSettings.restartProfile(item.id);
  } catch (error) {
    checks[item.id] = { ok: false, status: null, seconds: null, restarted: false, message: message(error) };
  }
}

function openForm(): void {
  Object.assign(form, { id: profiles.value?.length ? "" : "local", baseUrl: "http://127.0.0.1:8178/v1", language: "", model: "" });
  actionError.value = "";
  showForm.value = true;
}

function closeForm(): void {
  if (!busy.value) showForm.value = false;
}

const removing = ref<SpeechProfile | null>(null);

async function remove(): Promise<void> {
  const item = removing.value;
  if (!item) return;
  delete checks[item.id];
  if (await run(() => speechSettings.removeProfile(item.id))) removing.value = null;
}

onMounted(load);
</script>

<template>
  <section class="ui-page" data-testid="speech-settings">
    <SettingsHeader title="Голос" />

    <p v-if="loadError" class="ui-notice ui-notice--error" role="alert">
      <CircleAlert :size="16" aria-hidden="true" />
      <span class="ui-notice__text">{{ loadError }}</span>
      <button type="button" class="ui-link" @click="load">Повторить</button>
    </p>
    <p v-if="actionError && !showForm && !removing" class="ui-notice ui-notice--error" role="alert" data-testid="speech-error">
      <CircleAlert :size="16" aria-hidden="true" />
      <span class="ui-notice__text">{{ actionError }}</span>
    </p>

    <section v-if="profiles" class="ui-section" aria-labelledby="speech-title">
      <div class="ui-section__head">
        <h2 id="speech-title" class="ui-section__title">Распознавание речи</h2>
        <div class="ui-section__actions">
          <button type="button" class="ui-btn ui-btn--filled" data-testid="speech-add-open" @click="openForm">
            <Plus :size="16" aria-hidden="true" />
            <span>Добавить сервер</span>
          </button>
        </div>
      </div>
      <p class="ui-hint">
        Речь распознаёт сервер на этом компьютере, запись не уходит наружу и не сохраняется. Нужен
        OpenAI-совместимый сервер распознавания. Первый включённый профиль основной; если он не отвечает,
        используется следующий.
      </p>

      <p v-if="!profiles.length" class="ui-empty" data-testid="speech-empty">
        Серверов распознавания нет. Добавьте локальный сервер, и в поле ввода заработает микрофон.
      </p>

      <article
        v-for="item in profiles"
        :key="item.id"
        class="ui-card ui-card__row"
        :class="{ 'ui-card--off': item.disabled }"
        :data-profile="item.id"
      >
        <div class="ui-card__main">
          <strong class="ui-card__title">{{ item.id }}</strong>
          <span class="mono ui-card__meta">{{ item.base_url }}</span>
          <span class="ui-card__meta">
            {{ roleOf(item) }} · {{ languageLabel(item.language) }}{{ item.model ? ` · модель ${item.model}` : "" }}
          </span>
          <span
            v-if="checks[item.id]"
            class="ui-small"
            :class="{
              'ui-ok': finished(item.id)?.ok === true,
              'ui-bad': finished(item.id)?.ok === false,
            }"
            data-testid="speech-check"
          >{{ checkText(checks[item.id]) }}</span>
          <button
            v-if="roleOf(item) !== 'Основной' && !item.disabled"
            type="button"
            class="ui-link ui-small primary-link"
            :disabled="busy"
            data-testid="speech-primary"
            @click="run(() => speechSettings.makePrimary(item.id))"
          >Сделать основным</button>
        </div>
        <div class="ui-card__actions">
          <button
            type="button"
            class="ui-btn ui-btn--outlined"
            :disabled="pending(item.id)"
            data-testid="speech-test"
            @click="check(item)"
          >Проверить</button>
          <button
            v-if="item.restart"
            type="button"
            class="ui-btn ui-btn--outlined"
            :disabled="pending(item.id)"
            data-testid="speech-restart"
            @click="restart(item)"
          >Перезапустить</button>
          <button
            type="button"
            class="ui-btn ui-btn--icon ui-btn--danger"
            :aria-label="`Удалить профиль ${item.id}`"
            title="Удалить"
            :disabled="busy"
            data-testid="speech-remove"
            @click="removing = item"
          >
            <Trash2 :size="16" aria-hidden="true" />
          </button>
          <ToggleSwitch
            :model-value="!item.disabled"
            :label="`Профиль ${item.id} включён`"
            :disabled="busy"
            data-testid="speech-toggle"
            @update:model-value="run(() => speechSettings.putProfile(item.id, { disabled: !$event }))"
          />
        </div>
      </article>
    </section>
    <p v-else-if="!loadError" class="ui-hint">Загрузка…</p>

    <SettingsDialog :open="showForm" title="Новый сервер распознавания" @close="closeForm">
      <form id="speech-form" class="ui-form" novalidate data-testid="speech-form" @submit.prevent="add">
        <label class="ui-field">
          <span class="ui-field__label">ID профиля</span>
          <input v-model.trim="form.id" class="ui-input mono" autocomplete="off" spellcheck="false" data-autofocus data-testid="speech-id" />
        </label>
        <label class="ui-field">
          <span class="ui-field__label">Адрес API</span>
          <input v-model.trim="form.baseUrl" class="ui-input mono" autocomplete="off" spellcheck="false" data-testid="speech-url" />
        </label>
        <label class="ui-field">
          <span class="ui-field__label">Язык речи</span>
          <select v-model="form.language" class="ui-input" data-testid="speech-language">
            <option v-for="item in LANGUAGES" :key="item.value" :value="item.value">{{ item.label }}</option>
          </select>
        </label>
        <label class="ui-field">
          <span class="ui-field__label">Модель на сервере</span>
          <input v-model.trim="form.model" class="ui-input mono" autocomplete="off" spellcheck="false" data-testid="speech-model" />
          <small class="ui-field__help">Нужна, только если сервер её просит.</small>
        </label>
        <p v-if="form.id && formError" class="ui-field__help ui-field__help--error" data-testid="speech-form-error">{{ formError }}</p>
        <p v-if="actionError" class="ui-notice ui-notice--error" role="alert">
          <CircleAlert :size="16" aria-hidden="true" />
          <span class="ui-notice__text">{{ actionError }}</span>
        </p>
      </form>
      <template #actions>
        <button type="button" class="ui-btn ui-btn--text" :disabled="busy" @click="closeForm">Отмена</button>
        <button type="submit" form="speech-form" class="ui-btn ui-btn--filled" :disabled="!!formError || busy" data-testid="speech-save">
          {{ busy ? "Сохраняю…" : "Добавить" }}
        </button>
      </template>
    </SettingsDialog>

    <SettingsDialog
      :open="!!removing"
      :title="removing ? `Удалить профиль «${removing.id}»?` : ''"
      alert
      @close="busy || (removing = null)"
    >
      <p>Микрофон перестанет использовать этот сервер. Сам сервер не меняется.</p>
      <p v-if="actionError" class="ui-notice ui-notice--error" role="alert">
        <CircleAlert :size="16" aria-hidden="true" />
        <span class="ui-notice__text">{{ actionError }}</span>
      </p>
      <template #actions>
        <button type="button" class="ui-btn ui-btn--text" data-autofocus :disabled="busy" @click="removing = null">Отмена</button>
        <button
          type="button"
          class="ui-btn ui-btn--filled ui-btn--danger"
          :disabled="busy"
          data-testid="speech-remove-confirm"
          @click="remove"
        >{{ busy ? "Удаляю…" : "Удалить" }}</button>
      </template>
    </SettingsDialog>
  </section>
</template>

<style scoped>
/* An optional action sits under the label it belongs to, so every card has the same buttons. */
.primary-link {
  align-self: flex-start;
  margin-top: 2px;
}
</style>
