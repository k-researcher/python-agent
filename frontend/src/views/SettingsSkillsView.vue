<script setup lang="ts">
// Skills: shared skills serve every project, project skills serve only their project.
// A switch acts at once. The shared tab sets the default for all projects; the project tab
// sets the choice of one project and shows the shared skills next to its own.

import { computed, onMounted, reactive, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import { ArrowLeft, CircleAlert, FileUp, Pencil, Plus, Trash2, X } from "lucide-vue-next";
import { api } from "../api/client";
import type { Project } from "../api/types";
import {
  MAX_DESCRIPTION,
  MAX_IMPORT_BYTES,
  SKILL_NAME,
  skillsApi,
  type SkillCatalog,
  type SkillItem,
  type SkillScope,
} from "../api/skills";
import SettingsDialog from "../components/settings/SettingsDialog.vue";
import ToggleSwitch from "../components/settings/ToggleSwitch.vue";
import { useAppStore } from "../stores/app";

const router = useRouter();
const route = useRoute();
const appStore = useAppStore();

const tab = ref<SkillScope>(route.query.tab === "project" ? "project" : "shared");
const projects = ref<Project[]>([]);
const projectId = ref<string | null>(null);
const catalog = ref<SkillCatalog | null>(null);
const loadError = ref("");
const actionError = ref("");
const status = ref("");
const pending = ref("");
const importing = ref(false);
const fileInput = ref<HTMLInputElement | null>(null);

const project = computed(() => projects.value.find((item) => item.id === projectId.value) ?? null);
const shared = computed(() => catalog.value?.skills.filter((item) => item.scope === "shared") ?? []);
const own = computed(() => catalog.value?.skills.filter((item) => item.scope === "project") ?? []);
const brokenCount = computed(() => catalog.value?.errors.length ?? 0);

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function plural(count: number, one: string, few: string, many: string): string {
  const tens = count % 100;
  const units = count % 10;
  if (tens >= 11 && tens <= 14) return many;
  if (units === 1) return one;
  if (units >= 2 && units <= 4) return few;
  return many;
}

function filesText(count: number): string {
  return count ? `${count} ${plural(count, "файл", "файла", "файлов")} кроме SKILL.md` : "";
}

async function loadCatalog(): Promise<void> {
  loadError.value = "";
  try {
    catalog.value = await skillsApi.list(projectId.value);
  } catch (error) {
    loadError.value = `Не удалось загрузить навыки: ${message(error)}`;
  }
}

async function load(): Promise<void> {
  loadError.value = "";
  try {
    projects.value = await api.projects();
  } catch (error) {
    loadError.value = `Не удалось загрузить проекты: ${message(error)}`;
    return;
  }
  const wanted = typeof route.query.project === "string" ? route.query.project : appStore.selectedProjectId;
  projectId.value = projects.value.find((item) => item.id === wanted)?.id ?? projects.value[0]?.id ?? null;
  await loadCatalog();
}

watch(projectId, (next, previous) => {
  if (previous !== null && next !== previous) {
    status.value = "";
    void loadCatalog();
  }
});

watch(tab, (next) => {
  status.value = "";
  actionError.value = "";
  void router.replace({ query: { ...route.query, tab: next } });
});

// Arrow keys move between the two tabs, as in a native segmented control.
function tabKey(event: KeyboardEvent): void {
  if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
  event.preventDefault();
  tab.value = tab.value === "shared" ? "project" : "shared";
  document.getElementById(`skills-tab-${tab.value}`)?.focus();
}

async function change(key: string, action: () => Promise<SkillCatalog>, done = ""): Promise<boolean> {
  pending.value = key;
  actionError.value = "";
  status.value = "";
  try {
    catalog.value = await action();
    status.value = done;
    return true;
  } catch (error) {
    actionError.value = message(error);
    return false;
  } finally {
    pending.value = "";
  }
}

const keyOf = (item: SkillItem) => `${item.scope}:${item.name}`;

function sharedDefault(item: SkillItem): boolean {
  return item.default_enabled ?? true;
}

function setDefault(item: SkillItem, enabled: boolean): void {
  void change(keyOf(item), () => skillsApi.setSwitch("shared", item.name, null, enabled));
}

function setInProject(item: SkillItem, enabled: boolean): void {
  // The same value as the default removes the own choice: the project follows the default again.
  const choice = item.scope === "shared" && enabled === sharedDefault(item) ? null : enabled;
  void change(keyOf(item), () => skillsApi.setSwitch(item.scope, item.name, projectId.value, choice));
}

function sharedNote(item: SkillItem): string {
  if (item.replaced) return "Вместо него работает навык проекта с тем же именем";
  if (item.project_choice === null) return sharedDefault(item) ? "Как во всех проектах" : "Выключен во всех проектах";
  return item.enabled ? "Включён только для этого проекта" : "Выключен только для этого проекта";
}

// ---------- create and edit ----------

const editor = reactive({
  open: false,
  mode: "create" as "create" | "edit",
  scope: "shared" as SkillScope,
  name: "",
  description: "",
  instructions: "",
  revision: "",
  files: [] as string[],
  loading: false,
  saving: false,
  touched: false,
  error: "",
});

const nameError = computed(() => {
  if (editor.mode === "edit") return "";
  if (!editor.name) return editor.touched ? "Введите имя навыка" : "";
  if (!SKILL_NAME.test(editor.name)) return "Только строчные латинские буквы, цифры и дефис, например release-notes";
  const taken = catalog.value?.skills.some((item) => item.scope === editor.scope && item.name === editor.name);
  return taken ? "Навык с таким именем уже есть" : "";
});
const descriptionError = computed(() => {
  if (!editor.description.trim()) return editor.touched ? "Опишите, когда агенту брать этот навык" : "";
  return editor.description.length > MAX_DESCRIPTION ? `Не больше ${MAX_DESCRIPTION} символов` : "";
});
const canSave = computed(
  () =>
    !editor.saving &&
    !editor.loading &&
    !!editor.name &&
    !!editor.description.trim() &&
    !nameError.value &&
    !descriptionError.value,
);

function openCreate(): void {
  Object.assign(editor, {
    open: true,
    mode: "create",
    scope: tab.value === "project" && project.value ? "project" : "shared",
    name: "",
    description: "",
    instructions: "",
    revision: "",
    files: [],
    loading: false,
    saving: false,
    touched: false,
    error: "",
  });
}

async function openEdit(item: SkillItem): Promise<void> {
  Object.assign(editor, {
    open: true,
    mode: "edit",
    scope: item.scope,
    name: item.name,
    description: item.description,
    instructions: "",
    revision: "",
    files: [],
    loading: true,
    saving: false,
    touched: false,
    error: "",
  });
  try {
    const detail = await skillsApi.get(item.scope, item.name, projectId.value);
    Object.assign(editor, {
      description: detail.description,
      instructions: detail.instructions,
      revision: detail.revision,
      files: detail.file_list,
    });
  } catch (error) {
    editor.error = `Не удалось открыть навык: ${message(error)}`;
  } finally {
    editor.loading = false;
  }
}

function closeEditor(): void {
  if (!editor.saving) editor.open = false;
}

async function save(): Promise<void> {
  editor.touched = true;
  if (!canSave.value) return;
  editor.saving = true;
  editor.error = "";
  const draft = { description: editor.description.trim(), instructions: editor.instructions };
  const owner = editor.scope === "project" ? projectId.value : null;
  try {
    catalog.value =
      editor.mode === "create"
        ? await skillsApi.create(editor.scope, owner, editor.name, draft)
        : await skillsApi.update(editor.scope, editor.name, owner, draft, editor.revision);
    status.value = editor.mode === "create" ? `Навык «${editor.name}» создан` : `Навык «${editor.name}» сохранён`;
    editor.open = false;
  } catch (error) {
    editor.error = message(error);
  } finally {
    editor.saving = false;
  }
}

// ---------- delete ----------

const confirm = reactive({ open: false, item: null as SkillItem | null, busy: false, error: "" });

function askDelete(item: SkillItem): void {
  Object.assign(confirm, { open: true, item, busy: false, error: "" });
}

async function remove(): Promise<void> {
  const item = confirm.item;
  if (!item) return;
  confirm.busy = true;
  confirm.error = "";
  const owner = item.scope === "project" ? projectId.value : null;
  try {
    const detail = await skillsApi.get(item.scope, item.name, owner);
    catalog.value = await skillsApi.remove(item.scope, item.name, owner, detail.revision);
    status.value = `Навык «${item.name}» удалён`;
    confirm.open = false;
  } catch (error) {
    confirm.error = message(error);
  } finally {
    confirm.busy = false;
  }
}

const deleteText = computed(() => {
  const item = confirm.item;
  if (!item) return "";
  const where = item.scope === "shared" ? "Навык пропадёт во всех проектах." : "Папка навыка будет удалена из проекта.";
  const files = item.files ? ` Вместе с SKILL.md удалятся ещё ${item.files} ${plural(item.files, "файл", "файла", "файлов")}.` : "";
  return `${where}${files} Отменить это нельзя.`;
});

// ---------- import ----------

const importScope = computed<SkillScope>(() => (tab.value === "project" && project.value ? "project" : "shared"));

function chooseFile(): void {
  fileInput.value?.click();
}

async function importFile(event: Event): Promise<void> {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  input.value = "";
  if (!file) return;
  if (file.size > MAX_IMPORT_BYTES) {
    actionError.value = `Файл «${file.name}» больше 5 МБ. Уберите из навыка лишние файлы.`;
    status.value = "";
    return;
  }
  const before = new Set(catalog.value?.skills.map(keyOf));
  importing.value = true;
  const scope = importScope.value;
  const done = await change("import", () => skillsApi.importFile(scope, scope === "project" ? projectId.value : null, file));
  importing.value = false;
  if (done) {
    const added = catalog.value?.skills.find((item) => !before.has(keyOf(item)));
    status.value = added ? `Навык «${added.name}» установлен` : "Навык установлен";
  }
}

function back(): void {
  if (window.history.length > 1) router.back();
  else void router.push({ name: "home" });
}

onMounted(load);
</script>

<template>
  <section class="settings" data-testid="skills-settings">
    <header class="settings__header">
      <button type="button" class="ghost" data-testid="settings-back" @click="back">
        <ArrowLeft :size="16" aria-hidden="true" />
        <span>Назад</span>
      </button>
      <h1>Навыки</h1>
      <nav class="settings__nav" aria-label="Разделы настроек">
        <RouterLink :to="{ name: 'settings-models' }">Модели</RouterLink>
        <RouterLink :to="{ name: 'settings-speech' }">Голос</RouterLink>
        <RouterLink :to="{ name: 'settings-skills' }" aria-current="page">Навыки</RouterLink>
      </nav>
    </header>

    <p class="intro">
      Навык — это инструкция для определённой задачи. Агент видит описание навыка и открывает его, когда задача
      подходит. Общие навыки работают во всех проектах, навыки проекта — только в своём.
    </p>

    <p v-if="loadError" class="notice notice--error" role="alert">
      <CircleAlert :size="16" aria-hidden="true" />
      <span>{{ loadError }}</span>
      <button type="button" class="link" @click="load">Повторить</button>
    </p>

    <template v-if="catalog">
      <div class="toolbar">
        <div class="segmented" role="tablist" aria-label="Какие навыки показать" @keydown="tabKey">
          <button
            id="skills-tab-shared"
            type="button"
            role="tab"
            :aria-selected="tab === 'shared'"
            :tabindex="tab === 'shared' ? 0 : -1"
            aria-controls="skills-panel"
            data-testid="skills-tab-shared"
            @click="tab = 'shared'"
          >
            Общие <span class="count">{{ shared.length }}</span>
          </button>
          <button
            id="skills-tab-project"
            type="button"
            role="tab"
            :aria-selected="tab === 'project'"
            :tabindex="tab === 'project' ? 0 : -1"
            aria-controls="skills-panel"
            data-testid="skills-tab-project"
            @click="tab = 'project'"
          >
            Проект <span class="count">{{ own.length }}</span>
          </button>
        </div>
        <div class="toolbar__actions">
          <button type="button" class="tonal" :disabled="importing" data-testid="skills-import" @click="chooseFile">
            <FileUp :size="16" aria-hidden="true" />
            <span>{{ importing ? "Устанавливаю…" : "Импорт" }}</span>
          </button>
          <button
            type="button"
            class="primary"
            :disabled="tab === 'project' && !project"
            data-testid="skills-create"
            @click="openCreate"
          >
            <Plus :size="16" aria-hidden="true" />
            <span>Новый навык</span>
          </button>
          <input
            ref="fileInput"
            type="file"
            class="visually-hidden"
            accept=".md,.zip,.skill"
            tabindex="-1"
            aria-hidden="true"
            data-testid="skills-file"
            @change="importFile"
          />
        </div>
      </div>

      <div aria-live="polite" role="status" class="live">
        <p v-if="status" class="notice notice--ok" data-testid="skills-status">
          <span>{{ status }}</span>
          <button type="button" class="icon" aria-label="Скрыть сообщение" @click="status = ''">
            <X :size="15" aria-hidden="true" />
          </button>
        </p>
      </div>
      <p v-if="actionError" class="notice notice--error" role="alert" data-testid="skills-error">
        <CircleAlert :size="16" aria-hidden="true" />
        <span>{{ actionError }}</span>
        <button type="button" class="icon" aria-label="Скрыть сообщение" @click="actionError = ''">
          <X :size="15" aria-hidden="true" />
        </button>
      </p>
      <details v-if="brokenCount" class="notice notice--warning" data-testid="skills-broken">
        <summary>
          {{ brokenCount }} {{ plural(brokenCount, "навык не загружен", "навыка не загружены", "навыков не загружены") }}:
          исправьте SKILL.md
        </summary>
        <ul>
          <li v-for="(item, index) in catalog.errors" :key="index">
            <span class="badge">{{ item.scope === "shared" ? "Общий" : "Проект" }}</span>
            <span class="mono small">{{ item.message }}</span>
          </li>
        </ul>
      </details>

      <div id="skills-panel" role="tabpanel" :aria-labelledby="`skills-tab-${tab}`">
        <template v-if="tab === 'shared'">
          <p class="hint">
            Выключатель задаёт, работает ли навык во всех проектах. В отдельном проекте его можно включить или
            выключить на вкладке «Проект».
            <span class="mono small path" :title="catalog.shared_dir">Папка: {{ catalog.shared_dir }}</span>
          </p>
          <div v-if="!shared.length" class="empty" data-testid="skills-empty-shared">
            <p>Общих навыков пока нет.</p>
            <p class="muted">Создайте навык или импортируйте файл SKILL.md или zip-архив с папкой навыка.</p>
          </div>
          <ul v-else class="list" aria-label="Общие навыки">
            <li v-for="item in shared" :key="keyOf(item)" class="row" :data-skill="item.name">
              <div class="row__main">
                <strong class="mono">{{ item.name }}</strong>
                <span class="row__description">{{ item.description }}</span>
                <span class="muted small">
                  {{ sharedDefault(item) ? "Во всех проектах" : "Выключен по умолчанию" }}
                  <template v-if="item.files"> · {{ filesText(item.files) }}</template>
                  <template v-if="item.version"> · версия {{ item.version }}</template>
                </span>
              </div>
              <div class="row__actions">
                <button type="button" class="icon" :aria-label="`Изменить навык ${item.name}`" :title="`Изменить`" data-testid="skill-edit" @click="openEdit(item)">
                  <Pencil :size="16" aria-hidden="true" />
                </button>
                <button type="button" class="icon icon--danger" :aria-label="`Удалить навык ${item.name}`" :title="`Удалить`" data-testid="skill-delete" @click="askDelete(item)">
                  <Trash2 :size="16" aria-hidden="true" />
                </button>
                <ToggleSwitch
                  :model-value="sharedDefault(item)"
                  :label="`Навык ${item.name} во всех проектах`"
                  :disabled="pending === keyOf(item)"
                  data-testid="skill-switch"
                  @update:model-value="setDefault(item, $event)"
                />
              </div>
            </li>
          </ul>
        </template>

        <template v-else>
          <div v-if="!projects.length" class="empty" data-testid="skills-no-projects">
            <p>Проектов пока нет.</p>
            <p class="muted">Навыки проекта появятся, когда вы добавите проект.</p>
          </div>
          <template v-else>
            <label class="project-picker">
              <span>Проект</span>
              <select v-model="projectId" data-testid="skills-project">
                <option v-for="item in projects" :key="item.id" :value="item.id">{{ item.name }}</option>
              </select>
            </label>

            <h2 class="group-title">Навыки проекта</h2>
            <p class="hint">
              Работают только в этом проекте. Навык проекта заменяет общий навык с тем же именем.
              <span v-if="catalog.project_dir" class="mono small path" :title="catalog.project_dir">
                Папка: {{ catalog.project_dir }}
              </span>
            </p>
            <div v-if="!own.length" class="empty" data-testid="skills-empty-project">
              <p>У проекта нет своих навыков.</p>
            </div>
            <ul v-else class="list" aria-label="Навыки проекта">
              <li v-for="item in own" :key="keyOf(item)" class="row" :class="{ 'row--off': !item.enabled }" :data-skill="item.name">
                <div class="row__main">
                  <strong class="mono">{{ item.name }}</strong>
                  <span class="row__description">{{ item.description }}</span>
                  <span class="muted small">
                    {{ item.enabled ? "Включён" : "Выключен" }}
                    <template v-if="item.files"> · {{ filesText(item.files) }}</template>
                    <template v-if="item.version"> · версия {{ item.version }}</template>
                  </span>
                </div>
                <div class="row__actions">
                  <button type="button" class="icon" :aria-label="`Изменить навык ${item.name}`" title="Изменить" data-testid="skill-edit" @click="openEdit(item)">
                    <Pencil :size="16" aria-hidden="true" />
                  </button>
                  <button type="button" class="icon icon--danger" :aria-label="`Удалить навык ${item.name}`" title="Удалить" data-testid="skill-delete" @click="askDelete(item)">
                    <Trash2 :size="16" aria-hidden="true" />
                  </button>
                  <ToggleSwitch
                    :model-value="item.enabled"
                    :label="`Навык ${item.name} в этом проекте`"
                    :disabled="pending === keyOf(item)"
                    data-testid="skill-switch"
                    @update:model-value="setInProject(item, $event)"
                  />
                </div>
              </li>
            </ul>

            <h2 class="group-title">Общие навыки в этом проекте</h2>
            <div v-if="!shared.length" class="empty">
              <p class="muted">Общих навыков пока нет.</p>
            </div>
            <ul v-else class="list" aria-label="Общие навыки в этом проекте">
              <li
                v-for="item in shared"
                :key="keyOf(item)"
                class="row"
                :class="{ 'row--off': !item.enabled || item.replaced }"
                :data-skill="item.name"
              >
                <div class="row__main">
                  <strong class="mono">{{ item.name }}</strong>
                  <span class="row__description">{{ item.description }}</span>
                  <span class="muted small" data-testid="skill-note">{{ sharedNote(item) }}</span>
                </div>
                <div class="row__actions">
                  <ToggleSwitch
                    :model-value="item.enabled"
                    :label="`Общий навык ${item.name} в этом проекте`"
                    :disabled="pending === keyOf(item)"
                    data-testid="skill-switch"
                    @update:model-value="setInProject(item, $event)"
                  />
                </div>
              </li>
            </ul>
          </template>
        </template>
      </div>
    </template>
    <p v-else-if="!loadError" class="muted">Загрузка…</p>

    <SettingsDialog
      :open="editor.open"
      :title="editor.mode === 'create' ? 'Новый навык' : `Навык ${editor.name}`"
      wide
      @close="closeEditor"
    >
      <form id="skill-form" class="form" novalidate data-testid="skill-form" @submit.prevent="save">
        <fieldset v-if="editor.mode === 'create'" class="where">
          <legend>Где хранить</legend>
          <label class="choice">
            <input v-model="editor.scope" type="radio" value="shared" />
            <span>Общий — для всех проектов</span>
          </label>
          <label class="choice" :class="{ 'choice--off': !project }">
            <input v-model="editor.scope" type="radio" value="project" :disabled="!project" />
            <span>Только проект{{ project ? ` «${project.name}»` : "" }}</span>
          </label>
        </fieldset>
        <label class="field">
          <span>Имя</span>
          <input
            v-model.trim="editor.name"
            class="mono"
            :disabled="editor.mode === 'edit'"
            :aria-invalid="!!nameError"
            aria-describedby="skill-name-help"
            autocomplete="off"
            spellcheck="false"
            :data-autofocus="editor.mode === 'create' ? '' : undefined"
            data-testid="skill-name"
            @blur="editor.touched = true"
          />
          <small id="skill-name-help" :class="{ 'field__error': nameError }">
            {{ nameError || (editor.mode === "edit" ? "Имя навыка не меняется." : "Строчные латинские буквы, цифры и дефис.") }}
          </small>
        </label>
        <label class="field">
          <span>Когда применять</span>
          <textarea
            v-model="editor.description"
            rows="3"
            :aria-invalid="!!descriptionError"
            aria-describedby="skill-description-help"
            :data-autofocus="editor.mode === 'edit' ? '' : undefined"
            data-testid="skill-description"
          />
          <small id="skill-description-help" class="field__help" :class="{ 'field__error': descriptionError }">
            <span>{{ descriptionError || "Агент видит только это описание и по нему решает, открыть ли навык." }}</span>
            <span class="counter">{{ editor.description.length }} / {{ MAX_DESCRIPTION }}</span>
          </small>
        </label>
        <label class="field">
          <span>Инструкции</span>
          <textarea
            v-model="editor.instructions"
            class="mono instructions"
            rows="14"
            :disabled="editor.loading"
            :placeholder="editor.loading ? 'Загрузка…' : 'Шаги, правила и примеры в Markdown'"
            data-testid="skill-instructions"
          />
          <small>Агент читает инструкции целиком, когда открывает навык.</small>
        </label>
        <p v-if="editor.files.length" class="muted small" data-testid="skill-files">
          Другие файлы навыка: <span class="mono">{{ editor.files.join(", ") }}</span>
        </p>
        <p v-if="editor.error" class="notice notice--error" role="alert" data-testid="skill-form-error">
          <CircleAlert :size="16" aria-hidden="true" />
          <span>{{ editor.error }}</span>
        </p>
      </form>
      <template #actions>
        <button type="button" class="text" :disabled="editor.saving" @click="closeEditor">Отмена</button>
        <button type="submit" form="skill-form" class="primary" :disabled="!canSave" data-testid="skill-save">
          {{ editor.saving ? "Сохраняю…" : editor.mode === "create" ? "Создать" : "Сохранить" }}
        </button>
      </template>
    </SettingsDialog>

    <SettingsDialog
      :open="confirm.open"
      :title="confirm.item ? `Удалить навык «${confirm.item.name}»?` : ''"
      alert
      @close="confirm.busy || (confirm.open = false)"
    >
      <p class="confirm-text">{{ deleteText }}</p>
      <p v-if="confirm.error" class="notice notice--error" role="alert" data-testid="skill-delete-error">
        <CircleAlert :size="16" aria-hidden="true" />
        <span>{{ confirm.error }}</span>
      </p>
      <template #actions>
        <button type="button" class="text" data-autofocus :disabled="confirm.busy" @click="confirm.open = false">
          Отмена
        </button>
        <button type="button" class="danger-fill" :disabled="confirm.busy" data-testid="skill-delete-confirm" @click="remove">
          {{ confirm.busy ? "Удаляю…" : "Удалить" }}
        </button>
      </template>
    </SettingsDialog>
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

.intro {
  margin: 0 0 16px;
  color: var(--text-muted);
  font-size: 14px;
  line-height: 1.5;
}

.toolbar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.toolbar__actions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

/* Segmented control: one rounded track, the selected segment raised on it. */
.segmented {
  display: inline-flex;
  padding: 3px;
  border-radius: 10px;
  background: var(--bg-subtle);
}

.segmented button {
  min-height: 36px;
  padding: 0 16px;
  border: 0;
  border-radius: 8px;
  background: transparent;
  color: var(--text-muted);
  font-size: 14px;
  cursor: pointer;
}

.segmented button[aria-selected="true"] {
  background: var(--surface);
  color: var(--text);
  font-weight: 600;
  box-shadow: 0 1px 2px rgba(43, 33, 24, 0.12);
}

.segmented button:focus-visible {
  outline: 2px solid var(--accent);
  outline-offset: 1px;
}

.count {
  margin-left: 4px;
  color: var(--text-muted);
  font-weight: 400;
  font-variant-numeric: tabular-nums;
}

.live:empty {
  display: none;
}

.hint {
  margin: 16px 0 8px;
  color: var(--text-muted);
  font-size: 13px;
  line-height: 1.5;
}

.path {
  display: block;
  margin-top: 2px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.group-title {
  margin: 24px 0 0;
  font-size: 15px;
}

.project-picker {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-top: 16px;
  font-size: 13px;
  color: var(--text-muted);
}

.project-picker select {
  min-height: 36px;
  min-width: 200px;
}

/* An inset grouped list: rows share one rounded container, separated by hairlines. */
.list {
  margin: 0;
  padding: 0;
  list-style: none;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--surface);
  overflow: hidden;
}

.row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 8px 16px;
  padding: 12px 8px 12px 16px;
}

.row + .row {
  border-top: 1px solid var(--border);
}

.row--off .row__main {
  opacity: 0.6;
}

.row__main {
  display: flex;
  flex: 1 1 320px;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
  overflow-wrap: anywhere;
}

.row__description {
  display: -webkit-box;
  overflow: hidden;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 2;
  line-clamp: 2;
  font-size: 14px;
}

.row__actions {
  display: flex;
  align-items: center;
  gap: 2px;
  margin-left: auto;
}

.empty {
  padding: 20px 16px;
  border: 1px dashed var(--border);
  border-radius: var(--radius);
  text-align: center;
}

.empty p {
  margin: 0;
}

.empty p + p {
  margin-top: 4px;
  font-size: 13px;
}

.small {
  font-size: 12px;
}

.mono {
  font-family: var(--font-mono);
}

.badge {
  display: inline-block;
  margin-right: 6px;
  padding: 0 6px;
  border-radius: 999px;
  background: var(--bg-subtle);
  color: var(--text-muted);
  font-size: 11px;
}

button.ghost,
button.primary,
button.tonal,
button.text,
button.danger-fill {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  min-height: 36px;
  padding: 0 16px;
  border: 1px solid transparent;
  border-radius: 18px;
  background: transparent;
  color: var(--text);
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
}

button.ghost {
  padding: 0 10px;
  border-color: var(--border);
  border-radius: var(--radius-sm);
  font-size: 13px;
  font-weight: 400;
}

button.primary {
  background: var(--accent);
  color: var(--accent-contrast);
}

button.tonal {
  background: color-mix(in srgb, var(--accent) 14%, var(--surface));
  color: var(--text);
}

button.text {
  color: var(--accent);
}

button.text:not(:disabled):hover,
button.tonal:not(:disabled):hover {
  background: color-mix(in srgb, var(--accent) 10%, transparent);
}

button.danger-fill {
  background: var(--danger);
  color: #fff;
}

button.icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 40px;
  height: 40px;
  padding: 0;
  border: 0;
  border-radius: 50%;
  background: transparent;
  color: var(--text-muted);
  cursor: pointer;
}

button.icon:not(:disabled):hover {
  background: var(--bg-subtle);
  color: var(--text);
}

button.icon--danger:not(:disabled):hover {
  background: color-mix(in srgb, var(--danger) 10%, transparent);
  color: var(--danger);
}

button:disabled {
  opacity: 0.5;
  cursor: default;
}

button:focus-visible {
  outline: 2px solid var(--accent);
  outline-offset: 2px;
}

button.link {
  border: 0;
  background: none;
  color: var(--link);
  text-decoration: underline;
  cursor: pointer;
}

.notice {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 12px 0 0;
  padding: 8px 8px 8px 12px;
  border-radius: var(--radius-sm);
  font-size: 13px;
}

.notice > span {
  flex: 1;
}

.notice button.icon {
  width: 32px;
  height: 32px;
  color: inherit;
}

.notice--error {
  border: 1px solid color-mix(in srgb, var(--danger) 45%, transparent);
  background: color-mix(in srgb, var(--danger) 8%, transparent);
  color: var(--danger);
}

.notice--ok {
  border: 1px solid color-mix(in srgb, var(--success) 40%, transparent);
  background: color-mix(in srgb, var(--success) 8%, transparent);
  color: var(--success);
  animation: notice-in 0.18s ease-out;
}

.notice--warning {
  display: block;
  border: 1px solid color-mix(in srgb, var(--warning) 45%, transparent);
  background: color-mix(in srgb, var(--warning) 8%, transparent);
  color: var(--warning);
}

.notice--warning summary {
  cursor: pointer;
}

.notice--warning ul {
  margin: 8px 0 0;
  padding-left: 18px;
  color: var(--text);
}

.form {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.where {
  display: flex;
  flex-wrap: wrap;
  gap: 8px 20px;
  margin: 0;
  padding: 0;
  border: 0;
}

.where legend {
  margin-bottom: 6px;
  color: var(--text-muted);
  font-size: 13px;
}

.choice {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  min-height: 32px;
  font-size: 14px;
  cursor: pointer;
}

.choice input {
  accent-color: var(--accent);
}

.choice--off {
  opacity: 0.5;
  cursor: default;
}

.field {
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 13px;
  color: var(--text-muted);
}

.field > span {
  color: var(--text);
  font-weight: 500;
}

.field small {
  font-size: 12px;
}

.field__help {
  display: flex;
  justify-content: space-between;
  gap: 12px;
}

.counter {
  flex: none;
  font-variant-numeric: tabular-nums;
}

.field__error {
  color: var(--danger);
}

input:not([type="radio"]),
select,
textarea {
  padding: 8px 10px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--bg);
  color: var(--text);
  font: inherit;
  font-size: 14px;
}

input[aria-invalid="true"],
textarea[aria-invalid="true"] {
  border-color: var(--danger);
}

input:focus-visible,
select:focus-visible,
textarea:focus-visible {
  outline: 2px solid var(--accent);
  outline-offset: -1px;
}

input:disabled {
  color: var(--text-muted);
}

textarea {
  resize: vertical;
  line-height: 1.45;
}

textarea.instructions {
  min-height: 220px;
  font-family: var(--font-mono);
  font-size: 13px;
}

.confirm-text {
  margin: 0;
  line-height: 1.5;
}

.visually-hidden {
  position: absolute;
  width: 1px;
  height: 1px;
  overflow: hidden;
  clip-path: inset(50%);
  white-space: nowrap;
}

@keyframes notice-in {
  from {
    opacity: 0;
    transform: translateY(-4px);
  }
}

@media (prefers-reduced-motion: reduce) {
  .notice--ok {
    animation: none;
  }
}

@media (max-width: 599px) {
  .settings {
    padding: 12px 16px 40px;
  }

  .toolbar__actions {
    width: 100%;
  }

  .toolbar__actions button {
    flex: 1;
  }

  .project-picker {
    flex-direction: column;
    align-items: stretch;
  }
}
</style>
