<script setup lang="ts">
// Skills: shared skills serve every project, project skills serve only their project.
// A switch acts at once. The shared tab sets the default for all projects; the project tab
// sets the choice of one project and shows the shared skills next to its own.

import { computed, onMounted, reactive, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import { CircleAlert, FileUp, Pencil, Plus, Trash2, X } from "lucide-vue-next";
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
import SettingsHeader from "../components/settings/SettingsHeader.vue";
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

onMounted(load);
</script>

<template>
  <section class="ui-page" data-testid="skills-settings">
    <SettingsHeader title="Навыки" />

    <p class="ui-hint">
      Навык — это инструкция для определённой задачи. Агент видит описание навыка и открывает его, когда задача
      подходит. Общие навыки работают во всех проектах, навыки проекта — только в своём.
    </p>

    <p v-if="loadError" class="ui-notice ui-notice--error" role="alert">
      <CircleAlert :size="16" aria-hidden="true" />
      <span class="ui-notice__text">{{ loadError }}</span>
      <button type="button" class="ui-link" @click="load">Повторить</button>
    </p>

    <template v-if="catalog">
      <div class="toolbar">
        <div class="ui-tabs" role="tablist" aria-label="Какие навыки показать" @keydown="tabKey">
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
        <div class="ui-section__actions">
          <button
            type="button"
            class="ui-btn ui-btn--tonal"
            :disabled="importing"
            data-testid="skills-import"
            @click="chooseFile"
          >
            <FileUp :size="16" aria-hidden="true" />
            <span>{{ importing ? "Устанавливаю…" : "Импорт" }}</span>
          </button>
          <button
            type="button"
            class="ui-btn ui-btn--filled"
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
            class="sr-only"
            accept=".md,.zip,.skill"
            tabindex="-1"
            aria-hidden="true"
            data-testid="skills-file"
            @change="importFile"
          />
        </div>
      </div>

      <div aria-live="polite" role="status" class="live">
        <p v-if="status" class="ui-notice ui-notice--success" data-testid="skills-status">
          <span class="ui-notice__text">{{ status }}</span>
          <button type="button" class="ui-btn ui-btn--icon" aria-label="Скрыть сообщение" @click="status = ''">
            <X :size="15" aria-hidden="true" />
          </button>
        </p>
      </div>
      <p v-if="actionError" class="ui-notice ui-notice--error" role="alert" data-testid="skills-error">
        <CircleAlert :size="16" aria-hidden="true" />
        <span class="ui-notice__text">{{ actionError }}</span>
        <button type="button" class="ui-btn ui-btn--icon" aria-label="Скрыть сообщение" @click="actionError = ''">
          <X :size="15" aria-hidden="true" />
        </button>
      </p>
      <details v-if="brokenCount" class="ui-notice ui-notice--warning broken" data-testid="skills-broken">
        <summary>
          {{ brokenCount }} {{ plural(brokenCount, "навык не загружен", "навыка не загружены", "навыков не загружены") }}:
          исправьте SKILL.md
        </summary>
        <ul>
          <li v-for="(item, index) in catalog.errors" :key="index">
            <span class="ui-badge">{{ item.scope === "shared" ? "Общий" : "Проект" }}</span>
            <span class="mono ui-small">{{ item.message }}</span>
          </li>
        </ul>
      </details>

      <div id="skills-panel" role="tabpanel" :aria-labelledby="`skills-tab-${tab}`">
        <section v-if="tab === 'shared'" class="ui-section" aria-labelledby="skills-shared-title">
          <h2 id="skills-shared-title" class="ui-section__title">Общие навыки</h2>
          <p class="ui-hint">
            Выключатель задаёт, работает ли навык во всех проектах. В отдельном проекте его можно включить или
            выключить на вкладке «Проект».
            <span class="mono path" :title="catalog.shared_dir">Папка: {{ catalog.shared_dir }}</span>
          </p>
          <div v-if="!shared.length" class="ui-empty" data-testid="skills-empty-shared">
            <p>Общих навыков пока нет.</p>
            <p class="empty__more">Создайте навык или импортируйте файл SKILL.md или zip-архив с папкой навыка.</p>
          </div>
          <ul v-else class="list" aria-label="Общие навыки">
            <li v-for="item in shared" :key="keyOf(item)" class="ui-card ui-card__row" :data-skill="item.name">
              <div class="ui-card__main">
                <strong class="ui-card__title mono">{{ item.name }}</strong>
                <span class="description">{{ item.description }}</span>
                <span class="ui-card__meta">
                  {{ sharedDefault(item) ? "Во всех проектах" : "Выключен по умолчанию" }}
                  <template v-if="item.files"> · {{ filesText(item.files) }}</template>
                  <template v-if="item.version"> · версия {{ item.version }}</template>
                </span>
              </div>
              <div class="ui-card__actions">
                <button
                  type="button"
                  class="ui-btn ui-btn--icon"
                  :aria-label="`Изменить навык ${item.name}`"
                  title="Изменить"
                  data-testid="skill-edit"
                  @click="openEdit(item)"
                >
                  <Pencil :size="16" aria-hidden="true" />
                </button>
                <button
                  type="button"
                  class="ui-btn ui-btn--icon ui-btn--danger"
                  :aria-label="`Удалить навык ${item.name}`"
                  title="Удалить"
                  data-testid="skill-delete"
                  @click="askDelete(item)"
                >
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
        </section>

        <template v-else>
          <div v-if="!projects.length" class="ui-empty section-gap" data-testid="skills-no-projects">
            <p>Проектов пока нет.</p>
            <p class="empty__more">Навыки проекта появятся, когда вы добавите проект.</p>
          </div>
          <template v-else>
            <label class="ui-field project-picker">
              <span class="ui-field__label">Проект</span>
              <select v-model="projectId" class="ui-input" data-testid="skills-project">
                <option v-for="item in projects" :key="item.id" :value="item.id">{{ item.name }}</option>
              </select>
            </label>

            <section class="ui-section" aria-labelledby="skills-own-title">
              <h2 id="skills-own-title" class="ui-section__title">Навыки проекта</h2>
              <p class="ui-hint">
                Работают только в этом проекте. Навык проекта заменяет общий навык с тем же именем.
                <span v-if="catalog.project_dir" class="mono path" :title="catalog.project_dir">
                  Папка: {{ catalog.project_dir }}
                </span>
              </p>
              <div v-if="!own.length" class="ui-empty" data-testid="skills-empty-project">
                <p>У проекта нет своих навыков.</p>
              </div>
              <ul v-else class="list" aria-label="Навыки проекта">
                <li
                  v-for="item in own"
                  :key="keyOf(item)"
                  class="ui-card ui-card__row"
                  :class="{ 'ui-card--off': !item.enabled }"
                  :data-skill="item.name"
                >
                  <div class="ui-card__main">
                    <strong class="ui-card__title mono">{{ item.name }}</strong>
                    <span class="description">{{ item.description }}</span>
                    <span class="ui-card__meta">
                      {{ item.enabled ? "Включён" : "Выключен" }}
                      <template v-if="item.files"> · {{ filesText(item.files) }}</template>
                      <template v-if="item.version"> · версия {{ item.version }}</template>
                    </span>
                  </div>
                  <div class="ui-card__actions">
                    <button
                      type="button"
                      class="ui-btn ui-btn--icon"
                      :aria-label="`Изменить навык ${item.name}`"
                      title="Изменить"
                      data-testid="skill-edit"
                      @click="openEdit(item)"
                    >
                      <Pencil :size="16" aria-hidden="true" />
                    </button>
                    <button
                      type="button"
                      class="ui-btn ui-btn--icon ui-btn--danger"
                      :aria-label="`Удалить навык ${item.name}`"
                      title="Удалить"
                      data-testid="skill-delete"
                      @click="askDelete(item)"
                    >
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
            </section>

            <section class="ui-section" aria-labelledby="skills-inherited-title">
              <h2 id="skills-inherited-title" class="ui-section__title">Общие навыки в этом проекте</h2>
              <p class="ui-hint">Выключатель меняет выбор только для этого проекта.</p>
              <div v-if="!shared.length" class="ui-empty">
                <p>Общих навыков пока нет.</p>
              </div>
              <ul v-else class="list" aria-label="Общие навыки в этом проекте">
                <li
                  v-for="item in shared"
                  :key="keyOf(item)"
                  class="ui-card ui-card__row"
                  :class="{ 'ui-card--off': !item.enabled || item.replaced }"
                  :data-skill="item.name"
                >
                  <div class="ui-card__main">
                    <strong class="ui-card__title mono">{{ item.name }}</strong>
                    <span class="description">{{ item.description }}</span>
                    <span class="ui-card__meta" data-testid="skill-note">{{ sharedNote(item) }}</span>
                  </div>
                  <div class="ui-card__actions">
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
            </section>
          </template>
        </template>
      </div>
    </template>
    <p v-else-if="!loadError" class="ui-hint">Загрузка…</p>

    <SettingsDialog
      :open="editor.open"
      :title="editor.mode === 'create' ? 'Новый навык' : `Навык ${editor.name}`"
      wide
      @close="closeEditor"
    >
      <form id="skill-form" class="ui-form" novalidate data-testid="skill-form" @submit.prevent="save">
        <fieldset v-if="editor.mode === 'create'" class="where">
          <legend class="ui-field__label">Где хранить</legend>
          <label class="ui-choice">
            <input v-model="editor.scope" type="radio" value="shared" />
            <span>Общий — для всех проектов</span>
          </label>
          <label class="ui-choice" :class="{ 'choice--off': !project }">
            <input v-model="editor.scope" type="radio" value="project" :disabled="!project" />
            <span>Только проект{{ project ? ` «${project.name}»` : "" }}</span>
          </label>
        </fieldset>
        <label class="ui-field">
          <span class="ui-field__label">Имя</span>
          <input
            v-model.trim="editor.name"
            class="ui-input mono"
            :disabled="editor.mode === 'edit'"
            :aria-invalid="!!nameError"
            aria-describedby="skill-name-help"
            autocomplete="off"
            spellcheck="false"
            :data-autofocus="editor.mode === 'create' ? '' : undefined"
            data-testid="skill-name"
            @blur="editor.touched = true"
          />
          <small id="skill-name-help" class="ui-field__help" :class="{ 'ui-field__help--error': nameError }">
            {{ nameError || (editor.mode === "edit" ? "Имя навыка не меняется." : "Строчные латинские буквы, цифры и дефис.") }}
          </small>
        </label>
        <label class="ui-field">
          <span class="ui-field__label">Когда применять</span>
          <textarea
            v-model="editor.description"
            class="ui-input"
            rows="3"
            :aria-invalid="!!descriptionError"
            aria-describedby="skill-description-help"
            :data-autofocus="editor.mode === 'edit' ? '' : undefined"
            data-testid="skill-description"
          />
          <small
            id="skill-description-help"
            class="ui-field__help"
            :class="{ 'ui-field__help--error': descriptionError }"
          >
            <span>{{ descriptionError || "Агент видит только это описание и по нему решает, открыть ли навык." }}</span>
            <span class="counter">{{ editor.description.length }} / {{ MAX_DESCRIPTION }}</span>
          </small>
        </label>
        <label class="ui-field">
          <span class="ui-field__label">Инструкции</span>
          <textarea
            v-model="editor.instructions"
            class="ui-input mono instructions"
            rows="14"
            :disabled="editor.loading"
            :placeholder="editor.loading ? 'Загрузка…' : 'Шаги, правила и примеры в Markdown'"
            data-testid="skill-instructions"
          />
          <small class="ui-field__help">Агент читает инструкции целиком, когда открывает навык.</small>
        </label>
        <p v-if="editor.files.length" class="ui-field__help" data-testid="skill-files">
          <span>Другие файлы навыка: <span class="mono">{{ editor.files.join(", ") }}</span></span>
        </p>
        <p v-if="editor.error" class="ui-notice ui-notice--error" role="alert" data-testid="skill-form-error">
          <CircleAlert :size="16" aria-hidden="true" />
          <span class="ui-notice__text">{{ editor.error }}</span>
        </p>
      </form>
      <template #actions>
        <button type="button" class="ui-btn ui-btn--text" :disabled="editor.saving" @click="closeEditor">Отмена</button>
        <button type="submit" form="skill-form" class="ui-btn ui-btn--filled" :disabled="!canSave" data-testid="skill-save">
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
      <p v-if="confirm.error" class="ui-notice ui-notice--error" role="alert" data-testid="skill-delete-error">
        <CircleAlert :size="16" aria-hidden="true" />
        <span class="ui-notice__text">{{ confirm.error }}</span>
      </p>
      <template #actions>
        <button
          type="button"
          class="ui-btn ui-btn--text"
          data-autofocus
          :disabled="confirm.busy"
          @click="confirm.open = false"
        >
          Отмена
        </button>
        <button
          type="button"
          class="ui-btn ui-btn--filled ui-btn--danger"
          :disabled="confirm.busy"
          data-testid="skill-delete-confirm"
          @click="remove"
        >
          {{ confirm.busy ? "Удаляю…" : "Удалить" }}
        </button>
      </template>
    </SettingsDialog>
  </section>
</template>

<style scoped>
.toolbar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-top: 8px;
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

.path {
  display: block;
  margin-top: 2px;
  overflow: hidden;
  font-size: 12px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

/* The gap between cards comes from the list, not from the card rule. */
.list > .ui-card + .ui-card {
  margin-top: 0;
}

.description {
  display: -webkit-box;
  overflow: hidden;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 2;
  line-clamp: 2;
  font-size: 14px;
}

.empty__more {
  margin-top: 4px;
  font-size: 13px;
}

.section-gap {
  margin-top: 28px;
}

.project-picker {
  max-width: 320px;
  margin-top: 20px;
}

.broken {
  display: block;
}

.broken summary {
  cursor: pointer;
}

.broken ul {
  display: flex;
  flex-direction: column;
  gap: 4px;
  margin-top: 8px;
  color: var(--text);
}

.broken .ui-badge {
  margin-right: 6px;
}

.where {
  display: flex;
  flex-wrap: wrap;
  gap: 4px 20px;
  margin: 0;
  padding: 0;
  border: 0;
}

.where legend {
  margin-bottom: 4px;
}

.choice--off {
  opacity: 0.5;
  cursor: default;
}

.counter {
  flex: none;
  font-variant-numeric: tabular-nums;
}

textarea.instructions {
  min-height: 220px;
  font-size: 13px;
}

.confirm-text {
  line-height: 1.5;
}

@media (max-width: 599px) {
  .toolbar .ui-section__actions {
    width: 100%;
  }

  .toolbar .ui-section__actions .ui-btn {
    flex: 1;
  }

  .project-picker {
    max-width: none;
  }
}
</style>
