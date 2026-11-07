import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { createMemoryHistory, createRouter } from "vue-router";
import SettingsSkillsView from "./SettingsSkillsView.vue";
import { api } from "../api/client";
import { skillsApi, type SkillCatalog, type SkillItem } from "../api/skills";

vi.mock("../api/client", () => ({ api: { projects: vi.fn() } }));
vi.mock("../api/skills", async (original) => ({
  ...(await original<typeof import("../api/skills")>()),
  skillsApi: {
    list: vi.fn(),
    get: vi.fn(),
    create: vi.fn(),
    update: vi.fn(),
    remove: vi.fn(),
    setSwitch: vi.fn(),
    importFile: vi.fn(),
  },
}));

const item = (name: string, extra: Partial<SkillItem> = {}): SkillItem => ({
  scope: "shared",
  name,
  description: `Описание ${name}`,
  version: "",
  enabled: true,
  default_enabled: true,
  project_choice: null,
  replaced: false,
  files: 0,
  path: `/skills/${name}`,
  ...extra,
});

const catalog = (skills: SkillItem[], errors: SkillCatalog["errors"] = []): SkillCatalog => ({
  shared_dir: "/data/skills",
  project_dir: "/work/app/.agent/skills",
  skills,
  errors,
});

const PROJECTS = [
  { id: "p1", name: "app", root_path: "/work/app" },
  { id: "p2", name: "site", root_path: "/work/site" },
];

async function mountView(skills: SkillItem[], query = "", errors: SkillCatalog["errors"] = []) {
  vi.mocked(api.projects).mockResolvedValue(PROJECTS);
  vi.mocked(skillsApi.list).mockResolvedValue(catalog(skills, errors));
  const router = createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: "/", name: "home", component: { template: "<div/>" } },
      { path: "/settings/models", name: "settings-models", component: { template: "<div/>" } },
      { path: "/settings/speech", name: "settings-speech", component: { template: "<div/>" } },
      { path: "/settings/skills", name: "settings-skills", component: SettingsSkillsView },
      { path: "/settings/mcp", name: "settings-mcp", component: { template: "<div/>" } },
    ],
  });
  await router.push(`/settings/skills${query}`);
  const wrapper = mount(SettingsSkillsView, { global: { plugins: [router] }, attachTo: document.body });
  await flushPromises();
  return wrapper;
}

beforeEach(() => setActivePinia(createPinia()));
afterEach(() => {
  vi.clearAllMocks();
  document.body.innerHTML = "";
});

it("shared tab switches the default of a shared skill for all projects", async () => {
  const wrapper = await mountView([item("review"), item("deploy", { scope: "project" })]);
  expect(wrapper.find('[data-testid="skills-tab-shared"]').text()).toContain("1");
  const row = wrapper.find('[data-skill="review"]');
  expect(row.text()).toContain("Во всех проектах");
  vi.mocked(skillsApi.setSwitch).mockResolvedValue(catalog([item("review", { default_enabled: false, enabled: false })]));

  await row.find('[role="switch"]').trigger("click");
  await flushPromises();

  expect(skillsApi.setSwitch).toHaveBeenCalledWith("shared", "review", null, false);
  expect(wrapper.find('[data-skill="review"]').text()).toContain("Выключен по умолчанию");
});

it("project tab keeps an own choice only when it differs from the default", async () => {
  const shared = item("review");
  const wrapper = await mountView([shared], "?tab=project");
  vi.mocked(skillsApi.setSwitch).mockResolvedValue(catalog([{ ...shared, enabled: false, project_choice: false }]));

  await wrapper.find('[data-skill="review"] [role="switch"]').trigger("click");
  await flushPromises();
  expect(skillsApi.setSwitch).toHaveBeenLastCalledWith("shared", "review", "p1", false);
  expect(wrapper.find('[data-testid="skill-note"]').text()).toBe("Выключен только для этого проекта");

  // Back to the default value: the own choice is removed instead of stored.
  vi.mocked(skillsApi.setSwitch).mockResolvedValue(catalog([shared]));
  await wrapper.find('[data-skill="review"] [role="switch"]').trigger("click");
  await flushPromises();
  expect(skillsApi.setSwitch).toHaveBeenLastCalledWith("shared", "review", "p1", null);
  expect(wrapper.find('[data-testid="skill-note"]').text()).toBe("Как во всех проектах");
});

it("project tab marks a shared skill that a project skill replaces", async () => {
  const wrapper = await mountView(
    [item("review", { replaced: true }), item("review", { scope: "project" })],
    "?tab=project",
  );
  const notes = wrapper.findAll('[data-testid="skill-note"]').map((note) => note.text());
  expect(notes).toEqual(["Вместо него работает навык проекта с тем же именем"]);
});

it("changing the project loads its skills", async () => {
  const wrapper = await mountView([], "?tab=project");
  expect(skillsApi.list).toHaveBeenLastCalledWith("p1");
  await wrapper.find('[data-testid="skills-project"]').setValue("p2");
  await flushPromises();
  expect(skillsApi.list).toHaveBeenLastCalledWith("p2");
});

it("create checks the name and stores the skill in the project of the tab", async () => {
  const wrapper = await mountView([item("review", { scope: "project" })], "?tab=project");
  await wrapper.find('[data-testid="skills-create"]').trigger("click");
  const save = () => wrapper.find('[data-testid="skill-save"]');

  await wrapper.find('[data-testid="skill-name"]').setValue("Release Notes");
  expect(wrapper.text()).toContain("Только строчные латинские буквы");
  expect(save().attributes("disabled")).toBeDefined();
  await wrapper.find('[data-testid="skill-name"]').setValue("review");
  expect(wrapper.text()).toContain("Навык с таким именем уже есть");

  await wrapper.find('[data-testid="skill-name"]').setValue("release-notes");
  await wrapper.find('[data-testid="skill-description"]').setValue("Когда нужен список изменений");
  await wrapper.find('[data-testid="skill-instructions"]').setValue("Сгруппируй изменения.");
  vi.mocked(skillsApi.create).mockResolvedValue(catalog([item("release-notes", { scope: "project" })]));
  await wrapper.find('[data-testid="skill-form"]').trigger("submit");
  await flushPromises();

  expect(skillsApi.create).toHaveBeenCalledWith("project", "p1", "release-notes", {
    description: "Когда нужен список изменений",
    instructions: "Сгруппируй изменения.",
  });
  expect(wrapper.find('[data-testid="skills-status"]').text()).toContain("Навык «release-notes» создан");
  expect(wrapper.find('[data-testid="skill-form"]').exists()).toBe(false);
});

it("edit sends the revision it read and shows a conflict in the form", async () => {
  const wrapper = await mountView([item("review")]);
  vi.mocked(skillsApi.get).mockResolvedValue({
    ...item("review"),
    instructions: "Шаги",
    file_list: ["checklist.md"],
    revision: "a".repeat(64),
  });
  await wrapper.find('[data-skill="review"] [data-testid="skill-edit"]').trigger("click");
  await flushPromises();
  expect((wrapper.find('[data-testid="skill-instructions"]').element as HTMLTextAreaElement).value).toBe("Шаги");
  expect(wrapper.find('[data-testid="skill-files"]').text()).toContain("checklist.md");

  vi.mocked(skillsApi.update).mockRejectedValue(new Error("Навык изменился в другом месте. Обновите страницу."));
  await wrapper.find('[data-testid="skill-form"]').trigger("submit");
  await flushPromises();

  expect(skillsApi.update).toHaveBeenCalledWith(
    "shared",
    "review",
    null,
    { description: "Описание review", instructions: "Шаги" },
    "a".repeat(64),
  );
  expect(wrapper.find('[data-testid="skill-form-error"]').text()).toContain("изменился в другом месте");
});

it("delete asks first and names the files that go with the skill", async () => {
  const wrapper = await mountView([item("review", { files: 2 })]);
  await wrapper.find('[data-skill="review"] [data-testid="skill-delete"]').trigger("click");
  expect(wrapper.text()).toContain("Удалить навык «review»?");
  expect(wrapper.text()).toContain("удалятся ещё 2 файла");
  expect(skillsApi.remove).not.toHaveBeenCalled();

  vi.mocked(skillsApi.get).mockResolvedValue({ ...item("review"), instructions: "", file_list: [], revision: "b".repeat(64) });
  vi.mocked(skillsApi.remove).mockResolvedValue(catalog([]));
  await wrapper.find('[data-testid="skill-delete-confirm"]').trigger("click");
  await flushPromises();

  expect(skillsApi.remove).toHaveBeenCalledWith("shared", "review", null, "b".repeat(64));
  expect(wrapper.find('[data-testid="skills-empty-shared"]').exists()).toBe(true);
});

it("import refuses a file over 5 MB before sending it", async () => {
  const wrapper = await mountView([]);
  const input = wrapper.find('[data-testid="skills-file"]');
  const big = new File(["x"], "big.zip");
  Object.defineProperty(big, "size", { value: 6 * 1024 * 1024 });
  Object.defineProperty(input.element, "files", { value: [big], configurable: true });
  await input.trigger("change");
  await flushPromises();

  expect(skillsApi.importFile).not.toHaveBeenCalled();
  expect(wrapper.find('[data-testid="skills-error"]').text()).toContain("больше 5 МБ");
});

it("import names the installed skill", async () => {
  const wrapper = await mountView([item("review")]);
  const input = wrapper.find('[data-testid="skills-file"]');
  const file = new File(["zip"], "pdf.zip");
  Object.defineProperty(input.element, "files", { value: [file], configurable: true });
  vi.mocked(skillsApi.importFile).mockResolvedValue(catalog([item("pdf-tools"), item("review")]));
  await input.trigger("change");
  await flushPromises();

  expect(skillsApi.importFile).toHaveBeenCalledWith("shared", null, file);
  expect(wrapper.find('[data-testid="skills-status"]').text()).toBe("Навык «pdf-tools» установлен");
});

it("broken skills are listed apart and do not hide the others", async () => {
  const wrapper = await mountView([item("review")], "", [
    { scope: "shared", message: "No SKILL.md header in /data/skills/bad" },
  ]);
  expect(wrapper.find('[data-testid="skills-broken"]').text()).toContain("1 навык не загружен");
  expect(wrapper.find('[data-testid="skills-broken"]').text()).toContain("/data/skills/bad");
  expect(wrapper.find('[data-skill="review"]').exists()).toBe(true);
});

it("arrow keys move between the tabs", async () => {
  const wrapper = await mountView([]);
  await wrapper.find('[role="tablist"]').trigger("keydown", { key: "ArrowRight" });
  expect(wrapper.find('[data-testid="skills-tab-project"]').attributes("aria-selected")).toBe("true");
  expect(document.activeElement?.id).toBe("skills-tab-project");
});
