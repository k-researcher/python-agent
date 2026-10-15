import { describe, expect, it, vi } from "vitest";
import { nextTick } from "vue";
import { mount } from "@vue/test-utils";
import LeftSidebar from "./LeftSidebar.vue";
import type { SessionNode } from "./types";

const tree: SessionNode[] = [
  {
    id: "parent",
    title: "Родитель",
    status: "completed",
    archived: false,
    children: [
      { id: "child", title: "Дочерний", status: "running", archived: false, children: [] },
    ],
  },
  { id: "leaf", title: "Лист", status: "error", archived: false, children: [] },
  { id: "old", title: "Архивный", status: "completed", archived: true, children: [] },
];

const projects = [
  { id: "p1", name: "Проект один" },
  { id: "p2", name: "Проект два" },
];

function mountSidebar(props: Record<string, unknown> = {}) {
  return mount(LeftSidebar, {
    props: {
      projects,
      selectedProjectId: "p1",
      tree,
      selectedSessionId: null,
      ...props,
    },
  });
}

function labels(wrapper: ReturnType<typeof mountSidebar>): string[] {
  return wrapper.findAll(".tree-row__label").map((node) => node.text());
}

describe("LeftSidebar — projects", () => {
  it("renders the project list and marks the selected one", () => {
    const wrapper = mountSidebar();
    const rows = wrapper.findAll('[data-project]');
    expect(rows.map((row) => row.text())).toEqual(["Проект один", "Проект два"]);
    expect(rows[0].classes()).toContain("project-row--selected");
    expect(rows[0].attributes("aria-current")).toBe("true");
    expect(rows[1].attributes("aria-current")).toBeUndefined();
  });

  it("emits selectProject on click", async () => {
    const wrapper = mountSidebar();
    await wrapper.findAll('[data-project]')[1].trigger("click");
    expect(wrapper.emitted("selectProject")).toEqual([["p2"]]);
  });

  it("shows the create form after clicking '+'", async () => {
    const wrapper = mountSidebar({ showCreateForm: true });
    expect(wrapper.find('[data-create-form]').exists()).toBe(true);
  });

  it("emits createProject with trimmed values", async () => {
    const wrapper = mountSidebar({ showCreateForm: true });
    await wrapper.find('[data-create-name]').setValue("  Новый проект  ");
    await wrapper.find('[data-create-path]').setValue(" /abs/path ");
    await wrapper.find('[data-create-form]').trigger("submit");

    expect(wrapper.emitted("createProject")).toEqual([
      [{ name: "Новый проект", rootPath: "/abs/path" }],
    ]);
  });

  it("does not emit createProject for whitespace-only or missing fields", async () => {
    const wrapper = mountSidebar({ showCreateForm: true });

    // Empty name, valid path.
    await wrapper.find('[data-create-name]').setValue("  ");
    await wrapper.find('[data-create-path]').setValue("/abs/path");
    await wrapper.find('[data-create-form]').trigger("submit");
    expect(wrapper.emitted("createProject")).toBeUndefined();

    // Valid name, missing path.
    await wrapper.find('[data-create-name]').setValue("Имя");
    await wrapper.find('[data-create-path]').setValue("");
    await wrapper.find('[data-create-form]').trigger("submit");
    expect(wrapper.emitted("createProject")).toBeUndefined();
  });

  it("blocks duplicate submission and cancellation while creation is pending", async () => {
    const wrapper = mountSidebar({ showCreateForm: true });
    await wrapper.get('[data-create-name]').setValue("Project");
    await wrapper.get('[data-create-path]').setValue("/abs/path");
    await wrapper.setProps({ isCreatingProject: true });

    expect(wrapper.get('[data-create-form]').attributes("aria-busy")).toBe("true");
    for (const selector of [
      '[data-create-name]',
      '[data-create-path]',
      '[data-create-submit]',
      '[data-create-cancel]',
    ]) {
      expect(wrapper.get(selector).attributes("disabled")).toBeDefined();
    }
    await wrapper.get('[data-create-form]').trigger("submit");
    await wrapper.get('[data-create-cancel]').trigger("click");
    expect(wrapper.emitted("createProject")).toBeUndefined();
    expect(wrapper.emitted("closeCreateForm")).toBeUndefined();

    await wrapper.setProps({ isCreatingProject: false });
    await wrapper.get('[data-create-form]').trigger("submit");
    expect(wrapper.emitted("createProject")).toEqual([
      [{ name: "Project", rootPath: "/abs/path" }],
    ]);
  });

  it("clears project fields after successful creation closes the controlled form", async () => {
    const wrapper = mountSidebar({ showCreateForm: true });
    await wrapper.get('[data-create-name]').setValue("Project");
    await wrapper.get('[data-create-path]').setValue("/abs/path");
    await wrapper.setProps({ isCreatingProject: true });
    await wrapper.setProps({ isCreatingProject: false, showCreateForm: false });
    expect(wrapper.find('[data-create-form]').exists()).toBe(false);

    await wrapper.setProps({ showCreateForm: true });
    expect((wrapper.get('[data-create-name]').element as HTMLInputElement).value).toBe("");
    expect((wrapper.get('[data-create-path]').element as HTMLInputElement).value).toBe("");
  });

  it("emits newSession, openSettings and signOut", async () => {
    const wrapper = mountSidebar();
    await wrapper.find('[data-new-session]').trigger("click");
    await wrapper.find('[data-settings]').trigger("click");
    await wrapper.find('[data-signout]').trigger("click");
    expect(wrapper.emitted("newSession")).toHaveLength(1);
    expect(wrapper.emitted("openSettings")).toHaveLength(1);
    expect(wrapper.emitted("signOut")).toHaveLength(1);
  });
});

describe("LeftSidebar — session tree", () => {
  it("exposes WAI-ARIA tree semantics", () => {
    const wrapper = mountSidebar();
    expect(wrapper.get(".session-tree").attributes("role")).toBe("tree");
    expect(wrapper.get(".session-tree").attributes("aria-label")).toBe("Сессии");
    const node = wrapper.findAll(".tree-node")[0];
    expect(node.attributes("role")).toBe("treeitem");
  });

  it("renders tree rows with correct depth (aria-level) and indentation", () => {
    const wrapper = mountSidebar();
    const nodes = wrapper.findAll(".tree-node");
    // parent(level 1) → child(level 2) → leaf(level 1) → archive group(level 1)
    expect(nodes.map((node) => Number(node.attributes("aria-level")))).toEqual([1, 2, 1, 1]);
    const rows = wrapper.findAll(".tree-row");
    expect(rows[0].attributes("style")).toContain("padding-left: 8px");
    expect(rows[1].attributes("style")).toContain("padding-left: 24px");
  });

  it("applies the spec status dot classes to each session", () => {
    const wrapper = mountSidebar();
    const dotFor = (title: string) =>
      wrapper
        .findAll(".tree-row")
        .find((row) => row.text().includes(title))!
        .find(".status-dot");

    expect(dotFor("Родитель").classes()).toContain("status-dot--completed");
    expect(dotFor("Дочерний").classes()).toContain("status-dot--running");
    expect(dotFor("Лист").classes()).toContain("status-dot--error");
  });

  it("marks aria-selected on the selected session only", () => {
    const wrapper = mountSidebar({ selectedSessionId: "child" });
    const nodes = wrapper.findAll(".tree-node");
    const selected = nodes.find((node) => node.attributes("aria-selected") === "true");
    expect(selected?.attributes("aria-level")).toBe("2");
    expect(nodes.filter((node) => node.attributes("aria-selected") === "true")).toHaveLength(1);
  });

  it("keeps the archive collapsed and hidden by default", () => {
    const wrapper = mountSidebar();
    expect(labels(wrapper)).toEqual(["Родитель", "Дочерний", "Лист", "Архив"]);
    const archive = wrapper.findAll(".tree-node").find((n) => n.text().includes("Архив"))!;
    expect(archive.attributes("aria-expanded")).toBe("false");
  });

  it("expands the archive group on click and shows archived sessions", async () => {
    const wrapper = mountSidebar();
    const rows = wrapper.findAll(".tree-row");
    await rows[3].trigger("click"); // archive group row

    expect(labels(wrapper)).toEqual(["Родитель", "Дочерний", "Лист", "Архив", "Архивный"]);
    const archive = wrapper.findAll(".tree-node")[3];
    expect(archive.attributes("aria-expanded")).toBe("true");
    const archivedNode = wrapper.findAll(".tree-node")[4];
    expect(archivedNode.attributes("aria-level")).toBe("2");
  });

  it("emits selectSession when a session row is clicked", async () => {
    const wrapper = mountSidebar();
    await wrapper.findAll(".tree-row")[2].trigger("click"); // leaf
    expect(wrapper.emitted("selectSession")).toEqual([["leaf"]]);
  });

  it("moves DOM focus into the tree after a mouse click so ArrowDown still works", async () => {
    const wrapper = mountSidebar();
    const rows = wrapper.findAll(".tree-row");
    await rows[2].trigger("click"); // leaf
    await nextTick();

    const nodes = wrapper.findAll(".tree-node");
    expect(nodes[2].attributes("tabindex")).toBe("0");
    expect(nodes[1].attributes("tabindex")).toBe("-1");

    const li = rows[2].element.closest(".tree-node") as HTMLElement;
    li.dispatchEvent(
      new KeyboardEvent("keydown", { key: "ArrowDown", bubbles: true, cancelable: true }),
    );
    await nextTick();
    expect(wrapper.findAll(".tree-node")[3].attributes("tabindex")).toBe("0");
  });

  it("does not select the archive group on Enter, only expands/collapses it", async () => {
    const wrapper = mountSidebar();
    const treeEl = wrapper.get(".session-tree");
    for (let i = 0; i < 3; i++) await treeEl.trigger("keydown", { key: "ArrowDown" });

    await treeEl.trigger("keydown", { key: "Enter" });
    expect(wrapper.emitted("selectSession")).toBeUndefined();
    expect(labels(wrapper)).toEqual(["Родитель", "Дочерний", "Лист", "Архив", "Архивный"]);
    expect(wrapper.findAll(".tree-node")[3].attributes("aria-expanded")).toBe("true");
  });

  it("does not collide when a session id equals the archive group key", () => {
    const conflictingTree: SessionNode[] = [
      { id: "arch", title: "Настоящий arch", status: "running", archived: false, children: [] },
      { id: "x", title: "Архивный x", status: "completed", archived: true, children: [] },
    ];
    const wrapper = mountSidebar({ tree: conflictingTree });

    const rows = wrapper.findAll(".tree-row");
    expect(rows.map((row) => row.text().replace(/\s+/g, " ").trim())).toEqual([
      "Настоящий arch",
      "Архив",
    ]);
    const nodes = wrapper.findAll(".tree-node");
    expect(nodes.map((node) => node.attributes("data-node-key"))).toEqual(["n:arch", "arch"]);
    expect(nodes.map((node) => Number(node.attributes("aria-level")))).toEqual([1, 1]);
  });

  it("shows an empty hint when there are no sessions", () => {
    const wrapper = mountSidebar({ tree: [] });
    expect(wrapper.text()).toContain("Нет сессий");
    expect(wrapper.findAll(".tree-node")).toHaveLength(0);
  });
});

describe("LeftSidebar — keyboard navigation", () => {
  it("moves focus with ArrowDown across visible nodes", async () => {
    const wrapper = mountSidebar();
    const treeEl = wrapper.get(".session-tree");
    expect(wrapper.findAll(".tree-node")[0].attributes("tabindex")).toBe("0");

    await treeEl.trigger("keydown", { key: "ArrowDown" });
    const nodes = wrapper.findAll(".tree-node");
    expect(nodes[0].attributes("tabindex")).toBe("-1");
    expect(nodes[1].attributes("tabindex")).toBe("0");
  });

  it("clamps ArrowUp at the first node", async () => {
    const wrapper = mountSidebar();
    const treeEl = wrapper.get(".session-tree");
    await treeEl.trigger("keydown", { key: "ArrowUp" });
    expect(wrapper.findAll(".tree-node")[0].attributes("tabindex")).toBe("0");
  });

  it("collapses a parent with ArrowLeft and re-expands it with ArrowRight", async () => {
    const wrapper = mountSidebar();
    const treeEl = wrapper.get(".session-tree");

    await treeEl.trigger("keydown", { key: "ArrowLeft" });
    expect(labels(wrapper)).toEqual(["Родитель", "Лист", "Архив"]);
    expect(wrapper.findAll(".tree-node")[0].attributes("aria-expanded")).toBe("false");

    await treeEl.trigger("keydown", { key: "ArrowRight" });
    expect(labels(wrapper)).toEqual(["Родитель", "Дочерний", "Лист", "Архив"]);
    expect(wrapper.findAll(".tree-node")[0].attributes("aria-expanded")).toBe("true");
  });

  it("moves focus to the first child with ArrowRight on an expanded parent", async () => {
    const wrapper = mountSidebar();
    const treeEl = wrapper.get(".session-tree");
    await treeEl.trigger("keydown", { key: "ArrowRight" });
    const nodes = wrapper.findAll(".tree-node");
    expect(nodes[1].attributes("tabindex")).toBe("0");
  });

  it("moves focus to the parent with ArrowLeft on a child", async () => {
    const wrapper = mountSidebar();
    const treeEl = wrapper.get(".session-tree");
    await treeEl.trigger("keydown", { key: "ArrowDown" });
    await treeEl.trigger("keydown", { key: "ArrowLeft" });

    const nodes = wrapper.findAll(".tree-node");
    expect(nodes[0].attributes("tabindex")).toBe("0");
    expect(nodes[1].attributes("tabindex")).toBe("-1");
  });

  it("selects a leaf with Enter", async () => {
    const wrapper = mountSidebar();
    const treeEl = wrapper.get(".session-tree");
    await treeEl.trigger("keydown", { key: "ArrowDown" });
    await treeEl.trigger("keydown", { key: "ArrowDown" });
    await treeEl.trigger("keydown", { key: "Enter" });
    expect(wrapper.emitted("selectSession")).toEqual([["leaf"]]);
  });

  it("toggles a parent with Enter and also selects it", async () => {
    const wrapper = mountSidebar();
    const treeEl = wrapper.get(".session-tree");
    await treeEl.trigger("keydown", { key: "Enter" });
    expect(wrapper.emitted("selectSession")).toEqual([["parent"]]);
    expect(labels(wrapper)).toEqual(["Родитель", "Лист", "Архив"]);
  });
});
