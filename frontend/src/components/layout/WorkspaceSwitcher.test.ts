import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import WorkspaceSwitcher from "./WorkspaceSwitcher.vue";

interface Workspace {
  id: string;
  name: string;
  role: "owner" | "admin" | "member" | "viewer";
}

const workspaces: Workspace[] = [
  { id: "1", name: "Team Alpha", role: "owner" },
  { id: "2", name: "Beta Lab", role: "member" },
];

describe("WorkspaceSwitcher", () => {
  it("emits select when a new workspace is selected", async () => {
    const wrapper = mount(WorkspaceSwitcher, {
      props: {
        workspaces,
        selectedWorkspaceId: "1",
      },
    });

    const select = wrapper.find<HTMLSelectElement>('[data-testid="workspace-switcher"]');
    await select.setValue("2");

    expect(wrapper.emitted("select")).toBeTruthy();
    expect(wrapper.emitted("select")![0]).toEqual(["2"]);
  });

  it("does not emit select if the ID is invalid", async () => {
    const wrapper = mount(WorkspaceSwitcher, {
      props: {
        workspaces,
        selectedWorkspaceId: "999",
      },
    });

    const select = wrapper.find<HTMLSelectElement>('[data-testid="workspace-switcher"]');
    await select.setValue("999");

    expect(wrapper.emitted("select")).toBeFalsy();
  });

  it("shows error state and emits retry", async () => {
    const wrapper = mount(WorkspaceSwitcher, {
      props: {
        workspaces: [],
        selectedWorkspaceId: null,
        error: "Failed to fetch",
      },
    });

    expect(wrapper.find('[role="alert"]').exists()).toBe(true);
    expect(wrapper.text()).toContain("Failed to fetch");

    await wrapper.find(".workspace-switcher__retry").trigger("click");
    expect(wrapper.emitted("retry")).toBeTruthy();
  });

  it("shows loading state and disables select", async () => {
    const wrapper = mount(WorkspaceSwitcher, {
      props: {
        workspaces,
        selectedWorkspaceId: "1",
        loading: true,
      },
    });

    expect(wrapper.text()).toContain("Загрузка...");
    expect(wrapper.get('[data-testid="workspace-switcher"]').attributes("disabled")).toBeDefined();
  });

  it("shows empty state when no workspaces available", async () => {
    const wrapper = mount(WorkspaceSwitcher, {
      props: {
        workspaces: [],
        selectedWorkspaceId: null,
      },
    });

    expect(wrapper.text()).toContain("Нет доступных пространств");
  });

  it("displays the correct Russian role for the selected workspace", async () => {
    const wrapper = mount(WorkspaceSwitcher, {
      props: {
        workspaces,
        selectedWorkspaceId: "2",
      },
    });

    expect(wrapper.text()).toContain("Участник");
  });

  it("shows placeholder when selectedWorkspaceId is unknown", async () => {
    const wrapper = mount(WorkspaceSwitcher, {
      props: {
        workspaces,
        selectedWorkspaceId: "unknown",
      },
    });

    const select = wrapper.find<HTMLSelectElement>('[data-testid="workspace-switcher"]');
    expect(select.element.options[0].text).toBe("Выберите пространство");
  });

  it("does not emit select when props change but ID remains the same", async () => {
    const wrapper = mount(WorkspaceSwitcher, {
      props: {
        workspaces,
        selectedWorkspaceId: "1",
      },
    });

    await wrapper.setProps({ selectedWorkspaceId: "2" });
    expect(wrapper.emitted("select")).toBeFalsy();
    expect(wrapper.get<HTMLSelectElement>('[data-testid="workspace-switcher"]').element.value).toBe("2");
  });
});
