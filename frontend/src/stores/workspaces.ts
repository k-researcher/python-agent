import { defineStore } from "pinia";
import { ref, computed } from "vue";
import { listWorkspaces, type WorkspaceSummary } from "../api/workspaces";

export type ScopeGeneration = number;

export const useWorkspaceStore = defineStore("workspaces", () => {
  const workspaces = ref<WorkspaceSummary[]>([]);
  const selectedWorkspaceId = ref<string | null>(null);
  const loading = ref(false);
  const error = ref<string | null>(null);
  const scopeGeneration = ref<ScopeGeneration>(0);

  let latestRequestId = 0;

  const selectedWorkspace = computed<WorkspaceSummary | null>(() => {
    if (!selectedWorkspaceId.value) return null;
    return workspaces.value.find((workspace) => workspace.id === selectedWorkspaceId.value) || null;
  });

  function incrementScope() {
    scopeGeneration.value += 1;
  }

  function reset() {
    latestRequestId += 1;
    incrementScope();
    workspaces.value = [];
    selectedWorkspaceId.value = null;
    error.value = null;
    loading.value = false;
  }

  async function load(): Promise<void> {
    const requestId = ++latestRequestId;
    loading.value = true;
    error.value = null;

    try {
      const loadedWorkspaces = await listWorkspaces();

      if (requestId !== latestRequestId) return;

      const oldWorkspace = selectedWorkspace.value;
      const oldId = selectedWorkspaceId.value;

      let newSelectedId: string | null = null;
      if (oldId && loadedWorkspaces.some((workspace) => workspace.id === oldId)) {
        newSelectedId = oldId;
      } else if (loadedWorkspaces.length > 0) {
        newSelectedId = loadedWorkspaces[0].id;
      }

      const newWorkspace = loadedWorkspaces.find((workspace) => workspace.id === newSelectedId);

      if (oldId !== newSelectedId) {
        incrementScope();
      } else if (oldWorkspace && newWorkspace) {
        const roleChanged = oldWorkspace.role !== newWorkspace.role;
        const capsChanged =
          oldWorkspace.capabilities.view_project !== newWorkspace.capabilities.view_project ||
          oldWorkspace.capabilities.run_agent !== newWorkspace.capabilities.run_agent ||
          oldWorkspace.capabilities.manage_members !== newWorkspace.capabilities.manage_members ||
          oldWorkspace.capabilities.manage_workspace !== newWorkspace.capabilities.manage_workspace ||
          oldWorkspace.capabilities.manage_models !== newWorkspace.capabilities.manage_models;

        if (roleChanged || capsChanged) {
          incrementScope();
        }
      }

      workspaces.value = loadedWorkspaces;
      selectedWorkspaceId.value = newSelectedId;
      loading.value = false;
    } catch (failure: unknown) {
      if (requestId !== latestRequestId) return;

      incrementScope();
      workspaces.value = [];
      selectedWorkspaceId.value = null;
      error.value = failure instanceof Error ? failure.message : "Не удалось загрузить пространства";
      loading.value = false;
    }
  }

  function selectWorkspace(id: string): boolean {
    const exists = workspaces.value.some((workspace) => workspace.id === id);
    if (!exists) return false;

    const isNew = selectedWorkspaceId.value !== id;
    if (isNew) {
      incrementScope();
    }
    selectedWorkspaceId.value = id;
    return true;
  }

  return {
    workspaces,
    selectedWorkspaceId,
    loading,
    error,
    scopeGeneration,
    selectedWorkspace,
    load,
    selectWorkspace,
    reset,
  };
});
