<script setup lang="ts">
import { computed } from "vue";

interface Workspace {
  id: string;
  name: string;
  role: "owner" | "admin" | "member" | "viewer";
}

const props = defineProps<{
  workspaces: Workspace[];
  selectedWorkspaceId: string | null;
  loading?: boolean;
  error?: string;
}>();

const emit = defineEmits<{
  select: [workspaceId: string];
  retry: [];
}>();

const roleLabels: Record<Workspace["role"], string> = {
  owner: "Владелец",
  admin: "Администратор",
  member: "Участник",
  viewer: "Наблюдатель",
};

const selectedWorkspace = computed(() =>
  props.workspaces.find((workspace) => workspace.id === props.selectedWorkspaceId),
);
const selectedId = computed(() => selectedWorkspace.value?.id ?? "");
const disabled = computed(() => Boolean(props.loading || props.error || !props.workspaces.length));

function onSelect(event: Event): void {
  if (disabled.value || !(event.target instanceof HTMLSelectElement)) return;
  const workspaceId = event.target.value;
  if (workspaceId !== selectedId.value && props.workspaces.some((workspace) => workspace.id === workspaceId)) {
    emit("select", workspaceId);
  }
}
</script>

<template>
  <div class="workspace-switcher">
    <select
      :value="selectedId"
      :disabled="disabled"
      aria-label="Командное пространство"
      data-testid="workspace-switcher"
      class="workspace-switcher__select"
      @change="onSelect"
    >
      <option value="" disabled>Выберите пространство</option>
      <option v-for="workspace in workspaces" :key="workspace.id" :value="workspace.id">
        {{ workspace.name }}
      </option>
    </select>
    <div v-if="error" class="workspace-switcher__error" role="alert">
      <span>{{ error }}</span>
      <button type="button" class="workspace-switcher__retry" @click="emit('retry')">Повторить</button>
    </div>
    <span v-else-if="loading" class="workspace-switcher__status" role="status">Загрузка...</span>
    <span v-else-if="!workspaces.length" class="workspace-switcher__status">Нет доступных пространств</span>
    <span v-else-if="selectedWorkspace" class="workspace-switcher__status">
      {{ roleLabels[selectedWorkspace.role] }}
    </span>
  </div>
</template>

<style scoped>
.workspace-switcher {
  display: flex;
  flex-direction: column;
  gap: 4px;
  width: 100%;
  max-width: 240px;
  min-width: 0;
}
.workspace-switcher__select {
  box-sizing: border-box;
  width: 100%;
  min-width: 0;
  padding: 6px 8px;
  font: inherit;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--surface);
  color: var(--text);
  text-overflow: ellipsis;
}
.workspace-switcher__status {
  font-size: 12px;
  color: var(--text-muted);
}
.workspace-switcher__error {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  overflow-wrap: anywhere;
  font-size: 12px;
  color: var(--danger);
}
.workspace-switcher__retry {
  border: 0;
  padding: 0;
  background: none;
  color: var(--link);
  cursor: pointer;
}
</style>
