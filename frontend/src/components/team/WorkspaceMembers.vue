<script setup lang="ts">
import { computed } from 'vue';

export type MemberRole = 'owner' | 'admin' | 'member' | 'viewer';
export type AssignableRole = Exclude<MemberRole, 'owner'>;

export interface WorkspaceMember {
  user_id: string;
  display_name: string;
  role: MemberRole;
  enabled: boolean;
  can_remove: boolean;
  assignable_roles: AssignableRole[];
}

const props = defineProps<{
  members: WorkspaceMember[];
  canInvite: boolean;
  loading?: boolean;
  error?: string;
  busyUserId?: string | null;
}>();

const emit = defineEmits<{
  invite: [];
  retry: [];
  remove: [userId: string];
  changeRole: [userId: string, role: AssignableRole];
}>();

const roleLabels: Record<MemberRole, string> = {
  owner: 'Владелец',
  admin: 'Администратор',
  member: 'Участник',
  viewer: 'Наблюдатель',
};
const assignableRoles: AssignableRole[] = ['admin', 'member', 'viewer'];
const isBusy = computed(() => Boolean(props.loading || props.error || props.busyUserId));

function permittedRoles(member: WorkspaceMember): AssignableRole[] {
  if (member.role === 'owner') return [];
  return assignableRoles.filter((role) => member.assignable_roles.includes(role));
}

function invite(): void {
  if (!isBusy.value && props.canInvite) emit('invite');
}

function remove(userId: string): void {
  const member = props.members.find((candidate) => candidate.user_id === userId);
  if (!isBusy.value && member && member.role !== 'owner' && member.can_remove) {
    emit('remove', userId);
  }
}

function changeRole(userId: string, event: Event): void {
  const target = event.target;
  if (!(target instanceof HTMLSelectElement)) return;
  const requestedRole = target.value;
  const member = props.members.find((candidate) => candidate.user_id === userId);
  target.value = member?.role ?? '';
  if (!isBusy.value && member && requestedRole !== member.role) {
    const permittedRole = permittedRoles(member).find((role) => role === requestedRole);
    if (permittedRole) emit('changeRole', userId, permittedRole);
  }
}
</script>

<template>
  <section class="workspace-members" aria-labelledby="workspace-members-title">
    <header class="workspace-members__header">
      <h2 id="workspace-members-title">Участники</h2>
      <button v-if="canInvite" type="button" :disabled="isBusy" @click="invite">
        Пригласить
      </button>
    </header>
    <div v-if="error" class="workspace-members__error" role="alert">
      <span>{{ error }}</span>
      <button type="button" :disabled="loading" @click="emit('retry')">Повторить</button>
    </div>
    <p v-else-if="loading" role="status">Загрузка участников…</p>
    <ul v-else-if="members.length" class="workspace-members__list">
      <li v-for="member in members" :key="member.user_id" class="workspace-members__item">
        <div class="workspace-members__info">
          <span class="workspace-members__name">{{ member.display_name }}</span>
          <span class="workspace-members__role">{{ roleLabels[member.role] }}</span>
          <span v-if="!member.enabled" class="workspace-members__role">Отключён</span>
        </div>
        <div v-if="member.role !== 'owner'" class="workspace-members__actions">
          <select
            v-if="permittedRoles(member).length"
            :value="member.role"
            :disabled="isBusy"
            :aria-label="`Изменить роль для ${member.display_name}`"
            @change="changeRole(member.user_id, $event)"
          >
            <option :value="member.role">{{ roleLabels[member.role] }}</option>
            <option
              v-for="role in permittedRoles(member).filter((role) => role !== member.role)"
              :key="role"
              :value="role"
            >{{ roleLabels[role] }}</option>
          </select>
          <button
            v-if="member.can_remove"
            type="button"
            class="workspace-members__remove"
            :disabled="isBusy"
            :aria-label="`Удалить участника ${member.display_name}`"
            @click="remove(member.user_id)"
          >Удалить</button>
        </div>
      </li>
    </ul>
    <p v-else>Нет участников</p>
  </section>
</template>

<style scoped>
.workspace-members {
  min-width: 0;
  color: var(--text);
  background: var(--surface);
  font-family: var(--font-ui);
}
.workspace-members__header,
.workspace-members__item,
.workspace-members__actions,
.workspace-members__error {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}
.workspace-members__header,
.workspace-members__item {
  padding: 12px;
  border-bottom: 1px solid var(--border);
  justify-content: space-between;
}
h2 { margin: 0; font-size: 14px; }
p { padding: 12px; color: var(--text-muted); }
.workspace-members__list { list-style: none; padding: 0; margin: 0; }
.workspace-members__info { flex: 1; min-width: 100px; display: grid; gap: 4px; }
.workspace-members__name,
.workspace-members__error { overflow-wrap: anywhere; }
.workspace-members__role { color: var(--text-muted); font-size: 12px; }
.workspace-members__error { padding: 12px; color: var(--danger); }
button,
select {
  max-width: 100%;
  font: inherit;
  font-size: 12px;
  color: var(--text);
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 6px 8px;
}
button { cursor: pointer; }
.workspace-members__remove { color: var(--danger); }
button:disabled,
select:disabled { opacity: 0.5; cursor: not-allowed; }
</style>
