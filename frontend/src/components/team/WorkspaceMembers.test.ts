import { afterEach, describe, expect, it } from 'vitest';
import { enableAutoUnmount, mount } from '@vue/test-utils';
import WorkspaceMembers, { type WorkspaceMember } from './WorkspaceMembers.vue';

enableAutoUnmount(afterEach);

function member(overrides: Partial<WorkspaceMember> = {}): WorkspaceMember {
  return {
    user_id: 'member-id',
    display_name: 'Участник команды',
    role: 'member',
    enabled: true,
    can_remove: true,
    assignable_roles: ['viewer'],
    ...overrides,
  };
}

function render(members = [member()], overrides = {}) {
  return mount(WorkspaceMembers, { props: { members, canInvite: true, ...overrides } });
}

describe('WorkspaceMembers', () => {
  it('renders an empty state without fabricated users', () => {
    const wrapper = render([]);
    expect(wrapper.text()).toContain('Нет участников');
    expect(wrapper.findAll('li')).toHaveLength(0);
  });

  it('renders loading instead of stale members', () => {
    const wrapper = render([member()], { loading: true });
    expect(wrapper.get('[role="status"]').text()).toBe('Загрузка участников…');
    expect(wrapper.find('li').exists()).toBe(false);
    expect(wrapper.get<HTMLButtonElement>('button').element.disabled).toBe(true);
  });

  it('renders an error and emits a retry without member actions', async () => {
    const wrapper = render([member()], { error: 'Ошибка загрузки' });
    expect(wrapper.get('[role="alert"]').text()).toContain('Ошибка загрузки');
    expect(wrapper.find('li').exists()).toBe(false);
    await wrapper.get('[role="alert"] button').trigger('click');
    expect(wrapper.emitted('retry')).toEqual([[]]);
  });

  it('disables retry while loading', () => {
    const wrapper = render([], { error: 'Ошибка', loading: true });
    expect(wrapper.get<HTMLButtonElement>('[role="alert"] button').element.disabled).toBe(true);
  });

  it('emits invitation only when the capability is present', async () => {
    const wrapper = render();
    await wrapper.get('header button').trigger('click');
    expect(wrapper.emitted('invite')).toEqual([[]]);
    await wrapper.setProps({ canInvite: false });
    expect(wrapper.find('header button').exists()).toBe(false);
  });

  it('makes the owner immutable even with malformed capabilities', () => {
    const wrapper = render([member({ role: 'owner', assignable_roles: ['admin'] })]);
    expect(wrapper.get('li').text()).toContain('Владелец');
    expect(wrapper.find('select').exists()).toBe(false);
    expect(wrapper.find('li button').exists()).toBe(false);
  });

  it('shows disabled user status without inventing capabilities', () => {
    const wrapper = render([member({ enabled: false, can_remove: false, assignable_roles: [] })]);
    expect(wrapper.get('li').text()).toContain('Отключён');
    expect(wrapper.find('li button').exists()).toBe(false);
    expect(wrapper.find('select').exists()).toBe(false);
  });

  it('emits removal using the server user id', async () => {
    const wrapper = render();
    await wrapper.get('li button').trigger('click');
    expect(wrapper.emitted('remove')).toEqual([['member-id']]);
  });

  it('emits a permitted role without presenting it as saved', async () => {
    const wrapper = render();
    const select = wrapper.get<HTMLSelectElement>('select');
    expect(select.element.value).toBe('member');
    await select.setValue('viewer');
    expect(wrapper.emitted('changeRole')).toEqual([['member-id', 'viewer']]);
    expect(select.element.value).toBe('member');
    await wrapper.setProps({ members: [member({ role: 'viewer' })] });
    expect(select.element.value).toBe('viewer');
  });

  it.each(['admin', 'owner', 'unknown'])('rejects a forged %s selection', async (role) => {
    const wrapper = render();
    const select = wrapper.get<HTMLSelectElement>('select');
    const option = document.createElement('option');
    option.value = role;
    select.element.append(option);
    await select.setValue(role);
    expect(wrapper.emitted('changeRole')).toBeUndefined();
    expect(select.element.value).toBe('member');
  });

  it('does not emit for the unchanged role', async () => {
    const wrapper = render();
    await wrapper.get('select').setValue('member');
    expect(wrapper.emitted('changeRole')).toBeUndefined();
  });

  it('ignores stale removal after permission revocation', async () => {
    const wrapper = render();
    const button = wrapper.get('li button');
    await wrapper.setProps({ members: [member({ can_remove: false })] });
    await button.trigger('click');
    expect(wrapper.emitted('remove')).toBeUndefined();
  });

  it('ignores a stale selection after its user disappears', async () => {
    const wrapper = render();
    const select = wrapper.get('select');
    await wrapper.setProps({ members: [] });
    await select.setValue('viewer');
    expect(wrapper.emitted('changeRole')).toBeUndefined();
  });

  it('blocks invitation, removal and role changes while busy', async () => {
    const wrapper = render([member()], { busyUserId: 'member-id' });
    for (const button of wrapper.findAll<HTMLButtonElement>('button')) {
      expect(button.element.disabled).toBe(true);
      await button.trigger('click');
    }
    const select = wrapper.get<HTMLSelectElement>('select');
    expect(select.element.disabled).toBe(true);
    await select.setValue('viewer');
    expect(wrapper.emitted('invite')).toBeUndefined();
    expect(wrapper.emitted('remove')).toBeUndefined();
    expect(wrapper.emitted('changeRole')).toBeUndefined();
  });
});
