<script setup lang="ts">
// A modal dialog on the native <dialog> element: the browser keeps the focus inside, Escape
// closes it, and the page behind it is inert. On a narrow screen it takes the full screen,
// as Material 3 asks for long forms.

import { nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";

const props = defineProps<{ open: boolean; title: string; wide?: boolean; alert?: boolean }>();
const emit = defineEmits<{ close: [] }>();
const element = ref<HTMLDialogElement | null>(null);
const titleId = `dialog-title-${Math.random().toString(36).slice(2, 10)}`;

function show(): void {
  const dialog = element.value;
  if (!dialog || dialog.open) return;
  // Test environments may have no showModal; the open attribute keeps the content visible.
  if (typeof dialog.showModal === "function") dialog.showModal();
  else dialog.setAttribute("open", "");
  void nextTick(() => {
    const target = dialog.querySelector<HTMLElement>("[autofocus], [data-autofocus]");
    target?.focus();
  });
}

function hide(): void {
  const dialog = element.value;
  if (!dialog?.open) return;
  if (typeof dialog.close === "function") dialog.close();
  else dialog.removeAttribute("open");
}

watch(
  () => props.open,
  (open) => (open ? show() : hide()),
);
onMounted(() => {
  if (props.open) show();
});
onBeforeUnmount(hide);

function cancel(event: Event): void {
  // Escape: the parent decides, so a busy form can stay open.
  event.preventDefault();
  emit("close");
}
</script>

<template>
  <dialog
    ref="element"
    class="dialog"
    :class="{ 'dialog--wide': wide }"
    :role="alert ? 'alertdialog' : 'dialog'"
    :aria-labelledby="titleId"
    @cancel="cancel"
  >
    <template v-if="open">
      <h2 :id="titleId" class="dialog__title">{{ title }}</h2>
      <div class="dialog__body"><slot /></div>
      <div class="dialog__actions"><slot name="actions" /></div>
    </template>
  </dialog>
</template>

<style scoped>
.dialog {
  width: min(440px, calc(100vw - 32px));
  max-height: calc(100dvh - 48px);
  padding: 20px 24px 16px;
  border: 1px solid var(--border);
  border-radius: 16px;
  background: var(--surface);
  color: var(--text);
  box-shadow: var(--shadow);
  box-sizing: border-box;
  overflow: auto;
}

.dialog--wide {
  width: min(760px, calc(100vw - 32px));
}

.dialog::backdrop {
  background: rgba(20, 14, 9, 0.42);
}

.dialog[open] {
  animation: dialog-in 0.18s ease-out;
}

.dialog__title {
  margin: 0 0 12px;
  font-size: 18px;
  font-weight: 600;
}

.dialog__body {
  display: flex;
  flex-direction: column;
  gap: 14px;
  font-size: 14px;
}

/* Actions sit at the end, the confirming action last, as both guidelines put them. */
.dialog__actions {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 8px;
  margin-top: 20px;
}

@media (max-width: 599px) {
  .dialog--wide {
    width: 100vw;
    max-width: 100vw;
    height: 100dvh;
    max-height: 100dvh;
    margin: 0;
    border: 0;
    border-radius: 0;
  }
}

@keyframes dialog-in {
  from {
    opacity: 0;
    transform: translateY(8px) scale(0.98);
  }
}

@media (prefers-reduced-motion: reduce) {
  .dialog[open] {
    animation: none;
  }
}
</style>
