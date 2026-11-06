<script setup lang="ts">
// An on/off switch that acts at once, as in Material 3 and Apple HIG: no save button.
// The visible label stays outside; ``label`` names the switch for screen readers.

defineProps<{ modelValue: boolean; label: string; disabled?: boolean }>();
const emit = defineEmits<{ "update:modelValue": [value: boolean] }>();
</script>

<template>
  <button
    type="button"
    role="switch"
    class="switch"
    :aria-checked="modelValue"
    :aria-label="label"
    :title="label"
    :disabled="disabled"
    @click="emit('update:modelValue', !modelValue)"
  >
    <span class="switch__track" aria-hidden="true"><span class="switch__thumb" /></span>
  </button>
</template>

<style scoped>
/* The button is 44 px high so it is easy to hit; the track inside is smaller. */
.switch {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 52px;
  min-height: 44px;
  padding: 0;
  border: 0;
  background: none;
  cursor: pointer;
}

.switch__track {
  position: relative;
  width: 44px;
  height: 26px;
  border: 2px solid var(--text-muted);
  border-radius: 13px;
  background: var(--bg-subtle);
  box-sizing: border-box;
  transition: background-color 0.2s ease, border-color 0.2s ease;
}

.switch__thumb {
  position: absolute;
  top: 50%;
  left: 4px;
  width: 14px;
  height: 14px;
  border-radius: 50%;
  background: var(--text-muted);
  transform: translateY(-50%);
  transition: left 0.2s ease, width 0.2s ease, height 0.2s ease, background-color 0.2s ease;
}

.switch[aria-checked="true"] .switch__track {
  border-color: var(--accent);
  background: var(--accent);
}

/* The thumb grows when on, so the state does not depend on color alone. */
.switch[aria-checked="true"] .switch__thumb {
  left: 20px;
  width: 18px;
  height: 18px;
  background: var(--accent-contrast);
}

.switch:focus-visible {
  outline: none;
}

.switch:focus-visible .switch__track {
  outline: 2px solid var(--accent);
  outline-offset: 3px;
}

.switch:disabled {
  cursor: default;
  opacity: 0.45;
}

@media (prefers-reduced-motion: reduce) {
  .switch__track,
  .switch__thumb {
    transition: none;
  }
}
</style>
