<script setup lang="ts">
import { onMounted, ref } from "vue";
import { useRouter } from "vue-router";
import { ApiError } from "../api";
import { loginFromFragment } from "../auth";

const router = useRouter();
const checking = ref(true);
const error = ref("");

function describeLoginError(cause: unknown): string {
  if (cause instanceof ApiError && [401, 403, 410].includes(cause.status)) {
    return "Ссылка входа недействительна или истекла.";
  }
  return "Не удалось войти. Проверьте ссылку и попробуйте ещё раз.";
}

onMounted(async () => {
  try {
    const state = await loginFromFragment();
    if (state?.authenticated) {
      await router.replace("/");
      return;
    }
    if (state !== null) {
      error.value = "Ссылка входа недействительна или истекла.";
    }
  } catch (cause) {
    error.value = describeLoginError(cause);
  } finally {
    checking.value = false;
  }
});
</script>

<template>
  <div class="login-container">
    <h1>Вход в Агента</h1>
    <p v-if="checking">Проверяем ссылку входа…</p>
    <template v-else>
      <p>
        Выполните <code class="auth-code">agent auth link</code> и откройте ссылку.
      </p>
      <p v-if="error" class="login-error">{{ error }}</p>
      <p class="login-muted">Ссылка одноразовая — после входа её не нужно хранить.</p>
    </template>
  </div>
</template>

<style scoped>
.login-container {
  max-width: 560px;
  margin: 15vh auto;
  padding: 24px;
}

h1 {
  font-size: 1.5rem;
  margin-bottom: 1rem;
}

.auth-code {
  background: var(--bg-subtle);
  padding: 2px 6px;
  border-radius: 4px;
  font-family: ui-monospace, monospace;
}

.login-error {
  background: color-mix(in srgb, var(--danger) 8%, transparent);
  color: var(--danger);
  border-radius: 8px;
  padding: 12px;
  margin: 16px 0;
}

.login-muted {
  color: var(--text-muted);
  font-size: 0.875rem;
}

@media (max-width: 480px) {
  .login-container {
    margin: 10vh 16px;
    padding: 16px;
  }
}

:focus-visible {
  outline: 2px solid var(--link);
  outline-offset: 2px;
}
</style>
