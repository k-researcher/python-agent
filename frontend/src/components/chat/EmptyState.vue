<script setup lang="ts">
// Empty landing state: a heading and four suggestion cards. Clicking a card
// emits the card's template text so the parent can drop it into the composer.

import { ref } from "vue";

const emit = defineEmits<{ (e: "pick", templateText: string): void }>();

interface SuggestionCard {
  title: string;
  description: string;
  template: string;
}

const cards: SuggestionCard[] = [
  {
    title: "Новая идея",
    description: "Сгенерируй свежую идею для проекта",
    template: "Предложи три интересные идеи для этого проекта.",
  },
  {
    title: "Составить план",
    description: "Разбей задачу на понятные шаги",
    template: "Составь пошаговый план реализации этой задачи.",
  },
  {
    title: "Разобрать код",
    description: "Объясни, что делает этот фрагмент",
    template: "Разбери код в текущем проекте и объясни, что он делает.",
  },
  {
    title: "Найти ошибку",
    description: "Помоги отладить проблему",
    template: "Помоги найти и исправить ошибку в текущем проекте.",
  },
];

function pick(card: SuggestionCard): void {
  emit("pick", card.template);
}
</script>

<template>
  <div class="empty-state" data-testid="empty-state">
    <h1 class="title">Что будем делать?</h1>
    <div class="grid">
      <button
        v-for="card in cards"
        :key="card.title"
        type="button"
        class="card"
        data-testid="suggestion-card"
        @click="pick(card)"
      >
        <span class="card-title">{{ card.title }}</span>
        <span class="card-desc">{{ card.description }}</span>
      </button>
    </div>
  </div>
</template>

<style scoped>
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1.2rem;
  padding: 2.5rem 1.5rem;
  text-align: center;
  width: 100%;
  box-sizing: border-box;
}

.title {
  font-size: 1.35rem;
  font-weight: 600;
  margin: 0;
  color: var(--text, #111114);
}

.grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0.75rem;
  width: 100%;
  max-width: 760px;
}

.card {
  display: flex;
  flex-direction: column;
  gap: 0.3rem;
  padding: 1rem 1.1rem;
  border: 1px solid var(--border, #e6e6e9);
  border-radius: var(--radius, 10px);
  background: var(--surface, #fff);
  text-align: left;
  cursor: pointer;
  transition: border-color 0.15s ease, box-shadow 0.15s ease;
}
.card:hover {
  border-color: var(--link, #2563eb);
  box-shadow: var(--shadow, 0 2px 8px rgba(0, 0, 0, 0.06));
}

.card-title {
  font-weight: 600;
  font-size: 0.92rem;
  color: var(--text, #111114);
}
.card-desc {
  font-size: 0.78rem;
  color: var(--text-muted, #6b6b76);
}

@media (max-width: 480px) {
  .grid {
    grid-template-columns: minmax(0, 1fr);
  }
}
</style>
