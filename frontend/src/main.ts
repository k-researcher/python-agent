import { createApp } from "vue";
import { createPinia } from "pinia";
import App from "./App.vue";
import { router } from "./router";
import { registerServiceWorker } from "./pwa";
import "./styles/tokens.css";
import "./styles/base.css";
import "./styles/code.css";
import "./styles/components.css";

const app = createApp(App);
app.use(createPinia());
app.use(router);
app.mount("#app");
registerServiceWorker();
