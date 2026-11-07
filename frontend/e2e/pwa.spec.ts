import { expect, test } from "@playwright/test";
import { openApp, themeOf } from "./helpers";

// The installed app: a valid manifest, a worker that keeps only the app shell, and a start
// without the network.

test.use({ serviceWorkers: "allow" });

test("the manifest describes an installable app", async ({ page, request }, info) => {
  await openApp(page, info);
  const href = await page.locator('link[rel="manifest"]').getAttribute("href");
  const manifest = await (await request.get(href!)).json();
  expect(manifest).toMatchObject({ start_url: "/", scope: "/", display: "standalone", short_name: "Агент" });
  const sizes = new Map<string, string>();
  for (const icon of manifest.icons) {
    const response = await request.get(icon.src);
    expect(response.ok(), icon.src).toBe(true);
    if (icon.type === "image/png") {
      // The PNG header keeps the width and the height at bytes 16–23.
      const body = await response.body();
      sizes.set(`${icon.purpose ?? "any"} ${body.readUInt32BE(16)}x${body.readUInt32BE(20)}`, icon.sizes);
    }
  }
  expect(sizes).toEqual(
    new Map([
      ["any 192x192", "192x192"],
      ["any 512x512", "512x512"],
      ["maskable 512x512", "512x512"],
    ]),
  );
  const touch = await page.locator('link[rel="apple-touch-icon"]').getAttribute("href");
  expect((await request.get(touch!)).ok()).toBe(true);
});

test("the worker keeps the app shell, never the API, and the app starts without the network", async ({ page, context }, info) => {
  await openApp(page, info);
  await page.evaluate(() => navigator.serviceWorker.ready);
  // The first load came before the worker; this one goes through it.
  await page.reload();
  await expect.poll(() => page.evaluate(() => !!navigator.serviceWorker.controller)).toBe(true);
  await expect(page.locator("[data-theme-toggle]")).toBeVisible();

  const saved = await page.evaluate(async () => {
    const urls: string[] = [];
    for (const name of await caches.keys()) {
      for (const request of await (await caches.open(name)).keys()) urls.push(new URL(request.url).pathname);
    }
    return urls;
  });
  expect(saved).toContain("/");
  expect(saved.some((path) => path.startsWith("/assets/"))).toBe(true);
  expect(saved.filter((path) => !(path === "/" || path.startsWith("/assets/")))).toEqual([]);

  await context.setOffline(true);
  await page.reload();
  await expect(page.locator("#app")).not.toBeEmpty();
  await context.setOffline(false);
});

test("the window color follows the theme switch", async ({ page }, info) => {
  await openApp(page, info);
  const color = () =>
    page.evaluate(() => {
      const metas = document.querySelectorAll<HTMLMetaElement>('meta[name="theme-color"]');
      const background = getComputedStyle(document.documentElement).getPropertyValue("--bg").trim();
      return { count: metas.length, content: metas[0]?.content, background };
    });
  const before = await color();
  expect(before.count).toBe(1);
  expect(before.content).toBe(before.background);
  await page.locator("[data-theme-toggle]").click();
  const after = await color();
  expect(after.content).toBe(after.background);
  expect(after.content).not.toBe(before.content);
  expect(themeOf(info)).toMatch(/light|dark/);
});
