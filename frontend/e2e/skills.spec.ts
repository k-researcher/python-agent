import { expect, test, type Page } from "@playwright/test";
import type { SkillCatalog, SkillItem } from "../src/api/skills";
import { expectNoHorizontalOverflow, openApp } from "./helpers";

// The skills page: switches act at once, dialogs keep the focus, and nothing overflows.

const skill = (name: string, extra: Partial<SkillItem> = {}): SkillItem => ({
  scope: "shared",
  name,
  description: `Когда нужно ${name}: длинное описание, которое объясняет агенту, когда брать этот навык и зачем.`,
  version: "",
  enabled: true,
  default_enabled: true,
  project_choice: null,
  replaced: false,
  files: 0,
  path: `/data/skills/${name}`,
  ...extra,
});

function catalog(): SkillCatalog {
  return {
    shared_dir: "/Users/someone/python-agent/data/skills",
    project_dir: "/Users/someone/work/app/.agent/skills",
    skills: [
      skill("code-review", { files: 3, version: "1.2" }),
      skill("release-notes", { default_enabled: false, enabled: false }),
      skill("code-review", { scope: "project", description: "Наш порядок ревью", files: 1 }),
    ],
    errors: [{ scope: "shared", message: "No closing header delimiter in /data/skills/broken" }],
  };
}

async function openSkills(page: Page): Promise<string[]> {
  const switches: string[] = [];
  await page.route("**/api/skills**", async (route) => {
    const request = route.request();
    if (request.method() === "PUT" && request.url().endsWith("/switch")) {
      switches.push(request.postData() ?? "");
    }
    if (request.method() === "GET" && /\/api\/skills\/(shared|project)\//.test(request.url())) {
      return route.fulfill({
        json: { ...skill("code-review"), instructions: "1. Прочитай дифф.", file_list: ["a.md"], revision: "a".repeat(64) },
      });
    }
    return route.fulfill({ json: catalog() });
  });
  await page.goto("/settings/skills");
  await expect(page.getByTestId("skills-settings")).toBeVisible();
  return switches;
}

test("skills page fits the screen in both tabs", async ({ page }, info) => {
  await openApp(page, info);
  await openSkills(page);
  await expect(page.locator("[data-skill]")).toHaveCount(2);
  await expectNoHorizontalOverflow(page);
  const shots = process.env.UI_E2E_SHOTS;
  if (shots) await page.screenshot({ path: `${shots}/${info.project.name}-skills-shared.png`, fullPage: true });

  await page.getByTestId("skills-tab-project").click();
  await expect(page.getByTestId("skills-project")).toBeVisible();
  await expect(page.locator("[data-skill]")).toHaveCount(3);
  await expectNoHorizontalOverflow(page);
  if (shots) await page.screenshot({ path: `${shots}/${info.project.name}-skills-project.png`, fullPage: true });
});

test("a switch works from the keyboard and acts at once", async ({ page }, info) => {
  await openApp(page, info);
  const switches = await openSkills(page);
  const toggle = page.locator('[data-skill="code-review"] [role="switch"]');
  await expect(toggle).toHaveAttribute("aria-checked", "true");
  await toggle.focus();
  await page.keyboard.press("Space");
  await expect.poll(() => switches.length).toBe(1);
  expect(JSON.parse(switches[0])).toEqual({ project_id: null, enabled: false });
  // The hit area is at least 44 px high.
  expect((await toggle.boundingBox())!.height).toBeGreaterThanOrEqual(44);
});

test("the editor keeps the focus inside and closes with Escape", async ({ page }, info) => {
  await openApp(page, info);
  await openSkills(page);
  await page.getByTestId("skills-create").click();
  const dialog = page.getByRole("dialog", { name: "Новый навык" });
  await expect(dialog).toBeVisible();
  await expect(page.getByTestId("skill-name")).toBeFocused();
  await expectNoHorizontalOverflow(page);
  const shots = process.env.UI_E2E_SHOTS;
  if (shots) await page.screenshot({ path: `${shots}/${info.project.name}-skills-editor.png`, animations: "disabled" });
  await page.keyboard.press("Escape");
  await expect(dialog).toBeHidden();
});

test("delete asks first with the safe choice focused", async ({ page }, info) => {
  await openApp(page, info);
  await openSkills(page);
  await page.locator('[data-skill="code-review"] [data-testid="skill-delete"]').click();
  const alert = page.getByRole("alertdialog", { name: "Удалить навык «code-review»?" });
  await expect(alert).toBeVisible();
  await expect(alert.getByRole("button", { name: "Отмена" })).toBeFocused();
  await expect(alert).toContainText("удалятся ещё 3 файла");
  const shots = process.env.UI_E2E_SHOTS;
  if (shots) await page.screenshot({ path: `${shots}/${info.project.name}-skills-delete.png`, animations: "disabled" });
});
