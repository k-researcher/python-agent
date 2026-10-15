import { test, expect } from "@playwright/test";
import { openApp, ensureLeftPanelOpen } from "./helpers";

test.describe("sidebar footer and project form containment", () => {
  test("sidebar footer elements align and buttons stay inside the panel", async ({ page }, testInfo) => {
    await openApp(page, testInfo);
    await ensureLeftPanelOpen(page);

    const sidebarPanel = page.locator(".app-shell__left");
    const sidebarFooter = page.locator(".left-sidebar__footer");
    const signoutButton = page.locator("[data-signout]");
    const signoutIcon = signoutButton.locator("svg");
    const signoutLabel = signoutButton.locator("span");

    const sidebarBox = await sidebarPanel.boundingBox();
    const footerBox = await sidebarFooter.boundingBox();
    expect(sidebarBox, "sidebar panel bounding box should be available").not.toBeNull();
    expect(footerBox, "footer bounding box should be available").not.toBeNull();
    const panelGeometry = sidebarBox!;
    const footerGeometry = footerBox!;
    expect(footerGeometry.x).toBeGreaterThanOrEqual(panelGeometry.x - 1);
    expect(footerGeometry.x + footerGeometry.width).toBeLessThanOrEqual(
      panelGeometry.x + panelGeometry.width + 1,
    );

    await expect(signoutButton).toBeVisible();
    const iconBox = await signoutIcon.boundingBox();
    const labelBox = await signoutLabel.boundingBox();
    expect(iconBox, "signout icon bounding box should be available").not.toBeNull();
    expect(labelBox, "signout label bounding box should be available").not.toBeNull();
    const iconGeometry = iconBox!;
    const labelGeometry = labelBox!;
    const iconCenterY = iconGeometry.y + iconGeometry.height / 2;
    const labelCenterY = labelGeometry.y + labelGeometry.height / 2;
    expect(Math.abs(iconCenterY - labelCenterY)).toBeLessThan(5);

    const footerButtons = sidebarFooter.locator("button");
    const buttonCount = await footerButtons.count();
    expect(buttonCount, "footer should contain both actions").toBe(2);
    for (let index = 0; index < buttonCount; index++) {
      const buttonBox = await footerButtons.nth(index).boundingBox();
      expect(
        buttonBox,
        `footer button at index ${index} should have a bounding box`,
      ).not.toBeNull();
      const buttonGeometry = buttonBox!;
      expect(buttonGeometry.x).toBeGreaterThanOrEqual(panelGeometry.x - 1);
      expect(buttonGeometry.x + buttonGeometry.width).toBeLessThanOrEqual(
        panelGeometry.x + panelGeometry.width + 1,
      );
      expect(buttonGeometry.y).toBeGreaterThanOrEqual(panelGeometry.y - 1);
      expect(buttonGeometry.y + buttonGeometry.height).toBeLessThanOrEqual(
        panelGeometry.y + panelGeometry.height + 1,
      );
    }
  });

  test("project creation form stays inside the panel and exposes its inputs", async ({ page }, testInfo) => {
    await openApp(page, testInfo);
    await ensureLeftPanelOpen(page);

    const sidebarPanel = page.locator(".app-shell__left");
    const newProjectButton = page.locator("[data-new-project]");
    await expect(newProjectButton).toBeVisible();
    await newProjectButton.click();

    const projectForm = page.locator("[data-create-form]");
    await expect(projectForm).toBeVisible();

    const panelBox = await sidebarPanel.boundingBox();
    const formBox = await projectForm.boundingBox();
    expect(panelBox, "sidebar panel bounding box should be available").not.toBeNull();
    expect(formBox, "project form bounding box should be available").not.toBeNull();
    const panelGeometry = panelBox!;
    const formGeometry = formBox!;
    expect(formGeometry.x).toBeGreaterThanOrEqual(panelGeometry.x - 1);
    expect(formGeometry.x + formGeometry.width).toBeLessThanOrEqual(
      panelGeometry.x + panelGeometry.width + 1,
    );

    await expect(projectForm.locator("[data-create-name]")).toBeVisible();
    await expect(projectForm.locator("[data-create-path]")).toBeVisible();
  });

  test("session rows align labels, status dots and chevrons", async ({ page }, testInfo) => {
    await openApp(page, testInfo);
    await ensureLeftPanelOpen(page);
    const sessionRow = page.locator(".tree-row").first();
    await expect(sessionRow).toBeVisible();
    await expect(sessionRow).toHaveCSS("display", "flex");
    await expect(sessionRow).toHaveCSS("align-items", "center");
    const statusDot = sessionRow.locator("[data-status-dot]");
    const label = sessionRow.locator(".tree-row__label");
    const statusBox = await statusDot.boundingBox();
    const labelBox = await label.boundingBox();
    expect(statusBox).not.toBeNull();
    expect(labelBox).not.toBeNull();
    expect(statusBox!.width).toBeGreaterThan(0);
    expect(statusBox!.height).toBeGreaterThan(0);
    expect(Math.abs(
      statusBox!.y + statusBox!.height / 2 - labelBox!.y - labelBox!.height / 2,
    )).toBeLessThan(2);
  });
});
