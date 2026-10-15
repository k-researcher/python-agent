import { mount } from "@vue/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { nextTick } from "vue";
import ApprovalInline from "./ApprovalInline.vue";

function mountApproval(props: Record<string, unknown> = {}) {
  return mount(ApprovalInline, {
    props: { approvalId: "ap1", riskLevel: "high", hotkeys: false, ...props },
  });
}

function fireWindowKey(key: string, target?: EventTarget) {
  const event = new KeyboardEvent("keydown", { key, bubbles: true, cancelable: true });
  Object.defineProperty(event, "target", { value: target ?? document.body, configurable: true });
  window.dispatchEvent(event);
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("ApprovalInline", () => {
  it("shows a localized risk label", () => {
    const wrapper = mountApproval({ riskLevel: "high" });
    expect(wrapper.get('[data-testid="risk-label"]').text()).toBe("Высокий риск");
    expect(wrapper.get('[data-testid="risk-label"]').attributes("data-risk")).toBe("high");
  });

  it("passes through unknown risk levels", () => {
    const wrapper = mountApproval({ riskLevel: "custom-risk" });
    expect(wrapper.get('[data-testid="risk-label"]').text()).toBe("custom-risk");
  });

  it("emits approve with the approval id", async () => {
    const wrapper = mountApproval();
    await wrapper.get('[data-testid="approve-btn"]').trigger("click");
    expect(wrapper.emitted("approve")?.[0]).toEqual(["ap1"]);
  });

  it("reject opens the comment field; submitting emits reject with the comment", async () => {
    const wrapper = mountApproval();
    await wrapper.get('[data-testid="reject-btn"]').trigger("click");
    expect(wrapper.find('[data-testid="comment-field"]').exists()).toBe(true);
    await wrapper.get('[data-testid="reject-comment"]').setValue("выглядит опасно");
    await wrapper.get('[data-testid="reject-submit"]').trigger("click");
    expect(wrapper.emitted("reject")?.[0]).toEqual([{ approvalId: "ap1", comment: "выглядит опасно" }]);
  });

  it("submitting with an empty comment still emits reject with an empty comment", async () => {
    const wrapper = mountApproval();
    await wrapper.get('[data-testid="reject-btn"]').trigger("click");
    await wrapper.get('[data-testid="reject-submit"]').trigger("click");
    expect(wrapper.emitted("reject")?.[0]).toEqual([{ approvalId: "ap1", comment: "" }]);
  });

  it("does not emit approve on the A hotkey when hotkeys are disabled", () => {
    const wrapper = mountApproval({ hotkeys: false });
    fireWindowKey("a");
    expect(wrapper.emitted("approve")).toBeUndefined();
  });

  it("emits approve on the A hotkey when hotkeys are enabled", () => {
    const wrapper = mountApproval({ hotkeys: true });
    fireWindowKey("a");
    expect(wrapper.emitted("approve")?.[0]).toEqual(["ap1"]);
  });

  it("R hotkey opens the reject comment field when hotkeys are enabled", async () => {
    const wrapper = mountApproval({ hotkeys: true });
    expect(wrapper.find('[data-testid="comment-field"]').exists()).toBe(false);
    fireWindowKey("r");
    await nextTick();
    expect(wrapper.find('[data-testid="comment-field"]').exists()).toBe(true);
  });

  it("does not react to A/R hotkeys while focus is inside a textarea", async () => {
    const wrapper = mountApproval({ hotkeys: true });
    const input = document.createElement("textarea");
    document.body.appendChild(input);
    input.focus();
    expect(document.activeElement).toBe(input);
    try {
      fireWindowKey("a", input);
      fireWindowKey("r", input);
      expect(wrapper.emitted("approve")).toBeUndefined();
      expect(wrapper.find('[data-testid="comment-field"]').exists()).toBe(false);
    } finally {
      document.body.removeChild(input);
    }
  });

  it("does not react to A/R hotkeys while focus is inside a regular input", () => {
    const wrapper = mountApproval({ hotkeys: true });
    const input = document.createElement("input");
    document.body.appendChild(input);
    input.focus();
    try {
      fireWindowKey("a", input);
      expect(wrapper.emitted("approve")).toBeUndefined();
    } finally {
      document.body.removeChild(input);
    }
  });

  it("removes its keydown listener on unmount", async () => {
    const wrapper = mountApproval({ hotkeys: true });
    const keydown = vi.spyOn(window, "removeEventListener");
    wrapper.unmount();
    expect(keydown).toHaveBeenCalledWith("keydown", expect.any(Function));
  });
});
