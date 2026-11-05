import { describe, expect, it } from "vitest";
import { plural, reasoningStatsParts } from "./reasoning";

describe("reasoning stats label", () => {
  it("uses the Russian plural forms", () => {
    const forms = (count: number) => plural(count, "токен", "токена", "токенов");
    expect([1, 2, 5, 11, 12, 14, 21, 22, 25, 101, 111].map(forms)).toEqual([
      "токен", "токена", "токенов", "токенов", "токенов", "токенов", "токен", "токена", "токенов", "токен", "токенов",
    ]);
  });

  it("shows only what is known", () => {
    expect(reasoningStatsParts(5, null)).toEqual(["5 символов"]);
    expect(reasoningStatsParts(5, { tokens: null, exact: false, seconds: 3 })).toEqual(["5 символов"]);
    expect(reasoningStatsParts(5, { tokens: 3, exact: true, seconds: null })).toEqual(["5 символов", "3 токена"]);
  });

  it("gives a slow speed with one decimal and a fast one as a whole number", () => {
    expect(reasoningStatsParts(90, { tokens: 23, exact: true, seconds: 4 })[2]).toBe("5,8 ток/с");
    expect(reasoningStatsParts(90, { tokens: 230, exact: false, seconds: 4 })[2]).toBe("58 ток/с");
  });
});
