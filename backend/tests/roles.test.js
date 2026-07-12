const {
  ROLES,
  ROLE_HIERARCHY,
  getRoleLevel,
  hasPermission,
  isAtLeast,
} = require("../src/config/roles");

describe("role configuration", () => {
  test("defines roles in ascending permission order", () => {
    expect(ROLE_HIERARCHY).toEqual([
      ROLES.CLIENT,
      ROLES.PARALEGAL,
      ROLES.LAWYER,
    ]);
    expect(getRoleLevel(ROLES.CLIENT)).toBe(0);
    expect(getRoleLevel(ROLES.LAWYER)).toBe(2);
  });

  test("checks configured permissions", () => {
    expect(hasPermission(ROLES.LAWYER, "document:read:any")).toBe(true);
    expect(hasPermission(ROLES.CLIENT, "document:delete:any")).toBe(false);
    expect(hasPermission("unknown", "document:upload")).toBe(false);
  });

  test("compares valid role levels", () => {
    expect(isAtLeast(ROLES.LAWYER, ROLES.PARALEGAL)).toBe(true);
    expect(isAtLeast(ROLES.PARALEGAL, ROLES.PARALEGAL)).toBe(true);
    expect(isAtLeast(ROLES.CLIENT, ROLES.PARALEGAL)).toBe(false);
  });
});
