const ROLES = {
  LAWYER:    "lawyer",
  PARALEGAL: "paralegal",
  CLIENT:    "client",
};

// What each role can do
const PERMISSIONS = {
  lawyer: [
    "document:read:any",
    "document:delete:any",
    "document:upload",
    "analysis:read:full",
    "analysis:retry",
    "search:any",
    "chat:any",
    "compare:any",
  ],
  paralegal: [
    "document:read:own",
    "document:upload",
    "document:delete:own",
    "analysis:read:full",
    "analysis:retry",
    "search:any",
    "chat:own",
    "compare:any",
  ],
  client: [
    "document:read:own",
    "document:upload",
    "analysis:read:summary",
    "search:any",
    "chat:own",
  ],
};

const hasPermission = (role, permission) => {
  return PERMISSIONS[role]?.includes(permission) ?? false;
};

const ROLE_HIERARCHY = [ROLES.CLIENT, ROLES.PARALEGAL, ROLES.LAWYER];

const getRoleLevel = (role) => ROLE_HIERARCHY.indexOf(role);

const isAtLeast = (userRole, minimumRole) => {
  return getRoleLevel(userRole) >= getRoleLevel(minimumRole);
};

module.exports = { ROLES, PERMISSIONS, hasPermission, ROLE_HIERARCHY, getRoleLevel, isAtLeast };