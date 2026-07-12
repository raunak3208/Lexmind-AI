const { error } = require("../utils/apiResponse");

// Role hierarchy — higher index = more permissions
const ROLE_HIERARCHY = ["client", "paralegal", "lawyer"];

// Restrict route to specific roles only
const restrictTo = (...roles) => {
  return (req, res, next) => {
    if (!req.user) {
      return error(res, "Not authenticated", 401);
    }

    if (!roles.includes(req.user.role)) {
      return error(
        res,
        `Access denied. Required role: ${roles.join(" or ")}`,
        403
      );
    }

    next();
  };
};

// Restrict to roles at or above a minimum level in the hierarchy
const requireLevel = (minimumRole) => {
  return (req, res, next) => {
    if (!req.user) {
      return error(res, "Not authenticated", 401);
    }

    const userLevel = ROLE_HIERARCHY.indexOf(req.user.role);
    const requiredLevel = ROLE_HIERARCHY.indexOf(minimumRole);

    if (requiredLevel === -1) {
      return error(res, "Invalid role configuration", 500);
    }

    if (userLevel < requiredLevel) {
      return error(
        res,
        `Access denied. Minimum role required: ${minimumRole}`,
        403
      );
    }

    next();
  };
};

// Ensure user can only access their own resources unless they are a lawyer
const ownerOrLawyer = (getResourceUserId) => {
  return async (req, res, next) => {
    try {
      if (!req.user) {
        return error(res, "Not authenticated", 401);
      }

      if (req.user.role === "lawyer") {
        return next();
      }

      const resourceUserId = await getResourceUserId(req);

      if (!resourceUserId) {
        return error(res, "Resource not found", 404);
      }

      if (resourceUserId.toString() !== req.user._id.toString()) {
        return error(res, "Access denied. You do not own this resource.", 403);
      }

      next();
    } catch (err) {
      return error(res, "Authorization check failed", 500);
    }
  };
};

// Whether a user may access a document. Lawyers can access any document;
// everyone else may only access documents they uploaded.
const canAccessDocument = (user, doc) => {
  if (!user || !doc) return false;
  if (user.role === "lawyer") return true;
  const ownerId = doc.uploadedBy?._id || doc.uploadedBy;
  return !!ownerId && ownerId.toString() === user._id.toString();
};

module.exports = { restrictTo, requireLevel, ownerOrLawyer, canAccessDocument };