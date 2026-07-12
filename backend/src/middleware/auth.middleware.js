const jwt = require("jsonwebtoken");
const User = require("../models/User");
const { error } = require("../utils/apiResponse");

const protect = async (req, res, next) => {
  try {
    const authHeader = req.headers.authorization;

    if (!authHeader || !authHeader.startsWith("Bearer ")) {
      return error(res, "No token provided", 401);
    }

    const token = authHeader.split(" ")[1];

    let decoded;
    try {
      decoded = jwt.verify(token, (process.env.JWT_SECRET || "").trim());
    } catch (err) {
      if (err.name === "TokenExpiredError") {
        return error(res, "Token expired", 401);
      }
      return error(res, "Invalid token", 401);
    }

    const user = await User.findById(decoded.id).select("-password");
    if (!user) {
      return error(res, "User no longer exists", 401);
    }

    if (!user.isActive) {
      return error(res, "Account is deactivated", 403);
    }

    req.user = user;
    next();
  } catch (err) {
    return error(res, "Authentication failed", 500);
  }
};

module.exports = { protect };