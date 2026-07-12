const jwt = require("jsonwebtoken");
const User = require("../models/User");

const signToken = (userId) => {
  return jwt.sign({ id: userId }, (process.env.JWT_SECRET || "").trim(), {
    expiresIn: process.env.JWT_EXPIRES_IN || "7d",
  });
};

const register = async ({ name, email, password }) => {
  const existing = await User.findOne({ email });
  if (existing) {
    const err = new Error("Email already registered");
    err.statusCode = 409;
    throw err;
  }

  // Never trust a client-supplied role during self-registration — that would
  // let anyone grant themselves "lawyer" privileges. New accounts always start
  // as "client"; privileged roles must be assigned by an administrator.
  const user = await User.create({ name, email, password, role: "client" });
  const token = signToken(user._id);

  return { user: user.toSafeObject(), token };
};

const login = async ({ email, password }) => {
  const user = await User.findOne({ email }).select("+password");

  if (!user || !(await user.comparePassword(password))) {
    const err = new Error("Invalid email or password");
    err.statusCode = 401;
    throw err;
  }

  if (!user.isActive) {
    const err = new Error("Account is deactivated");
    err.statusCode = 403;
    throw err;
  }

  const token = signToken(user._id);
  return { user: user.toSafeObject(), token };
};

const getMe = async (userId) => {
  const user = await User.findById(userId);
  if (!user) {
    const err = new Error("User not found");
    err.statusCode = 404;
    throw err;
  }
  return user.toSafeObject();
};

module.exports = { register, login, getMe };