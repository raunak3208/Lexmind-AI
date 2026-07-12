const express = require("express");
const { body } = require("express-validator");
const { register, login, getMe } = require("../services/user.service");
const { protect } = require("../middleware/auth.middleware");
const { success, created } = require("../utils/apiResponse");
const asyncHandler = require("../utils/asyncHandler");
const validate = require("../middleware/validate.middleware");

const router = express.Router();

router.post(
  "/register",
  [
    body("name").trim().notEmpty().withMessage("Name is required"),
    body("email").isEmail().withMessage("Valid email is required"),
    body("password")
      .isLength({ min: 6 })
      .withMessage("Password must be at least 6 characters"),
    // Note: `role` is intentionally NOT accepted here. Self-registered users
    // always default to "client"; privileged roles are assigned by an admin.
  ],
  validate,
  asyncHandler(async (req, res) => {
    const result = await register(req.body);
    return created(res, result, "Account created");
  })
);

router.post(
  "/login",
  [
    body("email").isEmail().withMessage("Valid email is required"),
    body("password").notEmpty().withMessage("Password is required"),
  ],
  validate,
  asyncHandler(async (req, res) => {
    const result = await login(req.body);
    return success(res, result, "Login successful");
  })
);

router.get(
  "/me",
  protect,
  asyncHandler(async (req, res) => {
    const user = await getMe(req.user._id);
    return success(res, { user });
  })
);

module.exports = router;
