const express = require("express");
const { body, validationResult } = require("express-validator");
const { register, login, getMe } = require("../services/user.service");
const { protect } = require("../middleware/auth.middleware");
const { success, created, error } = require("../utils/apiResponse");

const router = express.Router();

const handleValidation = (req, res) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    error(res, errors.array()[0].msg, 422);
    return false;
  }
  return true;
};

router.post(
  "/register",
  [
    body("name").trim().notEmpty().withMessage("Name is required"),
    body("email").isEmail().withMessage("Valid email is required"),
    body("password")
      .isLength({ min: 6 })
      .withMessage("Password must be at least 6 characters"),
    body("role")
      .optional()
      .isIn(["lawyer", "paralegal", "client"])
      .withMessage("Role must be lawyer, paralegal or client"),
  ],
  async (req, res, next) => {
    if (!handleValidation(req, res)) return;
    try {
      const result = await register(req.body);
      return created(res, result, "Account created");
    } catch (err) {
      next(err);
    }
  }
);

router.post(
  "/login",
  [
    body("email").isEmail().withMessage("Valid email is required"),
    body("password").notEmpty().withMessage("Password is required"),
  ],
  async (req, res, next) => {
    if (!handleValidation(req, res)) return;
    try {
      const result = await login(req.body);
      return success(res, result, "Login successful");
    } catch (err) {
      next(err);
    }
  }
);

router.get("/me", protect, async (req, res, next) => {
  try {
    const user = await getMe(req.user._id);
    return success(res, { user });
  } catch (err) {
    next(err);
  }
});

module.exports = router;