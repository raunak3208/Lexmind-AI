const { body, param } = require("express-validator");
const mongoose = require("mongoose");

const isValidObjectId = (value) => mongoose.Types.ObjectId.isValid(value);

const mongoId = (fieldName) =>
  param(fieldName)
    .custom(isValidObjectId)
    .withMessage(`${fieldName} must be a valid ID`);

const registerValidator = [
  body("name").trim().notEmpty().withMessage("Name is required"),
  body("email").isEmail().normalizeEmail().withMessage("Valid email required"),
  body("password")
    .isLength({ min: 6 })
    .withMessage("Password must be at least 6 characters"),
  body("role")
    .optional()
    .isIn(["lawyer", "paralegal", "client"])
    .withMessage("Role must be lawyer, paralegal or client"),
];

const loginValidator = [
  body("email").isEmail().normalizeEmail().withMessage("Valid email required"),
  body("password").notEmpty().withMessage("Password is required"),
];

const searchValidator = [
  body("query").trim().notEmpty().withMessage("Search query is required"),
  body("documentId")
    .optional()
    .custom(isValidObjectId)
    .withMessage("documentId must be a valid ID"),
  body("k")
    .optional()
    .isInt({ min: 1, max: 20 })
    .withMessage("k must be between 1 and 20"),
];

const chatValidator = [
  mongoId("documentId"),
  body("question").trim().notEmpty().withMessage("Question is required"),
];

const compareValidator = [
  body("documentIdA")
    .custom(isValidObjectId)
    .withMessage("documentIdA must be a valid ID"),
  body("documentIdB")
    .custom(isValidObjectId)
    .withMessage("documentIdB must be a valid ID"),
];

module.exports = {
  mongoId,
  registerValidator,
  loginValidator,
  searchValidator,
  chatValidator,
  compareValidator,
};