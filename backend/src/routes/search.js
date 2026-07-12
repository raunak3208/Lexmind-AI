const express = require("express");
const { body, query } = require("express-validator");
const { protect } = require("../middleware/auth.middleware");
const { searchDocuments } = require("../services/ai.service");
const { success } = require("../utils/apiResponse");
const asyncHandler = require("../utils/asyncHandler");
const validate = require("../middleware/validate.middleware");

const router = express.Router();

router.use(protect);

router.post(
  "/",
  [
    body("query").trim().notEmpty().withMessage("Search query is required"),
    body("documentId").optional().isString(),
    body("k").optional().isInt({ min: 1, max: 20 }).withMessage("k must be between 1 and 20"),
  ],
  validate,
  asyncHandler(async (req, res) => {
    const { query: searchQuery, documentId = null, k = 5 } = req.body;

    const result = await searchDocuments({ query: searchQuery, documentId, k });
    return success(res, result);
  })
);

module.exports = router;
