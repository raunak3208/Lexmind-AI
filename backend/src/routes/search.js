const express = require("express");
const { body, query, validationResult } = require("express-validator");
const { protect } = require("../middleware/auth.middleware");
const { searchDocuments } = require("../services/ai.service");
const { success, error } = require("../utils/apiResponse");

const router = express.Router();

router.use(protect);

router.post(
  "/",
  [
    body("query").trim().notEmpty().withMessage("Search query is required"),
    body("documentId").optional().isString(),
    body("k").optional().isInt({ min: 1, max: 20 }).withMessage("k must be between 1 and 20"),
  ],
  async (req, res, next) => {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return error(res, errors.array()[0].msg, 422);
    }

    try {
      const { query: searchQuery, documentId = null, k = 5 } = req.body;

      const result = await searchDocuments({ query: searchQuery, documentId, k });
      return success(res, result);
    } catch (err) {
      next(err);
    }
  }
);

module.exports = router;