const express = require("express");
const { body, validationResult } = require("express-validator");
const { protect } = require("../middleware/auth.middleware");
const { requireLevel } = require("../middleware/rbac.middleware");
const { compareDocuments } = require("../services/ai.service");
const { getDocumentById } = require("../services/document.service");
const { success, error } = require("../utils/apiResponse");

const router = express.Router();

router.use(protect);

// Clients cannot compare — paralegal and above only
router.use(requireLevel("paralegal"));

router.post(
  "/",
  [
    body("documentIdA").notEmpty().withMessage("documentIdA is required"),
    body("documentIdB").notEmpty().withMessage("documentIdB is required"),
  ],
  async (req, res, next) => {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return error(res, errors.array()[0].msg, 422);
    }

    try {
      const { documentIdA, documentIdB } = req.body;

      if (documentIdA === documentIdB) {
        return error(res, "Cannot compare a document with itself", 400);
      }

      const [docA, docB] = await Promise.all([
        getDocumentById(documentIdA),
        getDocumentById(documentIdB),
      ]);

      if (!docA) return error(res, "First document not found", 404);
      if (!docB) return error(res, "Second document not found", 404);

      if (!docA.ingested || !docB.ingested) {
        return error(res, "Both documents must finish processing before comparison", 400);
      }

      const result = await compareDocuments({
        documentIdA,
        filenameA: docA.originalName,
        documentIdB,
        filenameB: docB.originalName,
      });

      return success(res, result);
    } catch (err) {
      next(err);
    }
  }
);

module.exports = router;