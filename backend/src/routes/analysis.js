const express = require("express");
const { protect } = require("../middleware/auth.middleware");
const { requireLevel } = require("../middleware/rbac.middleware");
const {
  getDocumentById,
  getAnalysis,
  ingestAndAnalyze,
} = require("../services/document.service");
const { success, error } = require("../utils/apiResponse");
const logger = require("../utils/logger");
const asyncHandler = require("../utils/asyncHandler");

const router = express.Router();

router.use(protect);

// Get analysis result for a document
router.get(
  "/:documentId",
  asyncHandler(async (req, res) => {
    const doc = await getDocumentById(req.params.documentId);
    if (!doc) return error(res, "Document not found", 404);

    const analysis = await getAnalysis(req.params.documentId);
    if (!analysis) {
      return error(res, "Analysis not found. Document may still be processing.", 404);
    }

    // Clients only see summary and risk score, not raw clauses
    if (req.user.role === "client") {
      return success(res, {
        status:      analysis.status,
        riskScore:   analysis.riskReport?.riskScore,
        overallRisk: analysis.riskReport?.overallRisk,
        summary:     analysis.summary,
      });
    }

    return success(res, { analysis });
  })
);

// Retry a failed analysis — paralegal and above only
router.post(
  "/:documentId/retry",
  requireLevel("paralegal"),
  asyncHandler(async (req, res) => {
    const doc = await getDocumentById(req.params.documentId);
    if (!doc) return error(res, "Document not found", 404);

    const analysis = await getAnalysis(req.params.documentId);
    if (analysis?.status === "completed") {
      return error(res, "Analysis already completed", 400);
    }

    ingestAndAnalyze(doc, req.user._id.toString()).catch((err) => {
      logger.error(
        `Background analysis retry failed for documentId=${req.params.documentId}: ${err.message}`
      );
    });

    return success(res, { documentId: req.params.documentId }, "Analysis restarted");
  })
);

module.exports = router;
