const express = require("express");
const { body, validationResult } = require("express-validator");
const { protect } = require("../middleware/auth.middleware");
const { ownerOrLawyer } = require("../middleware/rbac.middleware");
const {
  createReport,
  getAllReports,
  getReportById,
  deleteReport,
  runAndSaveReport,
} = require("../services/research.service");
const { success, created, error } = require("../utils/apiResponse");
const logger = require("../utils/logger");

const router = express.Router();

router.use(protect);

// Start a new research report — fires pipeline in background
router.post(
  "/",
  [body("topic").trim().notEmpty().withMessage("Research topic is required")],
  async (req, res, next) => {
    const errors = validationResult(req);
    if (!errors.isEmpty()) return error(res, errors.array()[0].msg, 422);

    try {
      const report = await createReport(req.body.topic, req.user._id);

      // Fire and forget
      runAndSaveReport(report._id.toString(), req.body.topic).catch((err) => {
        logger.error(
          `Background research pipeline failed for reportId=${report._id}: ${err.message}`
        );
      });

      return created(res, { report }, "Research started");
    } catch (err) {
      next(err);
    }
  }
);

// List all reports
router.get("/", async (req, res, next) => {
  try {
    const reports = await getAllReports(req.user._id, req.user.role);
    return success(res, { reports, total: reports.length });
  } catch (err) {
    next(err);
  }
});

// Get single report
router.get("/:id", async (req, res, next) => {
  try {
    const report = await getReportById(req.params.id);
    if (!report) return error(res, "Report not found", 404);
    return success(res, { report });
  } catch (err) {
    next(err);
  }
});

// Delete report — owner or lawyer only
router.delete(
  "/:id",
  ownerOrLawyer(async (req) => {
    const report = await getReportById(req.params.id);
    return report?.createdBy?._id || report?.createdBy;
  }),
  async (req, res, next) => {
    try {
      const result = await deleteReport(req.params.id);
      return success(res, result, "Report deleted");
    } catch (err) {
      next(err);
    }
  }
);

module.exports = router;
