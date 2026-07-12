const express = require("express");
const { body } = require("express-validator");
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
const asyncHandler = require("../utils/asyncHandler");
const validate = require("../middleware/validate.middleware");

const router = express.Router();

router.use(protect);

// Start a new research report — fires pipeline in background
router.post(
  "/",
  [body("topic").trim().notEmpty().withMessage("Research topic is required")],
  validate,
  asyncHandler(async (req, res) => {
    const report = await createReport(req.body.topic, req.user._id);

    // Fire and forget
    runAndSaveReport(report._id.toString(), req.body.topic).catch((err) => {
      logger.error(
        `Background research pipeline failed for reportId=${report._id}: ${err.message}`
      );
    });

    return created(res, { report }, "Research started");
  })
);

// List all reports
router.get(
  "/",
  asyncHandler(async (req, res) => {
    const reports = await getAllReports(req.user._id, req.user.role);
    return success(res, { reports, total: reports.length });
  })
);

// Get single report
router.get(
  "/:id",
  asyncHandler(async (req, res) => {
    const report = await getReportById(req.params.id);
    if (!report) return error(res, "Report not found", 404);
    return success(res, { report });
  })
);

// Delete report — owner or lawyer only
router.delete(
  "/:id",
  ownerOrLawyer(async (req) => {
    const report = await getReportById(req.params.id);
    return report?.createdBy?._id || report?.createdBy;
  }),
  asyncHandler(async (req, res) => {
    const result = await deleteReport(req.params.id);
    return success(res, result, "Report deleted");
  })
);

module.exports = router;
