const express = require("express");
const { protect } = require("../middleware/auth.middleware");
const { restrictTo, ownerOrLawyer } = require("../middleware/rbac.middleware");
const { upload, getFileType } = require("../middleware/upload.middleware");
const {
  createDocument,
  getAllDocuments,
  getDocumentById,
  deleteDocument,
  ingestAndAnalyze,
} = require("../services/document.service");
const { success, created, error } = require("../utils/apiResponse");

const router = express.Router();

router.use(protect);

router.post(
  "/upload",
  upload.single("file"),
  async (req, res, next) => {
    try {
      if (!req.file) {
        return error(res, "No file uploaded", 400);
      }

      const fileType = getFileType(req.file.mimetype);

      const doc = await createDocument({
        filename:      req.file.filename,
        originalName:  req.file.originalname,
        filePath:      req.file.path,
        fileType,
        fileSizeBytes: req.file.size,
        uploadedBy:    req.user._id,
      });

      // Fire and forget — runs ingest + analysis in background
      ingestAndAnalyze(doc, req.user._id.toString()).catch(() => {});

      return created(res, { document: doc }, "File uploaded and processing started");
    } catch (err) {
      next(err);
    }
  }
);

router.get("/", async (req, res, next) => {
  try {
    const docs = await getAllDocuments(req.user._id, req.user.role);
    return success(res, { documents: docs, total: docs.length });
  } catch (err) {
    next(err);
  }
});

router.get("/:id", async (req, res, next) => {
  try {
    const doc = await getDocumentById(req.params.id);
    if (!doc) return error(res, "Document not found", 404);
    return success(res, { document: doc });
  } catch (err) {
    next(err);
  }
});

router.delete(
  "/:id",
  ownerOrLawyer(async (req) => {
    const doc = await getDocumentById(req.params.id);
    return doc?.uploadedBy?._id || doc?.uploadedBy;
  }),
  async (req, res, next) => {
    try {
      const result = await deleteDocument(req.params.id);
      return success(res, result, "Document deleted");
    } catch (err) {
      next(err);
    }
  }
);

module.exports = router;