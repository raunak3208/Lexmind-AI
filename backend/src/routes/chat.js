const express = require("express");
const { body } = require("express-validator");
const { protect } = require("../middleware/auth.middleware");
const { canAccessDocument } = require("../middleware/rbac.middleware");
const { chatWithDocument, clearChatHistory } = require("../services/ai.service");
const { getDocumentById } = require("../services/document.service");
const ChatSession = require("../models/ChatSession");
const { success, error } = require("../utils/apiResponse");
const asyncHandler = require("../utils/asyncHandler");
const validate = require("../middleware/validate.middleware");

const router = express.Router();

router.use(protect);

router.post(
  "/:documentId",
  [
    body("question").trim().notEmpty().withMessage("Question is required"),
  ],
  validate,
  asyncHandler(async (req, res) => {
    const { documentId } = req.params;
    const { question } = req.body;

    const doc = await getDocumentById(documentId);
    if (!doc) return error(res, "Document not found", 404);
    if (!canAccessDocument(req.user, doc)) {
      return error(res, "Access denied. You do not own this document.", 403);
    }

    if (!doc.ingested) {
      return error(res, "Document is still being processed. Try again shortly.", 400);
    }

    const userId = req.user._id.toString();
    const aiResult = await chatWithDocument({ documentId, userId, question });

    // Persist message to MongoDB
    await ChatSession.findOneAndUpdate(
      { documentId, userId: req.user._id },
      {
        $push: {
          messages: [
            { role: "user",      content: question },
            { role: "assistant", content: aiResult.answer },
          ],
        },
        $inc: { totalTurns: 1 },
      },
      { upsert: true, new: true }
    );

    return success(res, {
      answer:     aiResult.answer,
      turnNumber: aiResult.turn_number,
      documentId,
    });
  })
);

router.get(
  "/:documentId/history",
  asyncHandler(async (req, res) => {
    const session = await ChatSession.findOne({
      documentId: req.params.documentId,
      userId:     req.user._id,
    });

    return success(res, {
      messages:   session?.messages || [],
      totalTurns: session?.totalTurns || 0,
    });
  })
);

router.delete(
  "/:documentId/history",
  asyncHandler(async (req, res) => {
    const userId = req.user._id.toString();
    const { documentId } = req.params;

    await clearChatHistory({ documentId, userId });
    await ChatSession.findOneAndUpdate(
      { documentId, userId: req.user._id },
      { $set: { messages: [], totalTurns: 0 } }
    );

    return success(res, {}, "Chat history cleared");
  })
);

module.exports = router;
