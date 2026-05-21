const express = require("express");
const { body, validationResult } = require("express-validator");
const { protect } = require("../middleware/auth.middleware");
const { chatWithDocument, clearChatHistory } = require("../services/ai.service");
const { getDocumentById } = require("../services/document.service");
const ChatSession = require("../models/ChatSession");
const { success, error } = require("../utils/apiResponse");

const router = express.Router();

router.use(protect);

router.post(
  "/:documentId",
  [
    body("question").trim().notEmpty().withMessage("Question is required"),
  ],
  async (req, res, next) => {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return error(res, errors.array()[0].msg, 422);
    }

    try {
      const { documentId } = req.params;
      const { question } = req.body;
      const userId = req.user._id.toString();

      const doc = await getDocumentById(documentId);
      if (!doc) return error(res, "Document not found", 404);

      if (!doc.ingested) {
        return error(res, "Document is still being processed. Try again shortly.", 400);
      }

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
    } catch (err) {
      next(err);
    }
  }
);

router.get("/:documentId/history", async (req, res, next) => {
  try {
    const session = await ChatSession.findOne({
      documentId: req.params.documentId,
      userId:     req.user._id,
    });

    return success(res, {
      messages:   session?.messages || [],
      totalTurns: session?.totalTurns || 0,
    });
  } catch (err) {
    next(err);
  }
});

router.delete("/:documentId/history", async (req, res, next) => {
  try {
    const userId = req.user._id.toString();
    const { documentId } = req.params;

    await clearChatHistory({ documentId, userId });
    await ChatSession.findOneAndUpdate(
      { documentId, userId: req.user._id },
      { $set: { messages: [], totalTurns: 0 } }
    );

    return success(res, {}, "Chat history cleared");
  } catch (err) {
    next(err);
  }
});

module.exports = router;