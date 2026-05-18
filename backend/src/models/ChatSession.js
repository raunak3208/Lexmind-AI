const mongoose = require("mongoose");

const messageSchema = new mongoose.Schema(
  {
    role: { type: String, enum: ["user", "assistant"], required: true },
    content: { type: String, required: true },
    createdAt: { type: Date, default: Date.now },
  },
  { _id: false }
);

const chatSessionSchema = new mongoose.Schema(
  {
    documentId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "Document",
      required: true,
    },
    userId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "User",
      required: true,
    },
    messages: [messageSchema],
    totalTurns: {
      type: Number,
      default: 0,
    },
  },
  { timestamps: true }
);

// One session per user per document
chatSessionSchema.index({ documentId: 1, userId: 1 }, { unique: true });

module.exports = mongoose.model("ChatSession", chatSessionSchema);