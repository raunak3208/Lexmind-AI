const mongoose = require("mongoose");

const documentSchema = new mongoose.Schema(
  {
    filename: {
      type: String,
      required: true,
    },
    originalName: {
      type: String,
      required: true,
    },
    filePath: {
      type: String,
      required: true,
    },
    fileType: {
      type: String,
      enum: ["pdf", "docx", "txt"],
      required: true,
    },
    fileSizeBytes: {
      type: Number,
      required: true,
    },
    uploadedBy: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "User",
      required: true,
    },
    // Set after Python ingestion completes
    ingested: {
      type: Boolean,
      default: false,
    },
    ingestedAt: {
      type: Date,
      default: null,
    },
    totalChunks: {
      type: Number,
      default: 0,
    },
    // Set after analysis pipeline completes
    analysed: {
      type: Boolean,
      default: false,
    },
    analysedAt: {
      type: Date,
      default: null,
    },
    tags: [String],
  },
  { timestamps: true }
);

module.exports = mongoose.model("Document", documentSchema);