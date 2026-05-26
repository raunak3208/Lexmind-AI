const mongoose = require("mongoose");

const sourceSchema = new mongoose.Schema(
  { url: String, title: String },
  { _id: false }
);

const criticSchema = new mongoose.Schema(
  {
    score:   Number,
    verdict: String,
    review:  String,
  },
  { _id: false }
);

const researchReportSchema = new mongoose.Schema(
  {
    topic: {
      type: String,
      required: true,
      trim: true,
    },
    createdBy: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "User",
      required: true,
    },
    status: {
      type: String,
      enum: ["pending", "completed", "failed"],
      default: "pending",
    },
    title:    String,
    summary:  String,
    findings: [String],
    analysis: String,
    sources:  [sourceSchema],
    critic:   criticSchema,
    meta: {
      wordCount: Number,
    },
    errorMessage: {
      type: String,
      default: null,
    },
  },
  { timestamps: true }
);

module.exports = mongoose.model("ResearchReport", researchReportSchema);