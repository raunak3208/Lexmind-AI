const mongoose = require("mongoose");

const riskFlagSchema = new mongoose.Schema(
  {
    flagId: String,
    clauseId: String,
    riskLevel: { type: String, enum: ["low", "medium", "high", "critical"] },
    category: String,
    description: String,
    suggestion: String,
    flaggedText: String,
  },
  { _id: false }
);

const clauseSchema = new mongoose.Schema(
  {
    clauseId: String,
    clauseType: String,
    heading: String,
    text: String,
    page: Number,
    section: String,
    partiesMentioned: [String],
  },
  { _id: false }
);

const analysisSchema = new mongoose.Schema(
  {
    documentId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "Document",
      required: true,
      unique: true,
    },
    status: {
      type: String,
      enum: ["pending", "completed", "failed"],
      default: "pending",
    },
    extraction: {
      totalClauses: Number,
      clauses: [clauseSchema],
    },
    riskReport: {
      overallRisk: { type: String, enum: ["low", "medium", "high", "critical"] },
      riskScore: { type: Number, min: 0, max: 100 },
      totalFlags: Number,
      flags: [riskFlagSchema],
      summary: String,
    },
    summary: {
      contractType: String,
      parties: [String],
      effectiveDate: String,
      expiryDate: String,
      governingLaw: String,
      keyObligations: [String],
      executiveSummary: String,
    },
    errorMessage: {
      type: String,
      default: null,
    },
  },
  { timestamps: true }
);

module.exports = mongoose.model("Analysis", analysisSchema);