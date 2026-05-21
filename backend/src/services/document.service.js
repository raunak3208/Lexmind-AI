const Document = require("../models/Document");
const Analysis = require("../models/Analysis");
const { deleteFile } = require("./storage.service");
const aiService = require("./ai.service");
const logger = require("../utils/logger");

const createDocument = async ({ filename, originalName, filePath, fileType, fileSizeBytes, uploadedBy }) => {
  return Document.create({
    filename,
    originalName,
    filePath,
    fileType,
    fileSizeBytes,
    uploadedBy,
  });
};

const getAllDocuments = async (userId, role) => {
  const filter = role === "lawyer" ? {} : { uploadedBy: userId };
  return Document.find(filter)
    .populate("uploadedBy", "name email role")
    .sort({ createdAt: -1 });
};

const getDocumentById = async (documentId) => {
  return Document.findById(documentId).populate("uploadedBy", "name email role");
};

const deleteDocument = async (documentId) => {
  const doc = await Document.findById(documentId);
  if (!doc) {
    const err = new Error("Document not found");
    err.statusCode = 404;
    throw err;
  }

  deleteFile(doc.filePath);
  await Analysis.deleteOne({ documentId });
  await Document.findByIdAndDelete(documentId);

  return { deleted: true, documentId };
};

const markIngested = async (documentId, totalChunks) => {
  return Document.findByIdAndUpdate(
    documentId,
    { ingested: true, ingestedAt: new Date(), totalChunks },
    { new: true }
  );
};

const markAnalysed = async (documentId) => {
  return Document.findByIdAndUpdate(
    documentId,
    { analysed: true, analysedAt: new Date() },
    { new: true }
  );
};

const saveAnalysis = async (documentId, pipelineResult) => {
  const payload = {
    documentId,
    status: "completed",
    extraction: {
      totalClauses: pipelineResult.extraction.total_clauses,
      clauses: pipelineResult.extraction.clauses.map((c) => ({
        clauseId: c.clause_id,
        clauseType: c.clause_type,
        heading: c.heading,
        text: c.text,
        page: c.page,
        section: c.section,
        partiesMentioned: c.parties_mentioned,
      })),
    },
    riskReport: {
      overallRisk: pipelineResult.risk_report.overall_risk,
      riskScore: pipelineResult.risk_report.risk_score,
      totalFlags: pipelineResult.risk_report.total_flags,
      flags: pipelineResult.risk_report.flags.map((f) => ({
        flagId: f.flag_id,
        clauseId: f.clause_id,
        riskLevel: f.risk_level,
        category: f.category,
        description: f.description,
        suggestion: f.suggestion,
        flaggedText: f.flagged_text,
      })),
      summary: pipelineResult.risk_report.summary,
    },
    summary: {
      contractType: pipelineResult.summary.contract_type,
      parties: pipelineResult.summary.parties,
      effectiveDate: pipelineResult.summary.effective_date,
      expiryDate: pipelineResult.summary.expiry_date,
      governingLaw: pipelineResult.summary.governing_law,
      keyObligations: pipelineResult.summary.key_obligations,
      executiveSummary: pipelineResult.summary.executive_summary,
    },
  };

  return Analysis.findOneAndUpdate(
    { documentId },
    payload,
    { upsert: true, new: true }
  );
};

const saveFailedAnalysis = async (documentId, errorMessage) => {
  return Analysis.findOneAndUpdate(
    { documentId },
    { documentId, status: "failed", errorMessage },
    { upsert: true, new: true }
  );
};

const getAnalysis = async (documentId) => {
  return Analysis.findOne({ documentId });
};

const ingestAndAnalyze = async (doc, uploadedByStr) => {
  logger.info(`Starting ingest+analyze for documentId=${doc._id}`);

  try {
    const ingestResult = await aiService.ingestDocument({
      filePath: doc.filePath,
      documentId: doc._id.toString(),
      filename: doc.originalName,
      uploadedBy: uploadedByStr,
    });

    await markIngested(doc._id, ingestResult.total_chunks);
    logger.info(`Ingested documentId=${doc._id} chunks=${ingestResult.total_chunks}`);

    const analysisResult = await aiService.analyzeDocument({
      filePath: doc.filePath,
      documentId: doc._id.toString(),
      filename: doc.originalName,
    });

    await saveAnalysis(doc._id, analysisResult);
    await markAnalysed(doc._id);
    logger.info(`Analysis complete for documentId=${doc._id}`);
  } catch (err) {
    logger.error(`Pipeline failed for documentId=${doc._id}: ${err.message}`);
    await saveFailedAnalysis(doc._id, err.message);
  }
};

module.exports = {
  createDocument,
  getAllDocuments,
  getDocumentById,
  deleteDocument,
  markIngested,
  markAnalysed,
  saveAnalysis,
  saveFailedAnalysis,
  getAnalysis,
  ingestAndAnalyze,
};