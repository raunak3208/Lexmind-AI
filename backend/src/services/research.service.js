const axios = require("axios");
const ResearchReport = require("../models/ResearchReport");
const logger = require("../utils/logger");

const aiClient = axios.create({
  baseURL: process.env.AI_SERVICE_URL || "http://localhost:8000",
  timeout: 300000,
});

const createReport = async (topic, userId) => {
  return ResearchReport.create({ topic, createdBy: userId, status: "pending" });
};

const getAllReports = async (userId, role) => {
  const filter = role === "lawyer" ? {} : { createdBy: userId };
  return ResearchReport.find(filter)
    .populate("createdBy", "name email role")
    .sort({ createdAt: -1 });
};

const getReportById = async (reportId) => {
  return ResearchReport.findById(reportId).populate("createdBy", "name email role");
};

const deleteReport = async (reportId) => {
  await ResearchReport.findByIdAndDelete(reportId);
  return { deleted: true, reportId };
};

const runAndSaveReport = async (reportId, topic) => {
  try {
    const { data } = await aiClient.post("/research", { topic });
    const result = data.result;

    await ResearchReport.findByIdAndUpdate(reportId, {
      status:   "completed",
      title:    result.title,
      summary:  result.summary,
      findings: result.findings,
      analysis: result.analysis,
      sources:  result.sources,
      critic:   result.critic,
      meta: { wordCount: result.meta?.word_count },
    });

    logger.info(`Research report saved: reportId=${reportId}`);
  } catch (err) {
    logger.error(`Research pipeline failed: ${err.message}`);
    await ResearchReport.findByIdAndUpdate(reportId, {
      status: "failed",
      errorMessage: err.message,
    });
  }
};

module.exports = {
  createReport,
  getAllReports,
  getReportById,
  deleteReport,
  runAndSaveReport,
};