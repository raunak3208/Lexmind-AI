const axios = require("axios");
const logger = require("../utils/logger");

const aiClient = axios.create({
  baseURL: process.env.AI_SERVICE_URL || "http://localhost:8000",
  timeout: 120000,
  headers: { "Content-Type": "application/json" },
});

aiClient.interceptors.response.use(
  (res) => res,
  (err) => {
    const msg = err.response?.data?.detail || err.message;
    logger.error(`AI service error: ${msg}`);
    const error = new Error(`AI service: ${msg}`);
    error.statusCode = err.response?.status || 502;
    return Promise.reject(error);
  }
);

const ingestDocument = async ({ filePath, documentId, filename, uploadedBy }) => {
  const { data } = await aiClient.post("/ingest", {
    file_path: filePath,
    document_id: documentId,
    filename,
    uploaded_by: uploadedBy,
  });
  return data;
};

const analyzeDocument = async ({ filePath, documentId, filename }) => {
  const { data } = await aiClient.post("/analyze", {
    file_path: filePath,
    document_id: documentId,
    filename,
  });
  return data;
};

const searchDocuments = async ({ query, documentId = null, k = 5 }) => {
  const { data } = await aiClient.post("/search", {
    query,
    document_id: documentId,
    k,
  });
  return data;
};

const chatWithDocument = async ({ documentId, userId, question }) => {
  const { data } = await aiClient.post("/chat", {
    document_id: documentId,
    user_id: userId,
    question,
  });
  return data;
};

const clearChatHistory = async ({ documentId, userId }) => {
  const { data } = await aiClient.delete("/chat/clear", {
    data: { document_id: documentId, user_id: userId },
  });
  return data;
};

const compareDocuments = async ({ documentIdA, filenameA, documentIdB, filenameB }) => {
  const { data } = await aiClient.post("/compare", {
    document_id_a: documentIdA,
    filename_a: filenameA,
    document_id_b: documentIdB,
    filename_b: filenameB,
  });
  return data;
};

const healthCheck = async () => {
  const { data } = await aiClient.get("/health");
  return data;
};

module.exports = {
  ingestDocument,
  analyzeDocument,
  searchDocuments,
  chatWithDocument,
  clearChatHistory,
  compareDocuments,
  healthCheck,
};