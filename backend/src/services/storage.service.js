const fs = require("fs");
const path = require("path");
const logger = require("../utils/logger");

const getUploadDir = () => {
  return process.env.UPLOAD_DIR || path.join(__dirname, "../../../data/uploads");
};

const deleteFile = (filePath) => {
  try {
    if (fs.existsSync(filePath)) {
      fs.unlinkSync(filePath);
      logger.info(`Deleted file: ${filePath}`);
    }
  } catch (err) {
    logger.error(`Failed to delete file ${filePath}: ${err.message}`);
  }
};

const fileExists = (filePath) => {
  return fs.existsSync(filePath);
};

const getAbsolutePath = (filename) => {
  return path.join(getUploadDir(), filename);
};

module.exports = { deleteFile, fileExists, getAbsolutePath, getUploadDir };