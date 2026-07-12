/**
 * backend/src/server.js
 * Entry point — load env, connect MongoDB, start Express.
 *
 * Run:  node src/server.js
 * Dev:  nodemon src/server.js
 */

require("dotenv").config({ path: `${__dirname}/../.env` });

const app       = require("./app");
const connectDB = require("./config/db");
const logger    = require("./utils/logger");

const PORT = process.env.PORT || 3000;

const start = async () => {
  await connectDB();

  app.listen(PORT, () => {
    logger.info(`LexMind backend running on http://localhost:${PORT}`);
    logger.info(`AI service expected at ${process.env.AI_SERVICE_URL}`);
    logger.info(` ENV: ${process.env.NODE_ENV}`);
  });
};

process.on("unhandledRejection", (reason) => {
  logger.error(`Unhandled promise rejection: ${reason?.stack || reason}`);
});

process.on("uncaughtException", (err) => {
  logger.error(`Uncaught exception: ${err.stack || err.message}`);
  process.exit(1);
});

start().catch((err) => {
  logger.error(`Failed to start server: ${err.stack || err.message}`);
  process.exit(1);
});