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

// Fail fast on a missing or weak JWT secret rather than signing tokens with a
// guessable key. A short/default secret makes forging auth tokens trivial.
const WEAK_SECRETS = new Set([
  "jwt_secret_key",
  "secret",
  "changeme",
  "your-secret-key",
]);

const validateConfig = () => {
  const secret = (process.env.JWT_SECRET || "").trim();
  if (!secret || secret.length < 32 || WEAK_SECRETS.has(secret)) {
    logger.error(
      "JWT_SECRET must be set to a strong random value (>= 32 chars). " +
        "Generate one with: openssl rand -hex 32"
    );
    process.exit(1);
  }
};

const start = async () => {
  validateConfig();
  await connectDB();

  app.listen(PORT, () => {
    logger.info(`LexMind backend running on http://localhost:${PORT}`);
    logger.info(`AI service expected at ${process.env.AI_SERVICE_URL}`);
    logger.info(` ENV: ${process.env.NODE_ENV}`);
  });
};

start();