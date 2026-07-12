const express  = require("express");
const cors     = require("cors");
const morgan   = require("morgan");
const path     = require("path");

const logger   = require("./utils/logger");

const app = express();

// Restrict CORS to an explicit allowlist. In development we fall back to the
// local dev origins; in production ALLOWED_ORIGINS must be set explicitly.
// We never reflect "*" because requests carry an Authorization header.
const DEV_ORIGINS = ["http://localhost:3000", "http://localhost:5173"];
const allowedOrigins = process.env.ALLOWED_ORIGINS
  ? process.env.ALLOWED_ORIGINS.split(",").map((o) => o.trim()).filter(Boolean)
  : process.env.NODE_ENV === "production"
  ? []
  : DEV_ORIGINS;

app.use(cors({
  origin: (origin, cb) => {
    // Allow non-browser clients (no Origin header) and allowlisted origins.
    if (!origin || allowedOrigins.includes(origin)) {
      return cb(null, true);
    }
    return cb(new Error(`Origin not allowed by CORS: ${origin}`));
  },
  methods: ["GET", "POST", "PUT", "PATCH", "DELETE"],
  allowedHeaders: ["Content-Type", "Authorization"],
}));

app.use(express.json({ limit: "2mb" }));
app.use(express.urlencoded({ extended: true }));

if (process.env.NODE_ENV !== "production") {
  app.use(morgan("dev"));
}

// Routes 
// Mounted here — files built in subsequent parts
app.use("/api/auth",      require("./routes/auth"));
app.use("/api/documents", require("./routes/documents"));
app.use("/api/analysis",  require("./routes/analysis"));
app.use("/api/search",    require("./routes/search"));
app.use("/api/chat",      require("./routes/chat"));
app.use("/api/compare",   require("./routes/compare"));

app.get("/health", (req, res) => {
  res.json({ status: "ok", service: "lexmind-backend", env: process.env.NODE_ENV });
});

app.use((req, res) => {
  res.status(404).json({ success: false, message: `Route not found: ${req.method} ${req.path}` });
});

// ── Global error handler 
app.use((err, req, res, next) => {
  logger.error(err);
  res.status(err.status || 500).json({
    success: false,
    message: err.message || "Internal server error",
  });
});

module.exports = app;