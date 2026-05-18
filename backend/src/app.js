const express  = require("express");
const cors     = require("cors");
const morgan   = require("morgan");
const path     = require("path");

const logger   = require("./utils/logger");

const app = express();

app.use(cors({
  origin: process.env.ALLOWED_ORIGINS?.split(",") || "*",
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