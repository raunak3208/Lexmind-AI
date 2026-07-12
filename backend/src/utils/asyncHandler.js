// Wraps an async route handler so rejected promises are forwarded to the
// Express error middleware, removing repetitive try/catch + next(err) blocks.
const asyncHandler = (fn) => (req, res, next) =>
  Promise.resolve(fn(req, res, next)).catch(next);

module.exports = asyncHandler;
